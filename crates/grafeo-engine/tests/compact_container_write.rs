//! `GrafeoDB::write_compact_container`: a database container written from a
//! [`CompactStore`] built incrementally from a row stream, with no live LPG
//! and no WAL — one durable write of the same three sections a compacted
//! database's `close` produces.
//!
//! ```bash
//! cargo test -p grafeo-engine \
//!     --features "lpg,gql,wal,grafeo-file,compact-store,mmap,testing-crash-injection" \
//!     --test compact_container_write
//! ```
//!
//! [`CompactStore`]: grafeo_core::graph::compact::CompactStore

#![cfg(all(feature = "compact-store", feature = "lpg", feature = "grafeo-file"))]
// reason: fixture indices are small, known values.
#![allow(clippy::cast_possible_truncation)]

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use grafeo_common::storage::SectionType;
use grafeo_common::types::{EdgeId, NodeId, PropertyKey, Value};
use grafeo_core::graph::Direction;
use grafeo_core::graph::compact::{CompactStore, IncrementalCompactStoreBuilder};
use grafeo_engine::{Config, GrafeoDB};
use grafeo_storage::file::GrafeoFileManager;

const PEOPLE: u64 = 1_200;
/// Node ids start above zero so a reopened overlay that restarted its
/// allocator at zero would collide visibly.
const FIRST_ID: u64 = 1_000;

fn key(name: &str) -> PropertyKey {
    PropertyKey::new(name)
}

/// The fixture rows: `PEOPLE` people with a unique `key`, each linked to
/// the next by `KNOWS`, plus one multi-label `City|Place` node every
/// person `LIVES_IN`.
fn fixture_store() -> Arc<CompactStore> {
    let mut builder = IncrementalCompactStoreBuilder::new();
    let city = NodeId(FIRST_ID + PEOPLE);
    builder
        .push_node(
            city,
            ["Place", "City"],
            [(&key("name"), &Value::from("Lyon"))],
        )
        .unwrap();
    for index in 0..PEOPLE {
        let properties = [
            (key("key"), Value::from(format!("person:{index:05}"))),
            (key("rank"), Value::Int64(i64::try_from(index).unwrap())),
            (
                key("payload"),
                Value::Bytes(vec![(index % 251) as u8; 16].into()),
            ),
        ];
        builder
            .push_node(
                NodeId(FIRST_ID + index),
                ["Person"],
                properties.iter().map(|(k, v)| (k, v)),
            )
            .unwrap();
    }
    for index in 0..PEOPLE {
        let src = NodeId(FIRST_ID + index);
        let dst = NodeId(FIRST_ID + (index + 1) % PEOPLE);
        let since = [(
            key("since"),
            Value::Int64(2_000 + i64::try_from(index).unwrap()),
        )];
        builder
            .push_edge(
                EdgeId(index),
                "KNOWS",
                src,
                dst,
                since.iter().map(|(k, v)| (k, v)),
            )
            .unwrap();
        builder
            .push_edge(
                EdgeId(PEOPLE + index),
                "LIVES_IN",
                src,
                city,
                std::iter::empty(),
            )
            .unwrap();
    }
    Arc::new(builder.finish().unwrap())
}

fn write_fixture(path: &Path) {
    GrafeoDB::write_compact_container(path, fixture_store(), ["key".to_owned()]).unwrap();
}

/// Section type -> payload bytes of a closed container.
fn section_payloads(path: &Path) -> BTreeMap<u8, Vec<u8>> {
    let fm = GrafeoFileManager::open_read_only(path).unwrap();
    let directory = fm.read_section_directory().unwrap().unwrap();
    directory
        .entries()
        .iter()
        .map(|entry| {
            (
                entry.section_type as u8,
                fm.read_section_data(entry).unwrap(),
            )
        })
        .collect()
}

/// Every fixture row reads back under its original id; `overlay_nodes` is
/// how many rows a writable reopen added on top of the base.
fn assert_fixture_readable(db: &GrafeoDB, overlay_nodes: usize) {
    let store = db.graph_store();
    assert_eq!(
        store.node_count(),
        usize::try_from(PEOPLE).unwrap() + 1 + overlay_nodes
    );
    assert_eq!(store.edge_count(), usize::try_from(PEOPLE).unwrap() * 2);

    let first = store
        .get_node(NodeId(FIRST_ID))
        .expect("original id resolves");
    assert_eq!(
        first.properties.get(&key("key")),
        Some(&Value::from("person:00000"))
    );
    assert_eq!(first.properties.get(&key("rank")), Some(&Value::Int64(0)));
    assert_eq!(
        first.properties.get(&key("payload")),
        Some(&Value::Bytes(vec![0u8; 16].into()))
    );
    let last = store
        .get_node(NodeId(FIRST_ID + PEOPLE - 1))
        .expect("last original id resolves");
    assert_eq!(
        last.properties.get(&key("key")),
        Some(&Value::from(format!("person:{:05}", PEOPLE - 1)))
    );

    let city = store
        .get_node(NodeId(FIRST_ID + PEOPLE))
        .expect("multi-label node resolves");
    assert_eq!(
        city.properties.get(&key("name")),
        Some(&Value::from("Lyon"))
    );

    let edge = store
        .get_edge(EdgeId(7))
        .expect("original edge id resolves");
    assert_eq!(edge.edge_type.as_str(), "KNOWS");
    assert_eq!(
        (edge.src, edge.dst),
        (NodeId(FIRST_ID + 7), NodeId(FIRST_ID + 8))
    );
    assert_eq!(
        edge.properties.get(&key("since")),
        Some(&Value::Int64(2_007))
    );
    let incoming = store.edges_from(NodeId(FIRST_ID + PEOPLE), Direction::Incoming);
    assert_eq!(incoming.len(), usize::try_from(PEOPLE).unwrap());

    // The catalog named `key` as indexed, so the reopened base answers the
    // lookup from its hash index rather than a scan.
    assert!(store.has_property_index("key"));
    let hits = store.find_nodes_by_property("key", &Value::from("person:00042"));
    assert_eq!(hits, vec![NodeId(FIRST_ID + 42)]);
}

#[test]
fn written_container_reopens_writable_and_read_only_with_every_row() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("direct.grafeo");
    write_fixture(&path);
    let written = section_payloads(&path);
    assert_eq!(
        written.keys().copied().collect::<Vec<_>>(),
        {
            let mut kinds = vec![
                SectionType::CompactStore as u8,
                SectionType::LpgStore as u8,
                SectionType::Catalog as u8,
            ];
            kinds.sort_unstable();
            kinds
        },
        "exactly the three sections a compacted close writes"
    );
    let mut sidecar_wal = path.as_os_str().to_owned();
    sidecar_wal.push(".wal");
    assert!(
        !Path::new(&sidecar_wal).exists(),
        "a direct write never opens a WAL"
    );

    {
        let db = GrafeoDB::with_config(Config::read_only(&path)).unwrap();
        assert_fixture_readable(&db, 0);
        db.close().unwrap();
    }
    assert_eq!(
        section_payloads(&path),
        written,
        "a read-only open and close leaves the sections untouched"
    );

    let db = GrafeoDB::with_config(Config {
        wal_enabled: false,
        ..Config::persistent(&path)
    })
    .unwrap();
    assert_fixture_readable(&db, 0);
    // The overlay allocator was seeded above the base: a fresh row gets a
    // fresh id and shadows nothing.
    let fresh = db.graph_store_mut().unwrap().create_node(&["Person"]);
    assert!(
        fresh.0 > FIRST_ID + PEOPLE,
        "fresh id {fresh:?} collides with the base"
    );
    assert_eq!(
        db.graph_store().node_count(),
        usize::try_from(PEOPLE).unwrap() + 2
    );
    db.close().unwrap();

    let db = GrafeoDB::with_config(Config::read_only(&path)).unwrap();
    assert!(db.graph_store().get_node(fresh).is_some());
    assert_fixture_readable(&db, 1);
}

#[test]
fn identical_rows_write_identical_sections() {
    // Enough property indexes that hash-map iteration order would almost
    // surely differ from insertion order; the catalog must persist them in
    // one canonical order regardless of how the caller listed them.
    let indexes: Vec<String> = (0..12).map(|i| format!("index_{i}")).collect();
    let mut reversed = indexes.clone();
    reversed.reverse();
    let dir = tempfile::TempDir::new().unwrap();
    let first = dir.path().join("first.grafeo");
    let second = dir.path().join("second.grafeo");
    GrafeoDB::write_compact_container(&first, fixture_store(), indexes).unwrap();
    GrafeoDB::write_compact_container(&second, fixture_store(), reversed).unwrap();
    let first = section_payloads(&first);
    assert!(first.values().all(|bytes| !bytes.is_empty()));
    assert_eq!(first, section_payloads(&second));
}

#[test]
fn write_refuses_an_existing_path_and_leaves_it_untouched() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("direct.grafeo");
    write_fixture(&path);
    let before = std::fs::read(&path).unwrap();

    let mut builder = IncrementalCompactStoreBuilder::new();
    builder
        .push_node(NodeId(1), ["Other"], std::iter::empty())
        .unwrap();
    let other = Arc::new(builder.finish().unwrap());
    let err = GrafeoDB::write_compact_container(&path, other, std::iter::empty()).unwrap_err();
    assert!(err.to_string().contains("already exists"), "{err}");
    assert_eq!(std::fs::read(&path).unwrap(), before);
}

#[test]
fn abandoned_build_leaves_nothing_on_disk() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("abandoned.grafeo");
    {
        let mut builder = IncrementalCompactStoreBuilder::new();
        for index in 0..100u64 {
            builder
                .push_node(NodeId(index), ["Person"], std::iter::empty())
                .unwrap();
        }
        // The producer stops here — cancellation, error, whatever — and
        // the builder is dropped before any write.
        drop(builder);
    }
    assert!(std::fs::read_dir(dir.path()).unwrap().next().is_none());
    write_fixture(&path);
    GrafeoDB::with_config(Config::read_only(&path)).unwrap();
}

/// No crash point inside the section writer yields a partially populated
/// container: the file left behind is either the empty container `create`
/// wrote first (opens with no rows) or the finished one (opens with every
/// row), and the next attempt at a fresh path succeeds.
#[cfg(feature = "testing-crash-injection")]
#[test]
fn crash_during_write_never_yields_a_partial_container() {
    use grafeo_common::testing::crash::{CrashResult, with_crash_at};

    let mut crashed_points = 0;
    for crash_after in 1..=8 {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("crash.grafeo");
        let store = fixture_store();
        let result = with_crash_at(
            crash_after,
            std::panic::AssertUnwindSafe(|| {
                GrafeoDB::write_compact_container(&path, Arc::clone(&store), ["key".to_owned()])
            }),
        );
        match result {
            CrashResult::Completed(outcome) => {
                outcome.unwrap();
                let db = GrafeoDB::with_config(Config::read_only(&path)).unwrap();
                assert_fixture_readable(&db, 0);
            }
            CrashResult::Crashed => {
                crashed_points += 1;
                let db = GrafeoDB::with_config(Config::read_only(&path)).unwrap();
                if db.graph_store().node_count() == 0 {
                    assert_eq!(db.graph_store().edge_count(), 0);
                    assert!(!db.graph_store().has_property_index("key"));
                } else {
                    assert_fixture_readable(&db, 0);
                }
                drop(db);
                let retry = dir.path().join("retry.grafeo");
                GrafeoDB::write_compact_container(&retry, store, ["key".to_owned()]).unwrap();
                let db = GrafeoDB::with_config(Config::read_only(&retry)).unwrap();
                assert_fixture_readable(&db, 0);
            }
            _ => unreachable!("CrashResult is complete or crashed"),
        }
    }
    assert!(
        crashed_points >= 3,
        "the writer exposes several crash points; only {crashed_points} fired"
    );
}
