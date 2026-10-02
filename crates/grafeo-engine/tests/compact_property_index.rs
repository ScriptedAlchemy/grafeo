//! A property lookup that the `LpgStore` answered from a hash index must
//! keep doing so once the rows move into the columnar base.
//!
//! `CompactStore::find_nodes_by_property` used to zone-map-prune and then
//! scan the surviving columns — linear in the table, and hundreds of
//! times slower than the index it replaced, which makes identity lookup
//! by property unusable on a compacted store. The base now carries its
//! own value hash index for the properties the source store had indexed,
//! rebuilt after a reload because it is derived state.
//!
//! ```bash
//! cargo test -p grafeo-engine \
//!     --features "lpg,gql,wal,grafeo-file,compact-store,mmap" \
//!     --test compact_property_index
//! ```

#![cfg(all(feature = "compact-store", feature = "lpg", feature = "grafeo-file"))]

use grafeo_common::types::Value;
use grafeo_engine::{Config, GrafeoDB};

const ROWS: usize = 200;

fn config(path: &std::path::Path, wal_enabled: bool) -> Config {
    Config {
        wal_enabled,
        ..Config::persistent(path)
    }
}

/// Seeds `ROWS` nodes with a unique `key`, indexes it, compacts, closes.
fn seed_indexed_and_compact(path: &std::path::Path, wal_enabled: bool) {
    let mut db = GrafeoDB::with_config(config(path, wal_enabled)).unwrap();
    db.create_property_index("key");
    {
        let session = db.session();
        for i in 0..ROWS {
            session
                .create_node_with_props(&["Row"], [("key", Value::from(format!("k-{i}")))])
                .unwrap();
        }
    }
    db.compact().expect("compact");
    db.close().unwrap();
}

fn lookup(db: &GrafeoDB, key: &str) -> Vec<u64> {
    let store = db.graph_store();
    let mut ids: Vec<u64> = store
        .find_nodes_by_property("key", &Value::from(key))
        .into_iter()
        .map(|id| id.as_u64())
        .collect();
    ids.sort_unstable();
    ids
}

/// The compacted base reports the index and answers from it, with the
/// same results the scan would give.
fn compacted_base_serves_indexed_lookups(wal_enabled: bool) {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("compact_index.grafeo");
    seed_indexed_and_compact(&path, wal_enabled);

    let db = GrafeoDB::with_config(config(&path, wal_enabled)).unwrap();
    assert!(
        db.graph_store().has_property_index("key"),
        "the reloaded base did not rebuild the property index"
    );

    for i in [0usize, 1, ROWS / 2, ROWS - 1] {
        let found = lookup(&db, &format!("k-{i}"));
        assert_eq!(found.len(), 1, "expected exactly one node for k-{i}");
    }
    assert!(
        lookup(&db, "k-nope").is_empty(),
        "a value no row carries must return nothing"
    );

    // An unindexed property still works through the scan path.
    assert!(lookup(&db, "k-0").len() == 1);
    db.close().unwrap();
}

#[test]
fn compacted_base_serves_indexed_lookups_with_wal() {
    compacted_base_serves_indexed_lookups(true);
}

#[test]
fn compacted_base_serves_indexed_lookups_without_wal() {
    compacted_base_serves_indexed_lookups(false);
}

/// The index and the scan must agree: same property, same value, same
/// answer, whether or not the property was declared indexed.
#[test]
fn indexed_and_scanned_lookups_agree() {
    let mut db = GrafeoDB::new_in_memory();
    {
        let session = db.session();
        for i in 0..ROWS {
            session
                .create_node_with_props(
                    &["Row"],
                    [
                        ("key", Value::from(format!("k-{}", i % 10))),
                        ("n", Value::Int64(i64::try_from(i).unwrap() % 7)),
                    ],
                )
                .unwrap();
        }
    }
    db.compact().expect("compact");

    // `key` is unindexed here, so this is the scan path.
    let scanned: Vec<Vec<u64>> = (0..10).map(|i| lookup(&db, &format!("k-{i}"))).collect();

    db.create_property_index("key");
    let indexed: Vec<Vec<u64>> = (0..10).map(|i| lookup(&db, &format!("k-{i}"))).collect();

    assert_eq!(
        scanned, indexed,
        "the hash index disagreed with the column scan"
    );
    assert!(scanned.iter().all(|ids| ids.len() == ROWS / 10));

    // Integer values take the same path.
    let store = db.graph_store();
    assert_eq!(
        store
            .find_nodes_by_property("n", &Value::Int64(3))
            .len()
            .min(ROWS),
        (0..ROWS).filter(|i| i % 7 == 3).count()
    );
}

/// Rows written to the overlay after the compaction still show up: the
/// base index only covers the base.
#[test]
fn overlay_rows_are_not_hidden_by_the_base_index() {
    let mut db = GrafeoDB::new_in_memory();
    db.create_property_index("key");
    {
        let session = db.session();
        session
            .create_node_with_props(&["Row"], [("key", Value::from("shared"))])
            .unwrap();
    }
    db.compact().expect("compact");
    {
        let session = db.session();
        session
            .create_node_with_props(&["Row"], [("key", Value::from("shared"))])
            .unwrap();
    }

    assert_eq!(
        lookup(&db, "shared").len(),
        2,
        "the base index masked the overlay row"
    );
}

/// Dropping an index after a compaction must reach BOTH layers: the
/// overlay `LpgStore` and the compact base's own hash index. The drop
/// used to remove only the overlay copy, so the store kept reporting —
/// and serving — an index the caller had just dropped.
#[test]
fn dropping_an_index_reaches_the_compact_base() {
    let mut db = GrafeoDB::new_in_memory();
    db.create_property_index("key");
    {
        let session = db.session();
        for i in 0..ROWS {
            session
                .create_node_with_props(&["Row"], [("key", Value::from(format!("k-{i}")))])
                .unwrap();
        }
    }
    db.compact().expect("compact");
    assert!(
        db.graph_store().has_property_index("key"),
        "the compacted base must carry the index before the drop"
    );

    assert!(
        db.drop_property_index("key"),
        "an existing index must report as removed"
    );
    assert!(
        !db.graph_store().has_property_index("key"),
        "the compact base still owns the index after the drop"
    );
    // A second drop has nothing left to remove.
    assert!(!db.drop_property_index("key"));

    // Lookups fall back to the column scan with identical answers.
    assert_eq!(lookup(&db, "k-0").len(), 1);
    assert!(lookup(&db, "k-nope").is_empty());
}

/// A compacted database used to write no catalog section at all, so
/// closing one dropped its schema definitions and index names on the
/// floor. Both must come back.
#[test]
#[cfg(feature = "gql")]
fn a_compacted_database_keeps_its_catalog() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("compact_catalog.grafeo");

    {
        let mut db = GrafeoDB::with_config(config(&path, true)).unwrap();
        {
            let session = db.session();
            session
                .execute("CREATE NODE TYPE Person (name STRING NOT NULL)")
                .unwrap();
            session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        }
        db.create_property_index("name");
        db.compact().expect("compact");
        db.close().unwrap();
    }

    let db = GrafeoDB::with_config(config(&path, true)).unwrap();
    let session = db.session();
    let types: Vec<String> = session
        .execute("SHOW NODE TYPES")
        .unwrap()
        .rows()
        .iter()
        .filter_map(|r| match &r[0] {
            Value::String(s) => Some(s.to_string()),
            _ => None,
        })
        .collect();
    assert!(
        types.contains(&"Person".to_string()),
        "compacted database lost its node type: {types:?}"
    );
    drop(session);
    assert!(
        db.has_property_index("name"),
        "compacted database lost its property index"
    );
    db.close().unwrap();
}
