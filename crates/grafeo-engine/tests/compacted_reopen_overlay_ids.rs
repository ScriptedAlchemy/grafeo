//! Regression tests for the overlay ID allocator after reopening a
//! database whose nodes and edges live in a compacted columnar base.
//!
//! Pre-fix behaviour: the base's IDs are preserved in the
//! `CompactStore`, but the overlay `LpgStore` rebuilt on the open path
//! derives `next_node_id` / `next_edge_id` from the rows it deserializes
//! — and a fully compacted database has an *empty* overlay. The
//! allocator therefore restarted at zero and the first post-reopen
//! insert shadowed the base's first node: reads returned the overlay
//! row, and deleting the "new" node tombstoned the base row that shared
//! its ID — losing a node nobody touched.
//!
//! ```bash
//! cargo test -p grafeo-engine \
//!     --features "lpg,gql,wal,grafeo-file,compact-store,mmap" \
//!     --test compacted_reopen_overlay_ids
//! ```

#![cfg(all(feature = "compact-store", feature = "lpg", feature = "grafeo-file"))]

use grafeo_common::types::Value;
use grafeo_engine::{Config, GrafeoDB};

/// Labels seeded before compaction, one node each.
const BASE_LABELS: [&str; 3] = ["Alpha", "Beta", "Gamma"];

/// Label created (and, in one test, deleted) after the reopen.
const OVERLAY_LABEL: &str = "Delta";

/// Builds a config for `path` with the WAL either on or off, so every
/// case below runs against both durability modes.
fn config(path: &std::path::Path, wal_enabled: bool) -> Config {
    Config {
        wal_enabled,
        ..Config::persistent(path)
    }
}

/// Sorted `label -> live node count` census of a database.
fn census(db: &GrafeoDB) -> Vec<(String, usize)> {
    let store = db.graph_store();
    let mut per_label: Vec<(String, usize)> = store
        .all_labels()
        .into_iter()
        .map(|label| {
            let count = store.nodes_by_label(&label).len();
            (label, count)
        })
        .collect();
    per_label.sort();
    per_label
}

fn count_of(census: &[(String, usize)], label: &str) -> usize {
    census
        .iter()
        .find(|(name, _)| name == label)
        .map_or(0, |(_, count)| *count)
}

/// Seeds one node per base label, compacts, and closes.
fn seed_compacted_base(path: &std::path::Path, wal_enabled: bool) -> Vec<u64> {
    let mut db = GrafeoDB::with_config(config(path, wal_enabled)).unwrap();
    let mut ids = Vec::new();
    {
        let session = db.session();
        for label in BASE_LABELS {
            let id = session
                .create_node_with_props(&[label], [("k", Value::from(label))])
                .unwrap();
            ids.push(id.as_u64());
        }
    }
    db.compact().expect("compact");
    db.close().unwrap();
    ids
}

/// A node inserted after reopening a compacted base must not reuse an ID
/// that the base already owns.
fn post_reopen_insert_gets_a_fresh_id(wal_enabled: bool) {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("overlay_ids.grafeo");
    let base_ids = seed_compacted_base(&path, wal_enabled);

    let db = GrafeoDB::with_config(config(&path, wal_enabled)).unwrap();
    let fresh = {
        let session = db.session();
        session
            .create_node_with_props(&[OVERLAY_LABEL], [("k", Value::from(OVERLAY_LABEL))])
            .unwrap()
    };

    assert!(
        !base_ids.contains(&fresh.as_u64()),
        "overlay allocated id {} which the compacted base already owns ({base_ids:?})",
        fresh.as_u64()
    );

    // Every base node is still readable, and the new one is too.
    let live = census(&db);
    for label in BASE_LABELS {
        assert_eq!(
            count_of(&live, label),
            1,
            "base label `{label}` was shadowed by the overlay insert"
        );
    }
    assert_eq!(count_of(&live, OVERLAY_LABEL), 1);
    db.close().unwrap();
}

#[test]
fn post_reopen_insert_gets_a_fresh_id_with_wal() {
    post_reopen_insert_gets_a_fresh_id(true);
}

#[test]
fn post_reopen_insert_gets_a_fresh_id_without_wal() {
    post_reopen_insert_gets_a_fresh_id(false);
}

/// Deleting an overlay node created after a compacted reopen must not
/// take an unrelated base node with it, in memory or on disk.
fn overlay_delete_leaves_the_base_intact(wal_enabled: bool) {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("overlay_delete.grafeo");
    seed_compacted_base(&path, wal_enabled);

    let after_delete = {
        let db = GrafeoDB::with_config(config(&path, wal_enabled)).unwrap();
        let transient = {
            let session = db.session();
            session
                .create_node_with_props(&[OVERLAY_LABEL], [("k", Value::from(OVERLAY_LABEL))])
                .unwrap()
        };
        {
            let session = db.session();
            session
                .execute(&format!(
                    "MATCH (n:{OVERLAY_LABEL}) WHERE id(n) = {} DELETE n",
                    transient.as_u64()
                ))
                .unwrap();
        }
        let after_delete = census(&db);
        db.close().unwrap();
        after_delete
    };

    let reopened = {
        let db = GrafeoDB::with_config(config(&path, wal_enabled)).unwrap();
        let reopened = census(&db);
        db.close().unwrap();
        reopened
    };

    for label in BASE_LABELS {
        assert_eq!(
            count_of(&after_delete, label),
            1,
            "deleting an overlay node dropped base label `{label}` from the live store"
        );
        assert_eq!(
            count_of(&reopened, label),
            1,
            "base label `{label}` did not survive the reopen"
        );
    }
    assert_eq!(
        count_of(&after_delete, OVERLAY_LABEL),
        0,
        "the overlay node was supposed to be deleted"
    );
}

#[test]
fn overlay_delete_leaves_the_base_intact_with_wal() {
    overlay_delete_leaves_the_base_intact(true);
}

#[test]
fn overlay_delete_leaves_the_base_intact_without_wal() {
    overlay_delete_leaves_the_base_intact(false);
}

/// The edge allocator needs the same treatment: an edge inserted after a
/// compacted reopen must not reuse a base edge's ID.
fn post_reopen_edge_gets_a_fresh_id(wal_enabled: bool) {
    use grafeo_core::graph::Direction;

    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("overlay_edge_ids.grafeo");

    let base_edge_ids: Vec<u64> = {
        let mut db = GrafeoDB::with_config(config(&path, wal_enabled)).unwrap();
        {
            let session = db.session();
            session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
            session.execute("INSERT (:Person {name: 'Gus'})").unwrap();
            session
                .execute(
                    "MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'}) \
                     INSERT (a)-[:KNOWS]->(b)",
                )
                .unwrap();
        }
        let store = db.graph_store();
        let ids: Vec<u64> = store
            .node_ids()
            .into_iter()
            .flat_map(|nid| store.edges_from(nid, Direction::Outgoing))
            .map(|(_, eid)| eid.as_u64())
            .collect();
        drop(store);
        db.compact().expect("compact");
        db.close().unwrap();
        ids
    };
    assert!(!base_edge_ids.is_empty(), "expected a seeded base edge");

    let db = GrafeoDB::with_config(config(&path, wal_enabled)).unwrap();
    {
        let session = db.session();
        session
            .execute(
                "MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'}) \
                 INSERT (a)-[:LIKES]->(b)",
            )
            .unwrap();
    }

    let store = db.graph_store();
    let all_edges: Vec<u64> = store
        .node_ids()
        .into_iter()
        .flat_map(|nid| store.edges_from(nid, Direction::Outgoing))
        .map(|(_, eid)| eid.as_u64())
        .collect();
    drop(store);

    let mut deduped = all_edges.clone();
    deduped.sort_unstable();
    deduped.dedup();
    assert_eq!(
        deduped.len(),
        2,
        "expected the base edge and a distinct overlay edge, got {all_edges:?}"
    );
    db.close().unwrap();
}

#[test]
fn post_reopen_edge_gets_a_fresh_id_with_wal() {
    post_reopen_edge_gets_a_fresh_id(true);
}

#[test]
fn post_reopen_edge_gets_a_fresh_id_without_wal() {
    post_reopen_edge_gets_a_fresh_id(false);
}
