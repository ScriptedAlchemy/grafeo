//! Torn checkpoints over a store with a populated HNSW index.
//!
//! The byte-level tear suite (`grafeo-storage/tests/torn_checkpoint.rs`)
//! proves a kill at any byte of a checkpoint leaves the container openable
//! at the previous generation — including its `VectorStore` section. What
//! it cannot prove is that the *engine* comes back whole: that the catalog
//! restore loop re-registers the persisted index, the topology loads, WAL
//! replay reapplies the writes the torn checkpoint failed to persist, and
//! vector search serves — whichever side of the header flip the kill fell
//! on. These tests kill the close-time checkpoint at every injection point
//! in the production write path and assert exactly that.
//!
//! ```bash
//! cargo test -p grafeo-engine \
//!     --features "lpg,gql,wal,grafeo-file,vector-index,testing-crash-injection" \
//!     --test torn_vector_checkpoint
//! ```

#![cfg(all(
    feature = "vector-index",
    feature = "lpg",
    feature = "grafeo-file",
    feature = "wal",
    feature = "testing-crash-injection"
))]

use std::panic::AssertUnwindSafe;

use grafeo_common::testing::crash::{CrashResult, with_crash_at};
use grafeo_common::types::Value;
use grafeo_engine::{Config, GrafeoDB};

const LABEL: &str = "Doc";
const PROPERTY: &str = "embedding";
const DIMS: usize = 8;
const ROWS: usize = 120;
const EXTRA_ROWS: usize = 15;

fn config(path: &std::path::Path) -> Config {
    Config::persistent(path)
}

/// Deterministic, well-spread corpus (same construction as the reopen
/// suite, smaller so every crash point can afford its own store).
fn embedding(row: usize) -> Vec<f32> {
    (0..DIMS)
        .map(|d| {
            let x = (row * 31 + d * 17) as f32;
            (x * 0.017).sin() + (row as f32) * 0.001 * (d as f32 + 1.0)
        })
        .collect()
}

fn insert_rows(db: &GrafeoDB, rows: std::ops::Range<usize>) {
    let session = db.session();
    for row in rows {
        session
            .create_node_with_props(
                &[LABEL],
                [
                    ("row", Value::Int64(i64::try_from(row).unwrap())),
                    (PROPERTY, Value::from(embedding(row).as_slice())),
                ],
            )
            .unwrap();
    }
}

/// Generation A: a durable store whose HNSW index covers every row.
fn seed(path: &std::path::Path) {
    let db = GrafeoDB::with_config(config(path)).unwrap();
    insert_rows(&db, 0..ROWS);
    db.create_vector_index(LABEL, PROPERTY, Some(DIMS), Some("cosine"), None, None, None)
        .expect("build the index once");
    assert_eq!(db.vector_index_len(LABEL, PROPERTY), Some(ROWS));
    db.close().unwrap();
}

#[test]
fn checkpoint_killed_over_a_populated_index_reopens_and_serves_search() {
    // Kill the close-time checkpoint at every injection point the write
    // path exposes (export, section-data, directory, fsync phases). Points
    // past the path's call count fire nothing; those runs close cleanly
    // and still must reopen whole.
    let mut kills = 0usize;
    for crash_point in 1..=10 {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("torn_vectors.grafeo");
        seed(&path);

        // Dirty the store beyond generation A: WAL-logged rows the torn
        // checkpoint may or may not have persisted, depending on where
        // the kill lands relative to the header flip.
        let db = GrafeoDB::with_config(config(&path)).unwrap();
        assert_eq!(
            db.vector_index_len(LABEL, PROPERTY),
            Some(ROWS),
            "crash_point={crash_point}: generation A must restore before the tear"
        );
        insert_rows(&db, ROWS..ROWS + EXTRA_ROWS);

        let crashed = {
            let db = AssertUnwindSafe(db);
            let result = with_crash_at(crash_point, move || {
                let db = db;
                db.close().unwrap();
            });
            matches!(result, CrashResult::Crashed)
        };
        kills += usize::from(crashed);

        // Reopen: whichever generation the kill left live, the store must
        // open without a CRC error, restore the index, and have every row
        // — the WAL replays whatever the torn checkpoint failed to write.
        let db = GrafeoDB::with_config(config(&path)).unwrap_or_else(|e| {
            panic!("crash_point={crash_point} (crashed={crashed}): reopen failed: {e}")
        });
        assert!(
            db.graph_store().has_vector_index(LABEL, PROPERTY),
            "crash_point={crash_point}: torn checkpoint lost the vector index"
        );
        let indexed = db
            .vector_index_len(LABEL, PROPERTY)
            .expect("index present after torn checkpoint");
        assert!(
            indexed >= ROWS,
            "crash_point={crash_point}: restored index covers {indexed} rows, \
             fewer than generation A's {ROWS}"
        );

        let session = db.session();
        let result = session.execute("MATCH (d:Doc) RETURN count(d)").unwrap();
        let count = match &result.rows()[0][0] {
            Value::Int64(n) => *n,
            other => panic!("count(d) returned {other:?}"),
        };
        assert_eq!(
            count,
            i64::try_from(ROWS + EXTRA_ROWS).unwrap(),
            "crash_point={crash_point}: rows lost across the torn checkpoint"
        );
        drop(session);

        let hits = db
            .vector_search(LABEL, PROPERTY, &embedding(7), 5, Some(64), None)
            .expect("vector search after torn checkpoint");
        assert_eq!(
            hits.len(),
            5,
            "crash_point={crash_point}: search answered nothing after the tear"
        );
        db.close().unwrap();
    }
    assert!(
        kills >= 4,
        "only {kills} of 10 injection points fired: the checkpoint path \
         lost its crash points and this test went vacuous"
    );
}
