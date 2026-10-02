//! Restoring a persisted HNSW index must be cheaper than rebuilding it.
//!
//! This is the claim the whole change rests on. A restore that merely
//! *worked* while costing what a rebuild costs would have bought
//! nothing; the point is that a reopen stops paying an O(N log N)
//! re-link and starts paying a linear topology decode.
//!
//! Run with `--nocapture` to see the two numbers:
//!
//! ```bash
//! cargo test -p grafeo-engine \
//!     --features "lpg,gql,wal,grafeo-file,vector-index" \
//!     --test vector_index_restore_cost -- --nocapture
//! ```

#![cfg(all(feature = "vector-index", feature = "lpg", feature = "grafeo-file"))]

use std::time::Instant;

use grafeo_common::types::Value;
use grafeo_engine::{Config, GrafeoDB};

const LABEL: &str = "Doc";
const PROPERTY: &str = "embedding";
const DIMS: usize = 64;
const ROWS: usize = 5_000;

fn embedding(row: usize) -> Vec<f32> {
    (0..DIMS)
        .map(|d| {
            let x = (row * 37 + d * 11) as f32;
            (x * 0.013).sin() * 0.7 + (x * 0.0031).cos() * 0.3
        })
        .collect()
}

fn seed_rows(db: &GrafeoDB) {
    let session = db.session();
    for row in 0..ROWS {
        session
            .create_node_with_props(
                &[LABEL],
                [
                    ("row", Value::Int64(i64::try_from(row).unwrap())),
                    ("embedding", Value::from(embedding(row).as_slice())),
                ],
            )
            .unwrap();
    }
}

/// A reopen that restores the index is materially faster than a reopen
/// that rebuilds it, and both answer the same queries.
#[test]
fn restore_is_cheaper_than_rebuild() {
    let dir = tempfile::TempDir::new().unwrap();

    // Lane A: rows only. Reopen, then build the index - the cost every
    // open used to pay.
    let rebuild_path = dir.path().join("rebuild.grafeo");
    {
        let db = GrafeoDB::with_config(Config::persistent(&rebuild_path)).unwrap();
        seed_rows(&db);
        db.close().unwrap();
    }
    let db = GrafeoDB::with_config(Config::persistent(&rebuild_path)).unwrap();
    assert!(!db.graph_store().has_vector_index(LABEL, PROPERTY));
    let rebuild_started = Instant::now();
    db.create_vector_index(LABEL, PROPERTY, Some(DIMS), Some("cosine"), None, None, None)
        .unwrap();
    let rebuild = rebuild_started.elapsed();
    assert_eq!(db.vector_index_len(LABEL, PROPERTY), Some(ROWS));
    let rebuilt_neighbours = neighbours(&db);
    let heap_bytes = db.vector_index_heap_bytes(LABEL, PROPERTY).unwrap();
    db.close().unwrap();

    // Lane B: rows plus a persisted index. The reopen restores it.
    let restore_path = dir.path().join("restore.grafeo");
    {
        let db = GrafeoDB::with_config(Config::persistent(&restore_path)).unwrap();
        seed_rows(&db);
        db.create_vector_index(LABEL, PROPERTY, Some(DIMS), Some("cosine"), None, None, None)
            .unwrap();
        db.close().unwrap();
    }
    let restore_started = Instant::now();
    let db = GrafeoDB::with_config(Config::persistent(&restore_path)).unwrap();
    let restore = restore_started.elapsed();
    assert_eq!(
        db.vector_index_len(LABEL, PROPERTY),
        Some(ROWS),
        "the reopen did not restore the index"
    );
    let restored_neighbours = neighbours(&db);
    db.close().unwrap();

    println!(
        "{ROWS} vectors x {DIMS} dims, {} KiB of topology\n  rebuild (index build alone): {rebuild:?}\n  restore (whole database open): {restore:?}",
        heap_bytes / 1024
    );

    assert_eq!(
        restored_neighbours, rebuilt_neighbours,
        "restore and rebuild must agree on every neighbour"
    );
    // The restore figure includes opening the entire database - every
    // other section, the rows, the property indexes - while the rebuild
    // figure is the index build on its own, on a database that is
    // already open. Even so it should not be close.
    assert!(
        restore * 2 < rebuild,
        "restoring the index ({restore:?}) was not meaningfully cheaper than rebuilding it ({rebuild:?})"
    );
}

fn neighbours(db: &GrafeoDB) -> Vec<Vec<u64>> {
    (0..24)
        .map(|i| {
            db.vector_search(
                LABEL,
                PROPERTY,
                &embedding(i * 173 % ROWS),
                10,
                Some(64),
                None,
            )
            .expect("vector search")
            .into_iter()
            .map(|(id, _)| id.as_u64())
            .collect()
        })
        .collect()
}
