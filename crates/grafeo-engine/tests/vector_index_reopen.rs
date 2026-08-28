//! A vector index that was built once must survive a close and reopen.
//!
//! The `VectorStore` section has always been written; it was never read
//! back, because the restore loop was guarded on the store already
//! having index entries and a cold open has none. So every reopen paid
//! a full HNSW rebuild and vector search was unavailable until it
//! finished.
//!
//! What these tests pin down:
//!
//! - a reopened index is present without anything asking for a rebuild;
//! - it returns the *same neighbours* as the index that was persisted,
//!   which is the only interesting claim - a restored index that
//!   answered differently would be a rebuild wearing a disguise;
//! - the two ways a restore can be incomplete leave the index absent
//!   rather than present-and-empty, because an empty index reports
//!   `has_vector_index() == true` and silently answers nothing;
//! - the binding token round-trips, so a caller can tell a restored
//!   index that still matches its data from one that has drifted.
//!
//! ```bash
//! cargo test -p grafeo-engine \
//!     --features "lpg,gql,wal,grafeo-file,vector-index" \
//!     --test vector_index_reopen
//! ```

#![cfg(all(feature = "vector-index", feature = "lpg", feature = "grafeo-file"))]

use grafeo_common::types::Value;
use grafeo_engine::{Config, GrafeoDB};

const LABEL: &str = "Doc";
const PROPERTY: &str = "embedding";
const DIMS: usize = 8;
const ROWS: usize = 400;

fn config(path: &std::path::Path) -> Config {
    Config::persistent(path)
}

/// A deterministic, well-spread corpus. Distinct enough that nearest
/// neighbours are unambiguous, so a neighbour-set comparison is a real
/// assertion rather than a coin flip between equidistant rows.
fn embedding(row: usize) -> Vec<f32> {
    (0..DIMS)
        .map(|d| {
            let x = (row * 31 + d * 17) as f32;
            (x * 0.017).sin() + (row as f32) * 0.001 * (d as f32 + 1.0)
        })
        .collect()
}

fn probes() -> Vec<Vec<f32>> {
    (0..16).map(|i| embedding(i * 23 % ROWS)).collect()
}

fn seed(path: &std::path::Path) {
    let db = GrafeoDB::with_config(config(path)).unwrap();
    {
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
    db.create_vector_index(LABEL, PROPERTY, Some(DIMS), Some("cosine"), None, None, None)
        .expect("build the index once");
    assert!(db.graph_store().has_vector_index(LABEL, PROPERTY));
    assert_eq!(
        db.vector_index_len(LABEL, PROPERTY),
        Some(ROWS),
        "the seeded index must cover every row before it is persisted"
    );
    db.close().unwrap();
}

/// Neighbour ids for every probe, as the caller would see them.
fn neighbours(db: &GrafeoDB) -> Vec<Vec<u64>> {
    probes()
        .iter()
        .map(|query| {
            db.vector_search(LABEL, PROPERTY, query, 10, Some(64), None)
                .expect("vector search")
                .into_iter()
                .map(|(id, _)| id.as_u64())
                .collect()
        })
        .collect()
}

/// The whole point: reopen answers vector queries with no rebuild.
#[test]
fn reopen_serves_vector_search_without_rebuilding() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("vectors.grafeo");
    seed(&path);

    let db = GrafeoDB::with_config(config(&path)).unwrap();
    assert!(
        db.graph_store().has_vector_index(LABEL, PROPERTY),
        "the reopened store lost its vector index"
    );

    assert_eq!(
        db.vector_index_len(LABEL, PROPERTY).expect("restored index"),
        ROWS,
        "the restored topology must cover every indexed row"
    );

    let hits = db
        .vector_search(LABEL, PROPERTY, &embedding(7), 5, Some(64), None)
        .expect("search the restored index");
    assert_eq!(hits.len(), 5, "restored index answered nothing");
    db.close().unwrap();
}

/// The restored index and the index that was persisted return the same
/// neighbours for every probe. HNSW search is deterministic given the
/// same topology, vectors, and `ef`, so anything short of an exact match
/// means the topology did not round-trip faithfully.
#[test]
fn restored_index_returns_the_same_neighbours_as_the_persisted_one() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("vectors.grafeo");

    let before = {
        let db = GrafeoDB::with_config(config(&path)).unwrap();
        {
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
        db.create_vector_index(LABEL, PROPERTY, Some(DIMS), Some("cosine"), None, None, None)
            .unwrap();
        let neighbours = neighbours(&db);
        db.close().unwrap();
        neighbours
    };

    let db = GrafeoDB::with_config(config(&path)).unwrap();
    let after = neighbours(&db);
    assert_eq!(
        after, before,
        "the restored index returned different neighbours than the one it was serialized from"
    );
    db.close().unwrap();
}

/// A caller's binding token survives the reopen alongside the index.
#[test]
fn binding_token_round_trips_with_the_index() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("vectors.grafeo");
    seed(&path);

    {
        let db = GrafeoDB::with_config(config(&path)).unwrap();
        assert!(
            db.set_vector_index_binding(LABEL, PROPERTY, "generation-abc123"),
            "stamping an existing index must succeed"
        );
        assert!(
            !db.set_vector_index_binding(LABEL, "nope", "x"),
            "stamping a pair with no index must report failure"
        );
        db.close().unwrap();
    }

    let db = GrafeoDB::with_config(config(&path)).unwrap();
    assert_eq!(
        db.vector_index_binding(LABEL, PROPERTY).as_deref(),
        Some("generation-abc123"),
        "the binding token did not survive the reopen"
    );
    db.close().unwrap();
}

/// An index with no persisted topology must come back absent, not
/// present-and-empty. Present-and-empty is the silent failure: the
/// store reports an index, and every search returns nothing.
#[test]
fn an_index_over_no_vectors_is_left_absent_for_rebuild() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("empty.grafeo");

    {
        let db = GrafeoDB::with_config(config(&path)).unwrap();
        // Explicit dimensions with no matching rows: grafeo creates the
        // index empty, and it is serialized with an empty topology.
        db.create_vector_index(LABEL, PROPERTY, Some(DIMS), Some("cosine"), None, None, None)
            .expect("empty index");
        assert!(db.graph_store().has_vector_index(LABEL, PROPERTY));
        db.close().unwrap();
    }

    let db = GrafeoDB::with_config(config(&path)).unwrap();
    assert!(
        !db.graph_store().has_vector_index(LABEL, PROPERTY),
        "an index with no restored topology must report absent so the caller rebuilds"
    );
    db.close().unwrap();
}

/// Restore is faithful, not lossy: the reopened index covers exactly
/// what the persisted one covered.
///
/// Session-created rows never enter a live vector index in the first
/// place - only grafeo's own property and batch write paths sync it -
/// so a corpus written through a session after the build is missing
/// from the index before the close as much as after the reopen. That
/// gap is grafeo's and is unchanged here. What this pins down is that
/// the reopen neither invents coverage nor drops any.
#[test]
fn restore_reproduces_the_persisted_coverage_exactly() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("vectors.grafeo");
    seed(&path);

    let live_len = {
        let db = GrafeoDB::with_config(config(&path)).unwrap();
        {
            let session = db.session();
            for row in ROWS..ROWS + 20 {
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
        let len = db.vector_index_len(LABEL, PROPERTY).expect("live index");
        db.close().unwrap();
        len
    };

    let db = GrafeoDB::with_config(config(&path)).unwrap();
    assert_eq!(
        db.vector_index_len(LABEL, PROPERTY).expect("restored index"),
        live_len,
        "the reopened index must cover exactly what the persisted one covered"
    );
    db.close().unwrap();
}

/// An index built *after* the file already existed still reaches disk.
///
/// Building an index writes nothing to the WAL, so a close that decides
/// it can skip its flush by looking at the WAL position throws the
/// index away - and the next open rebuilds it, and the next, and the
/// next. This is the rebuild-once-serve-forever path, so it is the one
/// that matters most.
#[test]
fn an_index_built_after_open_survives_the_next_reopen() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("late.grafeo");

    {
        let db = GrafeoDB::with_config(config(&path)).unwrap();
        {
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
        db.close().unwrap();
    }

    {
        // Second lifetime: the rows are already durable, so nothing new
        // reaches the WAL. Only the index is built.
        let db = GrafeoDB::with_config(config(&path)).unwrap();
        assert!(!db.graph_store().has_vector_index(LABEL, PROPERTY));
        db.create_vector_index(LABEL, PROPERTY, Some(DIMS), Some("cosine"), None, None, None)
            .expect("build the index");
        assert_eq!(db.vector_index_len(LABEL, PROPERTY), Some(ROWS));
        db.close().unwrap();
    }

    let db = GrafeoDB::with_config(config(&path)).unwrap();
    assert_eq!(
        db.vector_index_len(LABEL, PROPERTY),
        Some(ROWS),
        "an index built after open was discarded by close"
    );
    db.close().unwrap();
}
