//! Close-time checkpoint elision for unchanged stores.
//!
//! `GrafeoDB::close` re-serialized and rewrote every section on every close,
//! so closing a store that was only read cost a full container rewrite —
//! corpus-scale work for large graphs. Close now proves the container is
//! already current (empty WAL, no sidecar replay at open, no WAL-bypassing
//! mutation, header watermarks matching the live store) and skips the
//! checkpoint, leaving the committed generation untouched. These tests pin
//! both directions: unchanged stores skip, and every kind of change —
//! WAL-logged mutations, sidecar replay, index builds that bypass the WAL,
//! named-graph management — still forces the full checkpoint.

#![cfg(all(feature = "grafeo-file", feature = "wal", feature = "lpg"))]

use std::path::{Path, PathBuf};

use grafeo_engine::{Config, GrafeoDB};

fn open_db(path: &Path) -> GrafeoDB {
    GrafeoDB::with_config(Config::persistent(path)).expect("open database")
}

/// Header generation counter, read through the storage layer after all
/// engine handles are closed. Each checkpoint increments it exactly once.
fn header_iteration(path: &Path) -> u64 {
    let manager = grafeo_storage::file::GrafeoFileManager::open(path).expect("open container");
    let iteration = manager.active_header().iteration;
    manager.close().expect("close container");
    iteration
}

fn seeded_store(dir: &Path) -> PathBuf {
    let path = dir.join("store.grafeo");
    let db = open_db(&path);
    for i in 0..50 {
        let id = db.create_node(&["Doc"]);
        db.set_node_property(id, "title", format!("doc-{i}").into());
    }
    db.close().expect("close seeded store");
    path
}

#[test]
fn close_without_changes_skips_the_checkpoint() {
    let dir = tempfile::tempdir().unwrap();
    let path = seeded_store(dir.path());
    let baseline = header_iteration(&path);

    // Reopen, only read, close: the container is already current, so close
    // must not write a new generation.
    let db = open_db(&path);
    let result = db.execute("MATCH (d:Doc) RETURN count(d) AS total").unwrap();
    assert_eq!(result.scalar::<i64>().unwrap(), 50);
    db.close().expect("close unchanged store");

    assert_eq!(
        header_iteration(&path),
        baseline,
        "a read-only session must not force a container rewrite on close"
    );

    // The skipped checkpoint must leave a fully usable store behind.
    let db = open_db(&path);
    let result = db.execute("MATCH (d:Doc) RETURN count(d) AS total").unwrap();
    assert_eq!(result.scalar::<i64>().unwrap(), 50);
    db.close().expect("close after verification");
}

#[test]
fn close_after_mutations_still_checkpoints() {
    let dir = tempfile::tempdir().unwrap();
    let path = seeded_store(dir.path());
    let baseline = header_iteration(&path);

    let db = open_db(&path);
    db.create_node(&["Doc"]);
    db.close().expect("close mutated store");

    assert!(
        header_iteration(&path) > baseline,
        "a mutated store must checkpoint on close"
    );

    let db = open_db(&path);
    let result = db.execute("MATCH (d:Doc) RETURN count(d) AS total").unwrap();
    assert_eq!(result.scalar::<i64>().unwrap(), 51);
    db.close().expect("close after verification");
}

#[test]
fn close_after_sidecar_replay_still_checkpoints() {
    let dir = tempfile::tempdir().unwrap();
    let path = seeded_store(dir.path());

    // Reconstruct a crash: a container plus a sidecar WAL holding records
    // that never reached the container. Copy both out from a live session
    // (commit records are fsynced at commit time) before the clean close
    // erases the sidecar.
    let crash_path = dir.path().join("crashed.grafeo");
    {
        let db = open_db(&path);
        // A committed transaction: the commit record is what recovery uses
        // to accept the records, and committing force-syncs the WAL file so
        // the copy below observes the full frames.
        db.execute("CREATE (d:Doc {title: 'written-only-to-wal'})")
            .expect("stage WAL-only row");
        std::fs::copy(&path, &crash_path).unwrap();
        let wal_dir = format!("{}.wal", path.display());
        let crash_wal_dir = format!("{}.wal", crash_path.display());
        std::fs::create_dir_all(&crash_wal_dir).unwrap();
        for entry in std::fs::read_dir(&wal_dir).unwrap() {
            let entry = entry.unwrap();
            std::fs::copy(
                entry.path(),
                Path::new(&crash_wal_dir).join(entry.file_name()),
            )
            .unwrap();
        }
        db.close().expect("close original store");
    }

    let baseline = header_iteration(&crash_path);

    // Open replays the sidecar; even though this session appends no records
    // of its own, close must persist the replayed rows into the container.
    let db = open_db(&crash_path);
    let result = db.execute("MATCH (d:Doc) RETURN count(d) AS total").unwrap();
    assert_eq!(result.scalar::<i64>().unwrap(), 51);
    db.close().expect("close replayed store");

    assert!(
        header_iteration(&crash_path) > baseline,
        "sidecar replay must force a checkpoint on close"
    );

    // The replayed row must now live in the container itself.
    let db = open_db(&crash_path);
    let result = db
        .execute("MATCH (d:Doc {title: 'written-only-to-wal'}) RETURN count(d) AS total")
        .unwrap();
    assert_eq!(result.scalar::<i64>().unwrap(), 1);
    db.close().expect("close after verification");
}

#[cfg(feature = "vector-index")]
#[test]
fn close_after_vector_index_build_still_checkpoints() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("vectors.grafeo");
    {
        let db = open_db(&path);
        for i in 0..8 {
            let id = db.create_node(&["Doc"]);
            db.set_node_property(
                id,
                "embedding",
                grafeo_common::types::Value::Vector(vec![i as f32, 1.0, 0.0].into()),
            );
        }
        db.close().expect("close seeded store");
    }
    let baseline = header_iteration(&path);

    // The DB-level index build bypasses the WAL, so it must be tracked as
    // container staleness or the persisted index section would be lost by
    // the close-skip.
    let db = open_db(&path);
    db.create_vector_index("Doc", "embedding", Some(3), None, None, None, None)
        .expect("build vector index");
    db.close().expect("close after index build");

    assert!(
        header_iteration(&path) > baseline,
        "a WAL-bypassing index build must force a checkpoint on close"
    );

    // The checkpoint must have captured the freshly built index section.
    // (Engine-side HNSW restore additionally requires catalog-registered
    // index shells, which this DB-level API does not create — callers such
    // as TraceDecay re-issue create_vector_index at open — so the durable
    // evidence here is the persisted section itself.)
    let manager =
        grafeo_storage::file::GrafeoFileManager::open(&path).expect("open container");
    let directory = manager
        .read_section_directory()
        .expect("read section directory")
        .expect("checkpointed store has a directory");
    assert!(
        directory
            .find(grafeo_common::storage::SectionType::VectorStore)
            .is_some(),
        "the vector-index section must be checkpointed into the container"
    );
    manager.close().expect("close container");
}

#[test]
fn close_after_named_graph_management_still_checkpoints() {
    let dir = tempfile::tempdir().unwrap();
    let path = seeded_store(dir.path());
    let baseline = header_iteration(&path);

    let db = open_db(&path);
    assert!(db.create_graph("scratch").expect("create named graph"));
    db.close().expect("close after graph create");

    assert!(
        header_iteration(&path) > baseline,
        "DB-level named-graph creation must force a checkpoint on close"
    );

    let db = open_db(&path);
    assert!(db.list_graphs().iter().any(|name| name == "scratch"));
    db.close().expect("close after verification");
}
