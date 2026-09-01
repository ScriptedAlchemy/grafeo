//! Engine-level recovery from a torn checkpoint.
//!
//! A kill during the checkpoint's section-data phase must leave the store
//! openable at the previous consistent generation, and WAL replay onto
//! that generation must still function. This test stages exactly that
//! state: a committed generation, un-checkpointed mutations in the sidecar
//! WAL, and garbage from an aborted out-of-place checkpoint past the live
//! region.

#![cfg(all(feature = "grafeo-file", feature = "lpg", feature = "gql", feature = "wal"))]

use std::io::{Seek, SeekFrom, Write};

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

#[test]
fn torn_checkpoint_with_pending_wal_recovers_previous_generation_and_replays() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("origin.grafeo");

    // Generation A: committed by a clean close.
    {
        let db = GrafeoDB::open(&path).unwrap();
        let session = db.session();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        drop(session);
        db.close().unwrap();
    }
    let committed_len = std::fs::metadata(&path).unwrap().len();

    // Post-A mutations that reach only the sidecar WAL: flush the WAL so
    // the records are durable, then leak the handle instead of closing —
    // the state a kill leaves after logging but before the checkpoint.
    {
        let db = GrafeoDB::open(&path).unwrap();
        let session = db.session();
        session.execute("INSERT (:Person {name: 'Gus'})").unwrap();
        drop(session);
        db.wal().expect("wal enabled").flush().unwrap();
        std::mem::forget(db);
    }

    // The leaked handle still holds the origin's file lock, so stage the
    // crash state on a copy: store file plus sidecar WAL.
    let victim = dir.path().join("victim.grafeo");
    std::fs::copy(&path, &victim).unwrap();
    let wal_src = dir.path().join("origin.grafeo.wal");
    let wal_dst = dir.path().join("victim.grafeo.wal");
    std::fs::create_dir_all(&wal_dst).unwrap();
    for entry in std::fs::read_dir(&wal_src).unwrap() {
        let entry = entry.unwrap();
        std::fs::copy(entry.path(), wal_dst.join(entry.file_name())).unwrap();
    }

    // Simulate the torn checkpoint: an aborted out-of-place data phase is
    // partial section bytes past the live region, with no header flip.
    {
        let mut file = std::fs::OpenOptions::new().write(true).open(&victim).unwrap();
        file.seek(SeekFrom::Start(committed_len.div_ceil(4096) * 4096))
            .unwrap();
        file.write_all(&[0xDB; 10_000]).unwrap();
        file.sync_all().unwrap();
    }

    // Open: generation A must load with no CRC error, and the WAL must
    // replay the un-checkpointed mutation on top of it.
    {
        let db = GrafeoDB::open(&victim).unwrap();
        let session = db.session();
        let result = session
            .execute("MATCH (n:Person) RETURN n.name ORDER BY n.name")
            .unwrap();
        let names: Vec<_> = result.rows().iter().map(|row| row[0].clone()).collect();
        assert_eq!(
            names,
            vec![Value::String("Alix".into()), Value::String("Gus".into())],
            "generation A plus WAL replay must both survive the torn checkpoint"
        );
        session.execute("INSERT (:Person {name: 'Harm'})").unwrap();
        drop(session);

        // The recovered store must checkpoint and round-trip cleanly.
        db.close().unwrap();
    }
    {
        let db = GrafeoDB::open(&victim).unwrap();
        assert_eq!(db.node_count(), 3);
        db.close().unwrap();
    }
}
