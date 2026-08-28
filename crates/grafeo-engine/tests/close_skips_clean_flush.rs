//! `GrafeoDB::close()` used to re-serialize every section on every
//! close, whether or not anything had changed: `FlushReason::Explicit`
//! means "write all sections", and the per-section `is_dirty` flags
//! cannot say otherwise because `build_sections` mints fresh, clean
//! wrappers on each flush. Closing a large database therefore wrote the
//! entire store back over a byte-identical copy.
//!
//! The skip is gated on the WAL, the only witness that can testify that
//! a session wrote nothing. With the WAL disabled the full flush stands,
//! so the "persists on close" half of this file covers both modes while
//! the "writes nothing" half is a WAL-on claim.
//!
//! ```bash
//! cargo test -p grafeo-engine \
//!     --features "lpg,gql,wal,grafeo-file" \
//!     --test close_skips_clean_flush
//! ```

#![cfg(all(feature = "grafeo-file", feature = "lpg", feature = "gql"))]

use grafeo_engine::{Config, GrafeoDB};

fn config(path: &std::path::Path, wal_enabled: bool) -> Config {
    Config {
        wal_enabled,
        // Sync durability so the crash simulation below can copy a
        // sidecar WAL that already has the record on disk.
        wal_durability: grafeo_engine::config::DurabilityMode::Sync,
        ..Config::persistent(path)
    }
}

/// Sidecar WAL directory for a `.grafeo` container.
#[cfg(feature = "wal")]
fn sidecar_wal_path(db_path: &std::path::Path) -> std::path::PathBuf {
    let mut p = db_path.as_os_str().to_owned();
    p.push(".wal");
    std::path::PathBuf::from(p)
}

/// Shallow recursive copy, enough for a WAL directory.
#[cfg(feature = "wal")]
fn copy_dir(from: &std::path::Path, to: &std::path::Path) {
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        let target = to.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            copy_dir(&entry.path(), &target);
        } else {
            std::fs::copy(entry.path(), target).unwrap();
        }
    }
}

/// Byte-level fingerprint of the container: length plus contents.
fn fingerprint(path: &std::path::Path) -> (u64, Vec<u8>) {
    let bytes = std::fs::read(path).expect("container should exist");
    (bytes.len() as u64, bytes)
}

fn seed(path: &std::path::Path, wal_enabled: bool) {
    let db = GrafeoDB::with_config(config(path, wal_enabled)).unwrap();
    let session = db.session();
    session
        .execute("INSERT (:Person {name: 'Alix', age: 30})")
        .unwrap();
    session
        .execute("INSERT (:City {name: 'Amsterdam'})")
        .unwrap();
    drop(session);
    db.close().unwrap();
}

/// Opening a database, reading from it, and closing it must leave the
/// container byte-for-byte as it was.
#[test]
#[cfg(feature = "wal")]
fn close_after_only_reads_writes_nothing() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("clean_close.grafeo");
    seed(&path, true);

    let before = fingerprint(&path);

    {
        let db = GrafeoDB::with_config(config(&path, true)).unwrap();
        let session = db.session();
        let rows = session
            .execute("MATCH (p:Person) RETURN p.name")
            .unwrap()
            .rows()
            .len();
        assert_eq!(rows, 1);
        drop(session);
        db.close().unwrap();
    }

    let after = fingerprint(&path);
    assert_eq!(
        before.0, after.0,
        "a read-only session rewrote the container (length changed)"
    );
    assert!(
        before.1 == after.1,
        "a read-only session rewrote the container (contents changed)"
    );
}

/// The skip must not swallow a session that did write. Runs in both WAL
/// modes: with the WAL off the close still takes the full flush path.
fn close_after_mutations_persists(wal_enabled: bool) {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("dirty_close.grafeo");
    seed(&path, wal_enabled);

    {
        let db = GrafeoDB::with_config(config(&path, wal_enabled)).unwrap();
        let session = db.session();
        session.execute("INSERT (:City {name: 'Utrecht'})").unwrap();
        drop(session);
        db.close().unwrap();
    }

    let db = GrafeoDB::with_config(config(&path, wal_enabled)).unwrap();
    let session = db.session();
    let cities = session
        .execute("MATCH (c:City) RETURN c.name ORDER BY c.name")
        .unwrap()
        .rows()
        .len();
    assert_eq!(cities, 2, "a write before close did not survive the reopen");
    assert_eq!(db.node_count(), 3);
    drop(session);
    db.close().unwrap();
}

#[test]
fn close_after_mutations_persists_with_wal() {
    close_after_mutations_persists(true);
}

#[test]
fn close_after_mutations_persists_without_wal() {
    close_after_mutations_persists(false);
}

/// A close that follows a *recovering* open must still write: those
/// records live only in memory and the sidecar, and the sidecar is
/// removed on the way out. The WAL this session opened is untouched, so
/// its record count alone would wrongly say "nothing happened".
#[test]
#[cfg(feature = "wal")]
fn close_after_a_recovering_open_persists() {
    let dir = tempfile::TempDir::new().unwrap();
    let live = dir.path().join("live.grafeo");
    seed(&live, true);

    // Stand in for a crash: copy the container and its sidecar WAL while
    // the database is still open, so the copy carries records that were
    // never checkpointed. `close`ing the original afterwards only tidies
    // up the source.
    let crashed = dir.path().join("crashed.grafeo");
    {
        let db = GrafeoDB::with_config(config(&live, true)).unwrap();
        let session = db.session();
        session.execute("INSERT (:City {name: 'Utrecht'})").unwrap();
        drop(session);
        std::fs::copy(&live, &crashed).unwrap();
        copy_dir(&sidecar_wal_path(&live), &sidecar_wal_path(&crashed));

        db.close().unwrap();
    }

    // Open the copy (replays the sidecar) and close it without writing.
    {
        let db = GrafeoDB::with_config(config(&crashed, true)).unwrap();
        assert_eq!(db.node_count(), 3, "sidecar WAL should have been replayed");
        db.close().unwrap();
    }

    // The recovered record must now be in the container.
    let db = GrafeoDB::with_config(config(&crashed, true)).unwrap();
    assert_eq!(
        db.node_count(),
        3,
        "recovered records were dropped by a close that wrote nothing"
    );
    db.close().unwrap();
}

/// Once a full checkpoint has caught the WAL up, a close that follows
/// without another write must still write nothing - the watermark, not a
/// zero record count, is what the skip compares against.
#[test]
#[cfg(feature = "wal")]
fn close_after_a_checkpointed_write_writes_nothing() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("checkpointed_close.grafeo");
    seed(&path, true);

    let before = {
        let db = GrafeoDB::with_config(config(&path, true)).unwrap();
        let session = db.session();
        session.execute("INSERT (:City {name: 'Utrecht'})").unwrap();
        drop(session);
        db.wal_checkpoint().expect("checkpoint");
        let before = fingerprint(&path);
        db.close().unwrap();
        before
    };

    let after = fingerprint(&path);
    assert_eq!(
        before.0, after.0,
        "close rewrote a container the checkpoint had already brought up to date"
    );
    assert!(before.1 == after.1, "close rewrote the container contents");

    // And the checkpointed write is still there.
    let db = GrafeoDB::with_config(config(&path, true)).unwrap();
    assert_eq!(db.node_count(), 3);
    db.close().unwrap();
}

/// `compact()` rebuilds the store in memory without a single WAL record,
/// so a close after it has to write the new layout even though the WAL
/// is quiet.
#[test]
#[cfg(all(feature = "wal", feature = "compact-store"))]
fn close_after_compact_persists() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("compact_close.grafeo");
    seed(&path, true);

    let before = fingerprint(&path);
    {
        let mut db = GrafeoDB::with_config(config(&path, true)).unwrap();
        db.compact().expect("compact");
        db.close().unwrap();
    }
    let after = fingerprint(&path);
    assert!(
        before.1 != after.1,
        "close skipped the flush and left the compacted layout unpersisted"
    );

    let db = GrafeoDB::with_config(config(&path, true)).unwrap();
    assert_eq!(db.node_count(), 2);
    db.close().unwrap();
}
