//! Catalog section format-version guard.
//!
//! The catalog payload carries its format version as the leading byte. A
//! `.grafeo` store whose catalog was written by a revision with an
//! incompatible payload format must fail to open with a typed
//! "unsupported catalog version" error, while a store whose catalog bytes
//! are damaged must keep failing with the section CRC mismatch — the two
//! failures name different problems (revision drift vs corruption) and
//! must stay distinguishable.

#![cfg(all(feature = "grafeo-file", feature = "lpg", feature = "gql"))]

use std::io::{Seek, SeekFrom, Write};
use std::path::PathBuf;

use grafeo_common::storage::SectionType;
use grafeo_engine::GrafeoDB;
use grafeo_storage::file::GrafeoFileManager;

/// Writes a store with real catalog content (a node type definition and a
/// node) at `name` inside `dir`, closing it cleanly.
fn write_current_store(dir: &tempfile::TempDir, name: &str) -> PathBuf {
    let path = dir.path().join(name);
    let db = GrafeoDB::open(&path).unwrap();
    let session = db.session();
    session
        .execute("CREATE NODE TYPE Person (name STRING NOT NULL, age INT64)")
        .unwrap();
    session
        .execute("INSERT (:Person {name: 'Alix', age: 30})")
        .unwrap();
    drop(session);
    db.close().unwrap();
    path
}

/// Returns the `(offset, length, payload)` of the catalog section in the
/// store at `path`, CRC-verified through the production read path.
fn read_catalog_section(path: &PathBuf) -> (u64, u64, Vec<u8>) {
    let manager = GrafeoFileManager::open(path).unwrap();
    let dir = manager
        .read_section_directory()
        .unwrap()
        .expect("store written by the current revision has a v2 directory");
    let entry = dir
        .find(SectionType::Catalog)
        .expect("store written by the current revision has a catalog section")
        .clone();
    let payload = manager.read_section_data(&entry).unwrap();
    manager.close().unwrap();
    (entry.offset, entry.length, payload)
}

#[test]
fn store_with_foreign_catalog_version_fails_typed_not_crc() {
    let dir = tempfile::tempdir().unwrap();

    // Take a real catalog payload from a store written by the current
    // revision, then re-author it with a doctored version byte through the
    // production writer — exactly what a store written by an incompatible
    // revision looks like on disk: intact bytes, matching CRC, foreign
    // format version.
    let source = write_current_store(&dir, "source.grafeo");
    let (_, _, mut payload) = read_catalog_section(&source);
    assert_eq!(payload[0], 2, "current revision writes catalog version 2");
    payload[0] = 3;

    let victim = dir.path().join("foreign_version.grafeo");
    {
        let manager = GrafeoFileManager::create(&victim).unwrap();
        manager
            .write_sections(&[(SectionType::Catalog, payload.as_slice())], 1, 1, 0, 0)
            .unwrap();
        manager.close().unwrap();
    }

    let Err(err) = GrafeoDB::open(&victim) else {
        panic!("foreign catalog version must fail open")
    };
    let msg = err.to_string();
    assert!(
        msg.contains("unsupported catalog version 3 (supported 1, 2)"),
        "error must name found and supported versions, got: {msg}"
    );
    assert!(
        !msg.contains("CRC mismatch"),
        "version drift must not masquerade as corruption, got: {msg}"
    );
}

#[test]
fn corrupted_current_version_store_still_reports_crc_mismatch() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_current_store(&dir, "corrupted.grafeo");

    // Damage a payload byte past the version byte, so the store still
    // claims the current format but its bytes no longer match the CRC the
    // writer recorded in the section directory.
    let (offset, length, payload) = read_catalog_section(&path);
    assert!(length > 8, "catalog with a node type is larger than 8 bytes");
    {
        let mut file = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        file.seek(SeekFrom::Start(offset + 8)).unwrap();
        file.write_all(&[!payload[8]]).unwrap();
        file.sync_all().unwrap();
    }

    let Err(err) = GrafeoDB::open(&path) else {
        panic!("corrupted catalog bytes must fail open")
    };
    let msg = err.to_string();
    assert!(
        msg.contains("Catalog") && msg.contains("CRC mismatch"),
        "corruption must surface as the catalog section CRC mismatch, got: {msg}"
    );
}

#[test]
fn current_revision_store_round_trips_through_the_guard() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_current_store(&dir, "roundtrip.grafeo");

    let db = GrafeoDB::open(&path).unwrap();
    let session = db.session();
    let result = session.execute("MATCH (p:Person) RETURN p.name").unwrap();
    assert_eq!(
        result.rows().len(),
        1,
        "store written by the current revision must reopen through the version guard"
    );
    drop(session);
    db.close().unwrap();
}
