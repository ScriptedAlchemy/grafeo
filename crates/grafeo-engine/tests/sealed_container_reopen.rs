//! Reopening a sealed compact container: the open reads what it decodes and
//! serves the rest in place, verifying each page on the first read that
//! touches it.
//!
//! ```bash
//! cargo test -p grafeo-engine \
//!     --features "lpg,gql,wal,grafeo-file,compact-store,mmap" \
//!     --test sealed_container_reopen
//! ```

#![cfg(all(feature = "compact-store", feature = "lpg", feature = "grafeo-file"))]

use std::path::Path;

use grafeo_common::types::{EdgeId, NodeId, PropertyKey, Value};
use grafeo_core::graph::compact::IncrementalCompactStoreBuilder;
use grafeo_engine::config::StorageFormat;
use grafeo_engine::{Config, GrafeoDB};

const ENTITIES: u64 = 20_000;

fn key(index: u64) -> Value {
    Value::from(format!("entity:{index:06}"))
}

/// A long unique payload per entity, so the record dictionary is most of
/// the container.
fn record(index: u64) -> Value {
    Value::from(format!("record:{index:06}:{}", "r".repeat(600)))
}

fn write_container(path: &Path) {
    let mut builder = IncrementalCompactStoreBuilder::new();
    let (key_name, record_name) = (PropertyKey::new("key"), PropertyKey::new("record"));
    for index in 0..ENTITIES {
        builder
            .push_node(
                NodeId(index),
                ["Entity"],
                [(&key_name, &key(index)), (&record_name, &record(index))],
            )
            .unwrap();
    }
    for index in 0..ENTITIES - 1 {
        builder
            .push_edge(
                EdgeId(index),
                "NEXT",
                NodeId(index),
                NodeId(index + 1),
                std::iter::empty(),
            )
            .unwrap();
    }
    GrafeoDB::write_compact_container(path, builder, ["key".to_owned()]).unwrap();
}

fn open(path: &Path) -> grafeo_common::utils::error::Result<GrafeoDB> {
    GrafeoDB::with_config(Config::read_only(path).with_storage_format(StorageFormat::SingleFile))
}

fn verified_pages(db: &GrafeoDB) -> (usize, usize) {
    db.layered_store()
        .expect("a sealed container opens layered")
        .base_store_arc()
        .verified_pages()
        .expect("a sealed base is read in place from a paged section")
}

#[test]
fn a_sealed_container_opens_without_decoding_or_hashing_its_rows() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("sealed.grafeo");
    write_container(&path);

    let db = open(&path).unwrap();
    let usage = db.memory_usage().compact_base;
    assert!(
        usage.mapped_bytes > 12_000_000,
        "the fixture maps {} bytes",
        usage.mapped_bytes
    );
    assert!(
        usage.heap_bytes * 20 < usage.mapped_bytes,
        "the open built {} heap bytes over a {}-byte section",
        usage.heap_bytes,
        usage.mapped_bytes
    );
    let (opened, total) = verified_pages(&db);
    assert!(
        opened * 10 < total,
        "the open verified {opened} of {total} pages"
    );

    let store = db.graph_store();
    assert!(store.has_property_index("key"));
    assert_eq!(
        store.find_nodes_by_property("key", &key(13_579)),
        vec![NodeId(13_579)]
    );
    assert_eq!(
        store.get_node_property(NodeId(13_579), &PropertyKey::new("record")),
        Some(record(13_579))
    );
    assert_eq!(
        store.find_nodes_by_property("key", &Value::from("entity:missing")),
        Vec::<NodeId>::new()
    );
    // Each lookup binary-searches the stored row order, the key dictionary,
    // and the id map, so it verifies a few pages per probe region: a budget
    // in pages, independent of how large the container is.
    let (read, _) = verified_pages(&db);
    assert!(
        read - opened <= 32,
        "three point reads verified {} more of {total} pages",
        read - opened
    );
    assert_eq!(db.compact_base_integrity_fault(), None);
}

/// Every occurrence of `needle` in `haystack`.
fn occurrences(haystack: &[u8], needle: &[u8]) -> Vec<usize> {
    haystack
        .windows(needle.len())
        .enumerate()
        .filter(|(_, window)| *window == needle)
        .map(|(at, _)| at)
        .collect()
}

#[test]
fn a_corrupt_page_refuses_the_read_that_touches_it_and_latches_a_fault() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("sealed.grafeo");
    write_container(&path);
    let mut bytes = std::fs::read(&path).unwrap();
    let at = occurrences(&bytes, b"record:007777:");
    assert_eq!(at.len(), 1, "the record is stored once, in its dictionary");
    bytes[at[0] + 20] ^= 0x01;
    std::fs::write(&path, &bytes).unwrap();

    let db = open(&path).unwrap();
    let store = db.graph_store();
    let record_key = PropertyKey::new("record");
    assert_eq!(
        store.get_node_property(NodeId(1_000), &record_key),
        Some(record(1_000)),
        "a page the corruption does not touch still reads"
    );
    assert_eq!(db.compact_base_integrity_fault(), None);

    assert_eq!(store.get_node_property(NodeId(7_777), &record_key), None);
    let fault = db
        .compact_base_integrity_fault()
        .expect("the refused read latches its fault");
    assert!(fault.contains("CRC mismatch"), "{fault}");
}

#[test]
fn corrupt_metadata_refuses_the_open() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("sealed.grafeo");
    write_container(&path);
    let mut bytes = std::fs::read(&path).unwrap();
    // The record column's zone map bound is metadata, written after every
    // region, so it is the last occurrence of the first record.
    let at = *occurrences(&bytes, b"record:000000:").last().unwrap();
    bytes[at + 20] ^= 0x01;
    std::fs::write(&path, &bytes).unwrap();

    let error = open(&path)
        .err()
        .expect("corrupt metadata refuses the open");
    assert!(error.to_string().contains("CRC mismatch"), "{error}");
}
