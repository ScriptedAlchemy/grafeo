//! `GrafeoDB::memory_usage` on a reopened compact container reports the heap
//! the open left live: the compact base's tables, dictionaries, id maps, and
//! property indexes, and the section buffer only when it is a heap copy. A
//! mapped section is file-backed page cache and is reported apart.
//!
//! This binary counts every allocation, so the bytes an open leaves live are
//! the reference the report is held to. Both are measured against the open
//! of an empty sealed container, so the engine's fixed skeleton (plan
//! caches, file manager, overlay) is not mistaken for the base's heap.
//!
//! ```bash
//! cargo test -p grafeo-engine \
//!     --features "lpg,gql,wal,grafeo-file,compact-store,mmap" \
//!     --test compact_base_heap_accounting
//! ```

#![cfg(all(
    feature = "compact-store",
    feature = "lpg",
    feature = "grafeo-file",
    feature = "mmap"
))]
// A counting global allocator is the reference the report is held to.
#![allow(unsafe_code)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use grafeo_common::types::{EdgeId, NodeId, PropertyKey, Value};
use grafeo_core::graph::compact::IncrementalCompactStoreBuilder;
use grafeo_engine::config::StorageFormat;
use grafeo_engine::{Config, GrafeoDB, MemoryUsage};

struct CountingAllocator;

static LIVE: AtomicUsize = AtomicUsize::new(0);

// SAFETY: every call delegates to `System` with the caller's layout; the
// counter is a plain atomic.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        LIVE.fetch_add(layout.size(), Ordering::Relaxed);
        // SAFETY: forwarded unchanged.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        LIVE.fetch_add(layout.size(), Ordering::Relaxed);
        // SAFETY: forwarded unchanged.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
        // SAFETY: forwarded unchanged.
        unsafe { System.dealloc(pointer, layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        LIVE.fetch_add(size, Ordering::Relaxed);
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
        // SAFETY: forwarded unchanged.
        unsafe { System.realloc(pointer, layout, size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

/// The counter and `GRAFEO_COMPACT_BASE_MMAP` are process-wide, so the
/// measurements run one at a time.
static MEASUREMENT: Mutex<()> = Mutex::new(());

const ENTITIES: u64 = 20_000;

fn key(name: &str) -> PropertyKey {
    PropertyKey::new(name)
}

/// Entities with a unique string key and an opaque record, each linked to
/// the next by a relation carrying its own key: the shape a sealed code
/// graph generation stores.
fn write_container(path: &Path) {
    let mut builder = IncrementalCompactStoreBuilder::new();
    for index in 0..ENTITIES {
        let properties = [
            (key("key"), Value::from(format!("entity:{index:08}"))),
            (
                key("record"),
                Value::Bytes(
                    vec![
                        u8::try_from(index % 251).expect("byte");
                        64 + usize::try_from(index % 64).expect("length")
                    ]
                    .into(),
                ),
            ),
            (key("rank"), Value::Int64(index.cast_signed())),
        ];
        builder
            .push_node(
                NodeId(index),
                ["Entity"],
                properties.iter().map(|(name, value)| (name, value)),
            )
            .expect("push entity");
    }
    for index in 0..ENTITIES {
        let properties = [(
            key("relation_key"),
            Value::from(format!("relation:{index:08}")),
        )];
        builder
            .push_edge(
                EdgeId(index),
                "LINKS",
                NodeId(index),
                NodeId((index + 1) % ENTITIES),
                properties.iter().map(|(name, value)| (name, value)),
            )
            .expect("push relation");
    }
    GrafeoDB::write_compact_container(path, builder, ["key".to_owned()]).expect("write container");
}

fn open_measured(path: &Path, mapped: bool) -> (GrafeoDB, usize, MemoryUsage) {
    // SAFETY: the measurement lock serializes every test in this binary, and
    // nothing else reads the environment while it is held.
    unsafe {
        std::env::set_var("GRAFEO_COMPACT_BASE_MMAP", if mapped { "1" } else { "0" });
    }
    let before = LIVE.load(Ordering::Relaxed);
    let db = GrafeoDB::with_config(
        Config::read_only(path).with_storage_format(StorageFormat::SingleFile),
    )
    .expect("open the container");
    let live = LIVE.load(Ordering::Relaxed) - before;
    let usage = db.memory_usage();
    (db, live, usage)
}

/// Holds the reported growth over the empty container's open to the live
/// growth, within 10%.
fn assert_reports_live(
    (live, usage): (usize, &MemoryUsage),
    (empty_live, empty): (usize, &MemoryUsage),
) {
    let grown = live - empty_live;
    let reported = usage.total_bytes - empty.total_bytes;
    assert!(
        reported * 10 >= grown * 9 && reported * 10 <= grown * 11,
        "the open left {grown} bytes live over an empty container but memory_usage reports \
         {reported} more: {usage:?}"
    );
}

#[test]
fn an_empty_container_open_reports_the_heap_its_open_left_live() {
    let _measurement = MEASUREMENT
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("empty.grafeo");
    GrafeoDB::write_compact_container(
        &path,
        IncrementalCompactStoreBuilder::new(),
        ["key".to_owned()],
    )
    .expect("write empty container");
    for mapped in [true, false] {
        let (db, live, usage) = open_measured(&path, mapped);
        assert!(
            usage.total_bytes * 10 >= live * 9 && usage.total_bytes * 10 <= live * 11,
            "the open left {live} bytes live but memory_usage reports {}: {usage:?}",
            usage.total_bytes
        );
        drop(db);
    }
}

#[test]
fn a_reopened_compact_base_reports_the_heap_its_open_left_live() {
    let _measurement = MEASUREMENT
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("sealed.grafeo");
    write_container(&path);
    let empty_path = dir.path().join("empty.grafeo");
    GrafeoDB::write_compact_container(
        &empty_path,
        IncrementalCompactStoreBuilder::new(),
        ["key".to_owned()],
    )
    .expect("write empty container");
    let (empty, empty_mapped_live, empty_mapped) = open_measured(&empty_path, true);
    drop(empty);
    let (empty, empty_copied_live, empty_copied) = open_measured(&empty_path, false);
    drop(empty);

    let (db, mapped_live, mapped) = open_measured(&path, true);
    assert_eq!(u64::try_from(db.graph_store().node_count()), Ok(ENTITIES));
    drop(db);
    let (db, copied_live, copied) = open_measured(&path, false);
    assert_eq!(u64::try_from(db.graph_store().node_count()), Ok(ENTITIES));

    assert_reports_live((mapped_live, &mapped), (empty_mapped_live, &empty_mapped));
    assert_reports_live((copied_live, &copied), (empty_copied_live, &empty_copied));
    assert_eq!(copied.compact_base.mapped_bytes, 0);
    assert!(mapped.compact_base.mapped_bytes > 0);
    assert_eq!(
        copied.compact_base.heap_bytes - mapped.compact_base.heap_bytes,
        mapped.compact_base.mapped_bytes,
        "a heap-copied open holds exactly the section a mapped open maps"
    );
}
