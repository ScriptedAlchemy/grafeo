//! Peak-allocation regression for LPG checkpoint serialization.
//!
//! The live [`LpgStore`] already owns the graph. Checkpointing must not retain
//! a second materialized copy of every node, edge, and property container while
//! it builds the encoded blocks. This test measures the real section writer
//! with a thread-local counting allocator so unrelated test threads cannot
//! perturb the result.

#![allow(unsafe_code)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::io::sink;
use std::sync::Arc;

use grafeo_common::storage::Section;
use grafeo_common::types::Value;
use grafeo_core::graph::lpg::{LpgStore, LpgStoreSection};

thread_local! {
    static LIVE: Cell<isize> = const { Cell::new(0) };
    static PEAK: Cell<isize> = const { Cell::new(0) };
}

struct PeakTracking;

fn note_alloc(size: usize) {
    let _ = LIVE.try_with(|live| {
        let current = live.get() + size.cast_signed();
        live.set(current);
        let _ = PEAK.try_with(|peak| peak.set(peak.get().max(current)));
    });
}

fn note_dealloc(size: usize) {
    let _ = LIVE.try_with(|live| live.set(live.get() - size.cast_signed()));
}

// SAFETY: every allocator operation delegates unchanged to `System`; the
// thread-local accounting never affects pointer ownership or layout.
unsafe impl GlobalAlloc for PeakTracking {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            note_alloc(layout.size());
        }
        pointer
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc_zeroed(layout) };
        if !pointer.is_null() {
            note_alloc(layout.size());
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        note_dealloc(layout.size());
        unsafe { System.dealloc(pointer, layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let replacement = unsafe { System.realloc(pointer, layout, new_size) };
        if !replacement.is_null() {
            note_alloc(new_size);
            note_dealloc(layout.size());
        }
        replacement
    }
}

#[global_allocator]
static ALLOCATOR: PeakTracking = PeakTracking;

fn peak_bytes_of<T>(body: impl FnOnce() -> T) -> (T, usize) {
    LIVE.with(|live| live.set(0));
    PEAK.with(|peak| peak.set(0));
    let result = body();
    let peak = PEAK.with(Cell::get).max(0);
    #[allow(clippy::cast_sign_loss)]
    (result, peak as usize)
}

#[test]
fn lpg_checkpoint_does_not_retain_a_materialized_copy_of_every_entity() {
    const NODE_COUNT: usize = 20_000;
    const PROPERTIES_PER_NODE: usize = 32;
    // Regression boundary for this fixed fixture, not a production graph cap.
    // The previous whole-graph materialization peaked above 124 MiB.
    const MAX_CHECKPOINT_HEAP_BYTES: usize = 72 * 1024 * 1024;

    let store = Arc::new(LpgStore::new().expect("create LPG store"));
    for node_ordinal in 0..NODE_COUNT {
        let node = store.create_node(&["Indexed"]);
        for property_ordinal in 0..PROPERTIES_PER_NODE {
            let value = i64::try_from(node_ordinal * PROPERTIES_PER_NODE + property_ordinal)
                .expect("fixture value fits i64");
            store.set_node_property(
                node,
                &format!("property-{property_ordinal}"),
                Value::Int64(value),
            );
        }
    }

    let section = LpgStoreSection::new(store);
    let (result, peak_bytes) = peak_bytes_of(|| section.serialize_into(&mut sink()));
    result.expect("serialize LPG checkpoint");
    println!(
        "nodes={NODE_COUNT}, properties_per_node={PROPERTIES_PER_NODE}, peak={peak_bytes} bytes"
    );

    assert!(
        peak_bytes < MAX_CHECKPOINT_HEAP_BYTES,
        "checkpoint retained corpus-sized entity materialization: peak={peak_bytes} bytes, \
         maximum={MAX_CHECKPOINT_HEAP_BYTES} bytes"
    );
}
