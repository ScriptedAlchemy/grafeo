//! `memory::heap::dash_map_bytes` charges a `DashMap` exactly what it
//! allocated, held to a counting global allocator.

// A counting global allocator is the reference the helper is held to.
#![allow(unsafe_code)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use dashmap::DashMap;
use grafeo_common::collections::grafeo_concurrent_map;
use grafeo_common::memory::heap::{dash_map_bytes, vec_bytes};

struct CountingAllocator;

thread_local! {
    // Per thread, so the test harness's own allocations stay out of the count.
    static LIVE: Cell<isize> = const { Cell::new(0) };
}

fn count(delta: isize) {
    let _ = LIVE.try_with(|live| live.set(live.get() + delta));
}

fn signed(size: usize) -> isize {
    isize::try_from(size).expect("allocation size")
}

// SAFETY: every call delegates to `System` with the caller's layout; the
// counter is a const-initialized thread-local cell.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count(signed(layout.size()));
        // SAFETY: forwarded unchanged.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        count(signed(layout.size()));
        // SAFETY: forwarded unchanged.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        count(-signed(layout.size()));
        // SAFETY: forwarded unchanged.
        unsafe { System.dealloc(pointer, layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        count(signed(size) - signed(layout.size()));
        // SAFETY: forwarded unchanged.
        unsafe { System.realloc(pointer, layout, size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn allocated_by<T>(build: impl FnOnce() -> T) -> (T, usize) {
    let before = LIVE.with(Cell::get);
    let value = build();
    let grown = LIVE.with(Cell::get) - before;
    (
        value,
        usize::try_from(grown).expect("the build freed more than it allocated"),
    )
}

#[test]
fn a_dash_map_is_charged_what_it_allocates() {
    drop(DashMap::<u64, u64>::new());
    for entries in [0_u64, 1, 7, 100, 5_000] {
        for shards in [2_usize, 16, 512] {
            let (map, allocated) = allocated_by(|| {
                let map = DashMap::with_capacity_and_shard_amount(0, shards);
                for key in 0..entries {
                    map.insert(key, vec![key; usize::try_from(key % 5).expect("length")]);
                }
                map
            });
            assert_eq!(
                dash_map_bytes(&map, |_, value| vec_bytes(value)),
                allocated,
                "{entries} entries over {shards} shards"
            );
        }
    }
}

#[test]
fn an_empty_concurrent_map_is_charged_its_shard_array() {
    let (map, allocated) = allocated_by(grafeo_concurrent_map::<u64, u64>);
    assert!(allocated > 0, "an empty DashMap still allocates its shards");
    assert_eq!(dash_map_bytes(&map, |_, _| 0), allocated);
}
