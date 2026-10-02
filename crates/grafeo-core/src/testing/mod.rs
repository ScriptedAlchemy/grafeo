//! Testing utilities for Grafeo internals.
//!
//! Re-exports from `grafeo-common::testing` for backward compatibility.
//! New code should import from `grafeo_common::testing` directly.

pub use grafeo_common::testing::crash;

/// Call counters for store-wide `node_count` / `edge_count` walks.
///
/// These counts compile only under `debug_assertions` so `cargo test`
/// (which builds dependencies with debug assertions) can observe O(store)
/// snapshots. `#[cfg(test)]` would not work here: dependency crates are
/// not compiled with `cfg(test)` when another crate's tests run.
#[cfg(debug_assertions)]
pub mod count_probe {
    use std::sync::atomic::{AtomicU64, Ordering};

    static NODE_COUNT_CALLS: AtomicU64 = AtomicU64::new(0);
    static EDGE_COUNT_CALLS: AtomicU64 = AtomicU64::new(0);

    /// Clears both counters.
    pub fn reset() {
        NODE_COUNT_CALLS.store(0, Ordering::Relaxed);
        EDGE_COUNT_CALLS.store(0, Ordering::Relaxed);
    }

    /// Number of `LpgStore::node_count` calls since [`reset`].
    pub fn node_count_calls() -> u64 {
        NODE_COUNT_CALLS.load(Ordering::Relaxed)
    }

    /// Number of `LpgStore::edge_count` calls since [`reset`].
    pub fn edge_count_calls() -> u64 {
        EDGE_COUNT_CALLS.load(Ordering::Relaxed)
    }

    /// Records one `node_count` walk. Called from the store implementation.
    pub fn record_node_count() {
        NODE_COUNT_CALLS.fetch_add(1, Ordering::Relaxed);
    }

    /// Records one `edge_count` walk. Called from the store implementation.
    pub fn record_edge_count() {
        EDGE_COUNT_CALLS.fetch_add(1, Ordering::Relaxed);
    }
}
