//! Peak-allocation regression bound for the container write path.
//!
//! `write_sections` requires every section's bytes up front, so a caller
//! flushing a whole store had to materialise all of them at once: the
//! transient heap was O(store) on top of the store itself.
//! `write_sections_streaming` asks each section to emit itself into the
//! file in turn, so the transient is O(largest single buffer).
//!
//! This file measures that difference instead of asserting it by
//! inspection, using a thread-local counting allocator. Thread-local (not
//! global) so that other tests running concurrently in this binary cannot
//! perturb the reading.

// A `GlobalAlloc` implementation is inherently unsafe; every method here
// forwards to `System` and only records sizes. Test-only.
#![allow(unsafe_code)]
#![cfg(feature = "grafeo-file")]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use grafeo_common::storage::{Section, SectionType};
use grafeo_common::utils::error::Result;
use grafeo_storage::file::GrafeoFileManager;

// ── Thread-local peak allocation tracker ────────────────────────────

thread_local! {
    /// Live bytes on this thread since the last `reset`. Signed: a
    /// dealloc of memory allocated before the reset (or on another
    /// thread) legitimately drives this negative, and clamping would
    /// understate the peak.
    static LIVE: Cell<isize> = const { Cell::new(0) };
    /// High-water mark of `LIVE` since the last `reset`.
    static PEAK: Cell<isize> = const { Cell::new(0) };
}

struct PeakTracking;

fn note_alloc(size: usize) {
    // `try_with`: during thread teardown the TLS slot is gone, and a
    // panic inside the allocator would abort the process.
    let _ = LIVE.try_with(|live| {
        let now = live.get() + size.cast_signed();
        live.set(now);
        let _ = PEAK.try_with(|peak| {
            if now > peak.get() {
                peak.set(now);
            }
        });
    });
}

fn note_dealloc(size: usize) {
    let _ = LIVE.try_with(|live| live.set(live.get() - size.cast_signed()));
}

// SAFETY: every method forwards to `System` unchanged and only records
// sizes alongside; the returned pointers and their validity come
// entirely from the system allocator.
unsafe impl GlobalAlloc for PeakTracking {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            note_alloc(layout.size());
        }
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            note_alloc(layout.size());
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        note_dealloc(layout.size());
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let out = unsafe { System.realloc(ptr, layout, new_size) };
        if !out.is_null() {
            // Count the new block before releasing the old one: a
            // growing realloc may hold both, and that transient is
            // exactly what this test is about.
            note_alloc(new_size);
            note_dealloc(layout.size());
        }
        out
    }
}

#[global_allocator]
static ALLOCATOR: PeakTracking = PeakTracking;

/// Runs `body` with the thread's allocation counters zeroed and returns
/// its peak live bytes.
fn peak_bytes_of<T>(body: impl FnOnce() -> T) -> (T, usize) {
    LIVE.with(|c| c.set(0));
    PEAK.with(|c| c.set(0));
    let out = body();
    let peak = PEAK.with(Cell::get).max(0);
    // reason: clamped non-negative directly above
    #[allow(clippy::cast_sign_loss)]
    (out, peak as usize)
}

// ── Fixture ─────────────────────────────────────────────────────────

const SECTION_COUNT: usize = 8;
const SECTION_BYTES: usize = 4 * 1024 * 1024;

/// A section whose payload is already resident (it stands in for a live
/// store). `serialize` has to hand out an owned copy; `serialize_into`
/// writes the resident bytes straight through.
struct ResidentSection {
    section_type: SectionType,
    payload: Vec<u8>,
}

impl Section for ResidentSection {
    fn section_type(&self) -> SectionType {
        self.section_type
    }
    fn serialize(&self) -> Result<Vec<u8>> {
        Ok(self.payload.clone())
    }
    fn serialize_into(&self, sink: &mut dyn std::io::Write) -> Result<()> {
        sink.write_all(&self.payload)?;
        Ok(())
    }
    fn deserialize(&mut self, _data: &[u8]) -> Result<()> {
        Ok(())
    }
    fn is_dirty(&self) -> bool {
        true
    }
    fn mark_clean(&self) {}
    fn memory_usage(&self) -> usize {
        self.payload.len()
    }
}

fn fixture() -> Vec<ResidentSection> {
    // Distinct types so the section directory keeps all of them: the
    // directory is keyed by type and would otherwise collapse duplicates.
    let types = [
        SectionType::Catalog,
        SectionType::LpgStore,
        SectionType::RdfStore,
        SectionType::CompactStore,
        SectionType::OverlayDeletions,
        SectionType::VectorStore,
        SectionType::TextIndex,
        SectionType::RdfRing,
    ];
    assert_eq!(types.len(), SECTION_COUNT);
    types
        .into_iter()
        .enumerate()
        .map(|(i, section_type)| ResidentSection {
            section_type,
            // reason: index is bounded by SECTION_COUNT
            #[allow(clippy::cast_possible_truncation)]
            payload: vec![i as u8; SECTION_BYTES],
        })
        .collect()
}

#[test]
fn streaming_write_peak_is_bounded_by_one_section_not_the_store() {
    let dir = tempfile::TempDir::new().expect("tempdir");
    let sections = fixture();
    let total_bytes = SECTION_COUNT * SECTION_BYTES;

    // Buffered path: what the flush used to do — serialize every
    // section, hold all of them, then write.
    let buffered_path = dir.path().join("buffered.grafeo");
    let manager = GrafeoFileManager::create(&buffered_path).expect("create");
    let ((), buffered_peak) = peak_bytes_of(|| {
        let owned: Vec<(SectionType, Vec<u8>)> = sections
            .iter()
            .map(|s| (s.section_type, s.serialize().expect("serialize")))
            .collect();
        let refs: Vec<(SectionType, &[u8])> =
            owned.iter().map(|(t, d)| (*t, d.as_slice())).collect();
        manager.write_sections(&refs, 1, 1, 0, 0).expect("write");
    });
    manager.close().expect("close");

    // Streaming path: each section emits itself into the file in turn.
    let streamed_path = dir.path().join("streamed.grafeo");
    let manager = GrafeoFileManager::create(&streamed_path).expect("create");
    let ((), streaming_peak) = peak_bytes_of(|| {
        let refs: Vec<&dyn Section> = sections.iter().map(|s| s as &dyn Section).collect();
        manager
            .write_sections_streaming(&refs, 1, 1, 0, 0)
            .expect("write");
    });
    manager.close().expect("close");

    println!(
        "sections={SECTION_COUNT} x {SECTION_BYTES}B (total {total_bytes}B); \
         buffered peak {buffered_peak}B, streaming peak {streaming_peak}B"
    );

    // The buffered path must hold essentially the whole store.
    assert!(
        buffered_peak >= total_bytes,
        "buffered path should peak at ~O(store): {buffered_peak} < {total_bytes}"
    );

    // The streaming path must stay near one section. The bound is
    // deliberately loose (half a section plus the write buffer) so it
    // fails on a regression to O(store) — a 16x gap — without being
    // sensitive to allocator bookkeeping.
    let bound = SECTION_BYTES / 2;
    assert!(
        streaming_peak < bound,
        "streaming path should peak well under one section: {streaming_peak} >= {bound}"
    );
}
