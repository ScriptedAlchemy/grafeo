//! Section-based container format for `.grafeo` files.
//!
//! Extends the single-file format with a section directory, enabling
//! independent read/write of typed sections. The container treats
//! section data as opaque `&[u8]` bytes.
//!
//! ## File Layout (v2)
//!
//! | Offset | Size | Contents |
//! |--------|------|----------|
//! | 0x0000 | 4 KiB | FileHeader (magic, format version) |
//! | 0x1000 | 4 KiB | DbHeader H1 (iteration, checksum) |
//! | 0x2000 | 4 KiB | DbHeader H2 (alternating copy) |
//! | 0x3000 | 4 KiB | Section Directory (legacy fixed slot) |
//! | 0x4000+ | variable | Generation regions: section data (page-aligned) followed by that generation's directory page |
//!
//! Checkpoints write each new generation — sections plus its own directory
//! page — out of place, and the active `DbHeader` records the directory's
//! offset (`0` = the legacy fixed slot at 0x3000). Stores written before
//! out-of-place checkpoints keep reading through the fixed slot.

pub mod directory;

#[cfg(feature = "wal")]
pub mod mmap;

#[cfg(feature = "wal")]
pub mod page_fetcher;

#[cfg(feature = "wal")]
pub mod spill;

pub use directory::SectionDirectory;

#[cfg(feature = "wal")]
pub use mmap::MmapSection;

#[cfg(feature = "wal")]
pub use page_fetcher::{AccessHint, MmapPageFetcher, PageFetcher};

#[cfg(feature = "wal")]
pub use spill::write_and_mmap_spill_file;
