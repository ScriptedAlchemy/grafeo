//! Single-file database format (`.grafeo`).
//!
//! This module implements a portable, crash-safe, single-file storage format.
//! At rest, only the `.grafeo` file exists. During operation a sidecar
//! WAL directory (`<path>.wal/`) captures in-flight mutations and is
//! removed after each checkpoint.
//!
//! ## File layout
//!
//! | Offset | Size | Contents |
//! |--------|------|----------|
//! | 0 | 4 KiB | [`FileHeader`]: magic `GRAF`, version, page size |
//! | 4 KiB | 4 KiB | [`DbHeader`] slot 0 (H1) |
//! | 8 KiB | 4 KiB | [`DbHeader`] slot 1 (H2) |
//! | 12 KiB+ | variable | Snapshot data payload (bincode-encoded) |
//!
//! ## Crash safety
//!
//! Two database headers alternate writes. A checkpoint first writes the new
//! generation's payload **out of place** (never over bytes the active header
//! points at) and fsyncs it, then overwrites the inactive header slot with
//! metadata pointing at the new generation and fsyncs again — the header
//! flip is the single atomic commit point. Each slot carries a CRC tail so
//! a torn header write is detected and the other slot's generation wins.
//! A kill at any byte of the checkpoint leaves the store openable at the
//! previous consistent generation.

pub mod format;
pub mod header;
pub mod manager;

pub use format::{DbHeader, FileHeader, MAGIC};
pub use manager::GrafeoFileManager;
