//! On-disk format constants and header structures for `.grafeo` files.
//!
//! A `.grafeo` file has three regions:
//!
//! | Offset | Size | Contents |
//! |--------|------|----------|
//! | 0 | 4 KiB | [`FileHeader`] (magic, version, page size) |
//! | 4 KiB | 4 KiB | [`DbHeader`] slot 0 (H1) |
//! | 8 KiB | 4 KiB | [`DbHeader`] slot 1 (H2) |
//! | 12 KiB+ | variable | Snapshot data payload |
//!
//! The two database headers alternate writes for crash safety. The one
//! with the higher [`DbHeader::iteration`] counter is the current state.

use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

/// 4-byte file magic identifying a `.grafeo` database file.
pub const MAGIC: [u8; 4] = *b"GRAF";

/// Current on-disk format version. Bump when the layout changes incompatibly.
pub const FORMAT_VERSION: u32 = 1;

/// Size of the file header region (bytes).
pub const FILE_HEADER_SIZE: u64 = 4096;

/// Size of each database header slot (bytes).
pub const DB_HEADER_SIZE: u64 = 4096;

/// Byte offset where snapshot data begins (after file header + 2 DB headers).
pub const DATA_OFFSET: u64 = FILE_HEADER_SIZE + 2 * DB_HEADER_SIZE;

/// Fixed file header at offset 0. Written once at creation time.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FileHeader {
    /// Magic bytes identifying the file format. Must equal [`MAGIC`].
    pub magic: [u8; 4],
    /// On-disk format version for backward compatibility checks.
    pub format_version: u32,
    /// Page alignment size in bytes (always 4096 currently).
    pub page_size: u32,
    /// Milliseconds since UNIX epoch when the file was created.
    pub creation_timestamp_ms: u64,
    /// Grafeo version that created this file (UTF-8, zero-padded).
    pub creator_version: [u8; 32],
}

impl FileHeader {
    /// Creates a new file header with the current timestamp and crate version.
    #[must_use]
    pub fn new() -> Self {
        let mut creator_version = [0u8; 32];
        let version_bytes = env!("CARGO_PKG_VERSION").as_bytes();
        let copy_len = version_bytes.len().min(32);
        creator_version[..copy_len].copy_from_slice(&version_bytes[..copy_len]);

        // reason: millis since UNIX epoch fits in u64 for ~585 million years
        #[allow(clippy::cast_possible_truncation)]
        let creation_timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        Self {
            magic: MAGIC,
            format_version: FORMAT_VERSION,
            // reason: FILE_HEADER_SIZE is 4096, a constant that fits in u32
            #[allow(clippy::cast_possible_truncation)]
            page_size: FILE_HEADER_SIZE as u32,
            creation_timestamp_ms,
            creator_version,
        }
    }
}

impl Default for FileHeader {
    fn default() -> Self {
        Self::new()
    }
}

/// Database header stored in two alternating slots for crash safety.
///
/// On each checkpoint the *inactive* slot is overwritten with a new header
/// pointing to the freshly written snapshot data, then the file is fsynced.
/// If the process crashes mid-write, the other slot still contains valid
/// metadata for the previous checkpoint.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DbHeader {
    /// Monotonically increasing counter, incremented on each checkpoint.
    /// The header with the higher iteration is the current state.
    pub iteration: u64,
    /// CRC-32 checksum of the snapshot data payload.
    pub checksum: u32,
    /// Byte length of the snapshot data payload starting at [`DATA_OFFSET`].
    pub snapshot_length: u64,
    /// MVCC epoch at the time of this checkpoint.
    pub epoch: u64,
    /// Last committed transaction ID at this checkpoint.
    pub transaction_id: u64,
    /// Number of nodes in the database at this checkpoint.
    pub node_count: u64,
    /// Number of edges in the database at this checkpoint.
    pub edge_count: u64,
    /// Milliseconds since UNIX epoch when this header was written.
    pub timestamp_ms: u64,
    /// Absolute byte offset of this generation's section directory page.
    ///
    /// `0` means the legacy fixed directory location (12 KiB): header slots
    /// written before checkpoints became out-of-place are zero-padded, so
    /// decoding them with this field present yields `0`. Checkpoints written
    /// by current code place each generation's directory page inside that
    /// generation's own region and record its offset here, which is what
    /// lets a checkpoint avoid overwriting the previous generation in place.
    pub directory_offset: u64,
}

impl DbHeader {
    /// An empty header representing a freshly created (no data) database.
    pub const EMPTY: Self = Self {
        iteration: 0,
        checksum: 0,
        snapshot_length: 0,
        epoch: 0,
        transaction_id: 0,
        node_count: 0,
        edge_count: 0,
        timestamp_ms: 0,
        directory_offset: 0,
    };

    /// Returns `true` if this header has never been written (iteration == 0).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.iteration == 0
    }
}

impl Default for DbHeader {
    fn default() -> Self {
        Self::EMPTY
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_header_has_correct_magic() {
        let header = FileHeader::new();
        assert_eq!(header.magic, *b"GRAF");
    }

    #[test]
    fn file_header_has_current_version() {
        let header = FileHeader::new();
        assert_eq!(header.format_version, FORMAT_VERSION);
    }

    #[test]
    fn file_header_embeds_crate_version() {
        let header = FileHeader::new();
        let version = String::from_utf8_lossy(&header.creator_version);
        let version = version.trim_end_matches('\0');
        assert_eq!(version, env!("CARGO_PKG_VERSION"));
    }

    #[test]
    fn db_header_empty_is_default() {
        let header = DbHeader::default();
        assert!(header.is_empty());
        assert_eq!(header.iteration, 0);
        assert_eq!(header.snapshot_length, 0);
    }

    #[test]
    fn data_offset_is_12kib() {
        assert_eq!(DATA_OFFSET, 12288);
        assert_eq!(DATA_OFFSET, 3 * 4096);
    }
}
