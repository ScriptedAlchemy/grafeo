//! Read and write file headers and database headers.
//!
//! All I/O targets a [`File`] handle. Headers are serialized with bincode and
//! zero-padded to their full region size so that the file layout is always
//! page-aligned.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};

use grafeo_common::utils::error::{Error, Result};

use super::format::{DATA_OFFSET, DB_HEADER_SIZE, DbHeader, FILE_HEADER_SIZE, FileHeader, MAGIC};

// ---------------------------------------------------------------------------
// File header (offset 0, 4 KiB)
// ---------------------------------------------------------------------------

/// Writes a [`FileHeader`] at offset 0, padded to [`FILE_HEADER_SIZE`] bytes.
///
/// # Errors
///
/// Returns an error if serialization fails or the I/O write fails.
pub fn write_file_header(file: &mut File, header: &FileHeader) -> Result<()> {
    let encoded = bincode::serde::encode_to_vec(header, bincode::config::standard())
        .map_err(|e| Error::Serialization(e.to_string()))?;

    // reason: FILE_HEADER_SIZE is 4096, a constant that fits in usize on all targets
    #[allow(clippy::cast_possible_truncation)]
    let mut buf = vec![0u8; FILE_HEADER_SIZE as usize];
    if encoded.len() > buf.len() {
        return Err(Error::Internal(
            "FileHeader serialization exceeds page size".into(),
        ));
    }
    buf[..encoded.len()].copy_from_slice(&encoded);

    file.seek(SeekFrom::Start(0))?;
    file.write_all(&buf)?;
    Ok(())
}

/// Reads and deserializes the [`FileHeader`] from offset 0.
///
/// # Errors
///
/// Returns an error if the I/O read or deserialization fails.
pub fn read_file_header(file: &mut File) -> Result<FileHeader> {
    // reason: FILE_HEADER_SIZE is 4096, a constant that fits in usize on all targets
    #[allow(clippy::cast_possible_truncation)]
    let mut buf = vec![0u8; FILE_HEADER_SIZE as usize];
    file.seek(SeekFrom::Start(0))?;
    file.read_exact(&mut buf)?;

    let (header, _): (FileHeader, _) =
        bincode::serde::decode_from_slice(&buf, bincode::config::standard())
            .map_err(|e| Error::Serialization(e.to_string()))?;
    Ok(header)
}

/// Validates the file header: checks magic bytes and format version.
///
/// # Errors
///
/// Returns an error if the magic bytes are invalid or the format version is unsupported.
pub fn validate_file_header(header: &FileHeader) -> Result<()> {
    if header.magic != MAGIC {
        return Err(Error::Internal(format!(
            "invalid magic bytes: expected {:?}, got {:?}",
            MAGIC, header.magic
        )));
    }
    if header.format_version > super::format::FORMAT_VERSION {
        return Err(Error::Internal(format!(
            "unsupported format version {} (max supported: {})",
            header.format_version,
            super::format::FORMAT_VERSION
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Database headers (two slots at offsets 4 KiB and 8 KiB)
// ---------------------------------------------------------------------------

/// Magic tag marking a self-authenticated header slot tail.
const DB_HEADER_TAIL_MAGIC: [u8; 4] = *b"GDBH";

/// Size of the slot tail: magic (4) + encoded length (4) + CRC-32 (4).
const DB_HEADER_TAIL_SIZE: usize = 12;

/// Returns the byte offset of database header slot 0 or 1.
fn db_header_offset(slot: u8) -> u64 {
    FILE_HEADER_SIZE + u64::from(slot) * DB_HEADER_SIZE
}

/// Writes a [`DbHeader`] to the given slot (0 or 1), padded to
/// [`DB_HEADER_SIZE`] bytes.
///
/// The header slot is the atomic commit point of a checkpoint, so the slot
/// carries its own authentication: the last [`DB_HEADER_TAIL_SIZE`] bytes
/// hold a magic tag, the encoded header length, and a CRC-32 over the
/// encoded bytes. A torn slot write (power loss mid-page) then fails the
/// CRC and the reader falls back to the other slot instead of trusting
/// bytes that happen to decode. Slots written before this scheme have zero
/// padding where the tail lives; the reader accepts them without a tail.
///
/// # Errors
///
/// Returns an error if serialization fails or the I/O write fails.
pub fn write_db_header(file: &mut File, slot: u8, header: &DbHeader) -> Result<()> {
    debug_assert!(slot < 2, "db header slot must be 0 or 1");

    let encoded = bincode::serde::encode_to_vec(header, bincode::config::standard())
        .map_err(|e| Error::Serialization(e.to_string()))?;

    // reason: DB_HEADER_SIZE is 4096, a constant that fits in usize on all targets
    #[allow(clippy::cast_possible_truncation)]
    let mut buf = vec![0u8; DB_HEADER_SIZE as usize];
    if encoded.len() > buf.len() - DB_HEADER_TAIL_SIZE {
        return Err(Error::Internal(
            "DbHeader serialization exceeds page size".into(),
        ));
    }
    buf[..encoded.len()].copy_from_slice(&encoded);

    let tail_start = buf.len() - DB_HEADER_TAIL_SIZE;
    // reason: encoded.len() is bounded by the 4096-byte page just above
    #[allow(clippy::cast_possible_truncation)]
    let encoded_len = encoded.len() as u32;
    buf[tail_start..tail_start + 4].copy_from_slice(&DB_HEADER_TAIL_MAGIC);
    buf[tail_start + 4..tail_start + 8].copy_from_slice(&encoded_len.to_le_bytes());
    buf[tail_start + 8..tail_start + 12].copy_from_slice(&crc32fast::hash(&encoded).to_le_bytes());

    file.seek(SeekFrom::Start(db_header_offset(slot)))?;
    file.write_all(&buf)?;
    Ok(())
}

/// A header slot's state after reading: decodable or damaged.
enum SlotState {
    Valid(DbHeader),
    Invalid(String),
}

/// Reads a single header slot, classifying damage instead of failing.
///
/// I/O errors still surface as `Err`; a slot whose *contents* cannot be
/// authenticated or decoded returns [`SlotState::Invalid`] so the caller
/// can fall back to the other slot.
fn read_db_header_slot(file: &mut File, slot: u8) -> Result<SlotState> {
    debug_assert!(slot < 2, "db header slot must be 0 or 1");

    // reason: DB_HEADER_SIZE is 4096, a constant that fits in usize on all targets
    #[allow(clippy::cast_possible_truncation)]
    let mut buf = vec![0u8; DB_HEADER_SIZE as usize];
    file.seek(SeekFrom::Start(db_header_offset(slot)))?;
    file.read_exact(&mut buf)?;

    let tail_start = buf.len() - DB_HEADER_TAIL_SIZE;
    if buf[tail_start..tail_start + 4] == DB_HEADER_TAIL_MAGIC {
        let encoded_len =
            u32::from_le_bytes(buf[tail_start + 4..tail_start + 8].try_into().unwrap_or([0; 4]))
                as usize;
        let stored_crc =
            u32::from_le_bytes(buf[tail_start + 8..tail_start + 12].try_into().unwrap_or([0; 4]));
        if encoded_len > tail_start {
            return Ok(SlotState::Invalid(format!(
                "slot {slot}: tail claims {encoded_len} encoded bytes, page holds {tail_start}"
            )));
        }
        let actual_crc = crc32fast::hash(&buf[..encoded_len]);
        if actual_crc != stored_crc {
            return Ok(SlotState::Invalid(format!(
                "slot {slot}: tail CRC mismatch (stored {stored_crc:#010X}, computed {actual_crc:#010X}) — torn header write"
            )));
        }
    }

    match bincode::serde::decode_from_slice::<DbHeader, _>(&buf, bincode::config::standard()) {
        Ok((header, _)) => Ok(SlotState::Valid(header)),
        Err(e) => Ok(SlotState::Invalid(format!("slot {slot}: {e}"))),
    }
}

/// Reads both database headers from slots 0 and 1.
///
/// A slot that fails authentication or decoding — e.g. a header write torn
/// by power loss — is treated as [`DbHeader::EMPTY`] so the other slot's
/// generation stays reachable; the checkpoint protocol only ever writes to
/// the inactive slot, so the surviving slot is always the last committed
/// generation.
///
/// # Errors
///
/// Returns an error if the I/O read fails, or if *both* slots are damaged
/// (no committed generation is reachable).
pub fn read_db_headers(file: &mut File) -> Result<(DbHeader, DbHeader)> {
    let s0 = read_db_header_slot(file, 0)?;
    let s1 = read_db_header_slot(file, 1)?;
    match (s0, s1) {
        (SlotState::Valid(h0), SlotState::Valid(h1)) => Ok((h0, h1)),
        (SlotState::Valid(h0), SlotState::Invalid(_)) => Ok((h0, DbHeader::EMPTY)),
        (SlotState::Invalid(_), SlotState::Valid(h1)) => Ok((DbHeader::EMPTY, h1)),
        (SlotState::Invalid(r0), SlotState::Invalid(r1)) => Err(Error::Internal(format!(
            "both database header slots are damaged: {r0}; {r1}"
        ))),
    }
}

/// Returns the active (authoritative) database header.
///
/// The header with the higher `iteration` counter wins; ties go to slot 0.
/// Both headers are returned by-value alongside their slot index. Checksum
/// validation against the data payload is the caller's responsibility (see
/// [`GrafeoFileManager::read_snapshot`](crate::file::GrafeoFileManager::read_snapshot)
/// and [`GrafeoFileManager::read_section_directory`](crate::file::GrafeoFileManager::read_section_directory));
/// this function does not fall back to the other slot on a payload
/// mismatch.
///
/// Returns `(active_slot, header)`.
#[must_use]
pub fn active_db_header(h0: &DbHeader, h1: &DbHeader) -> (u8, DbHeader) {
    if h1.iteration > h0.iteration {
        (1, h1.clone())
    } else {
        (0, h0.clone())
    }
}

/// Returns the slot index (0 or 1) that should be written next.
///
/// This is always the *inactive* (stale) slot, i.e., the one with the
/// lower iteration counter.
#[must_use]
pub fn inactive_slot(h0: &DbHeader, h1: &DbHeader) -> u8 {
    u8::from(h1.iteration <= h0.iteration)
}

/// Returns the byte offset where snapshot data should be written/read.
///
/// Currently always [`DATA_OFFSET`] (12 KiB), but exposed as a function
/// so callers don't depend on the constant directly.
#[must_use]
pub const fn snapshot_data_offset() -> u64 {
    DATA_OFFSET
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    fn create_test_file() -> (File, tempfile::TempPath) {
        let tmp = NamedTempFile::new().expect("create temp file");
        let (file, path) = tmp.into_parts();
        (file, path)
    }

    #[test]
    fn file_header_roundtrip() {
        let (mut file, _path) = create_test_file();
        let original = FileHeader::new();

        write_file_header(&mut file, &original).unwrap();
        let loaded = read_file_header(&mut file).unwrap();

        assert_eq!(original, loaded);
    }

    #[test]
    fn file_header_validation_rejects_bad_magic() {
        let mut header = FileHeader::new();
        header.magic = *b"NOPE";

        let result = validate_file_header(&header);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("invalid magic"));
    }

    #[test]
    fn file_header_validation_rejects_future_version() {
        let mut header = FileHeader::new();
        header.format_version = 999;

        let result = validate_file_header(&header);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("unsupported"));
    }

    #[test]
    fn db_header_roundtrip_slot0() {
        let (mut file, _path) = create_test_file();

        // Write file header + both DB header slots to establish full layout
        write_file_header(&mut file, &FileHeader::new()).unwrap();
        write_db_header(&mut file, 1, &DbHeader::EMPTY).unwrap();

        let original = DbHeader {
            iteration: 1,
            checksum: 0xDEAD_BEEF,
            snapshot_length: 1024,
            epoch: 42,
            transaction_id: 7,
            node_count: 100,
            edge_count: 200,
            timestamp_ms: 1_700_000_000_000,
            directory_offset: 0,
        };

        write_db_header(&mut file, 0, &original).unwrap();
        let (h0, _h1) = read_db_headers(&mut file).unwrap();

        assert_eq!(original, h0);
    }

    #[test]
    fn db_header_roundtrip_slot1() {
        let (mut file, _path) = create_test_file();
        write_file_header(&mut file, &FileHeader::new()).unwrap();
        write_db_header(&mut file, 0, &DbHeader::EMPTY).unwrap();

        let original = DbHeader {
            iteration: 5,
            checksum: 0x1234,
            snapshot_length: 2048,
            epoch: 10,
            transaction_id: 3,
            node_count: 50,
            edge_count: 75,
            timestamp_ms: 1_700_000_001_000,
            directory_offset: 0,
        };

        write_db_header(&mut file, 1, &original).unwrap();
        let (_h0, h1) = read_db_headers(&mut file).unwrap();

        assert_eq!(original, h1);
    }

    #[test]
    fn active_header_picks_higher_iteration() {
        let h0 = DbHeader {
            iteration: 3,
            ..DbHeader::EMPTY
        };
        let h1 = DbHeader {
            iteration: 5,
            ..DbHeader::EMPTY
        };

        let (slot, active) = active_db_header(&h0, &h1);
        assert_eq!(slot, 1);
        assert_eq!(active.iteration, 5);
    }

    #[test]
    fn active_header_defaults_to_slot0_when_equal() {
        let h0 = DbHeader {
            iteration: 2,
            ..DbHeader::EMPTY
        };
        let h1 = DbHeader {
            iteration: 2,
            ..DbHeader::EMPTY
        };

        let (slot, _) = active_db_header(&h0, &h1);
        assert_eq!(slot, 0);
    }

    #[test]
    fn active_header_handles_both_empty() {
        let (slot, header) = active_db_header(&DbHeader::EMPTY, &DbHeader::EMPTY);
        assert_eq!(slot, 0);
        assert!(header.is_empty());
    }

    #[test]
    fn inactive_slot_alternates() {
        let h0 = DbHeader {
            iteration: 3,
            ..DbHeader::EMPTY
        };
        let h1 = DbHeader {
            iteration: 5,
            ..DbHeader::EMPTY
        };

        // h1 is active (higher), so inactive is slot 0
        assert_eq!(inactive_slot(&h0, &h1), 0);

        // h0 is active (higher), so inactive is slot 1
        assert_eq!(inactive_slot(&h1, &h0), 1);
    }

    #[test]
    fn legacy_padded_slot_decodes_with_zero_directory_offset() {
        // A slot written by the previous revision: the 8-field encoding
        // (no `directory_offset`, no authentication tail) zero-padded to
        // the page. Field additions must decode it with the legacy
        // sentinel, not error.
        let (mut file, _path) = create_test_file();
        write_file_header(&mut file, &FileHeader::new()).unwrap();

        let legacy: (u64, u32, u64, u64, u64, u64, u64, u64) =
            (3, 0xCAFE, 0, 9, 4, 10, 20, 1_700_000_002_000);
        let encoded = bincode::serde::encode_to_vec(legacy, bincode::config::standard()).unwrap();
        // reason: DB_HEADER_SIZE is 4096, a constant that fits in usize
        #[allow(clippy::cast_possible_truncation)]
        let mut slot = vec![0u8; DB_HEADER_SIZE as usize];
        slot[..encoded.len()].copy_from_slice(&encoded);
        file.seek(SeekFrom::Start(db_header_offset(0))).unwrap();
        file.write_all(&slot).unwrap();
        write_db_header(&mut file, 1, &DbHeader::EMPTY).unwrap();

        let (h0, _h1) = read_db_headers(&mut file).unwrap();
        assert_eq!(h0.iteration, 3);
        assert_eq!(h0.checksum, 0xCAFE);
        assert_eq!(h0.epoch, 9);
        assert_eq!(h0.directory_offset, 0, "legacy padding must decode as 0");
    }

    #[test]
    fn torn_slot_falls_back_to_the_other_slot() {
        let (mut file, _path) = create_test_file();
        write_file_header(&mut file, &FileHeader::new()).unwrap();

        let committed = DbHeader {
            iteration: 4,
            checksum: 0x1111,
            ..DbHeader::EMPTY
        };
        write_db_header(&mut file, 0, &committed).unwrap();
        let newer = DbHeader {
            iteration: 5,
            checksum: 0x2222,
            ..DbHeader::EMPTY
        };
        write_db_header(&mut file, 1, &newer).unwrap();

        // Tear slot 1's front while its authentication tail survives — the
        // state a partial header-page write leaves behind. The higher
        // iteration in the torn bytes must NOT win.
        file.seek(SeekFrom::Start(db_header_offset(1))).unwrap();
        file.write_all(&[0xFF; 32]).unwrap();

        let (h0, h1) = read_db_headers(&mut file).unwrap();
        assert_eq!(h0.iteration, 4);
        assert!(h1.is_empty(), "torn slot must read as empty");
        let (_, active) = active_db_header(&h0, &h1);
        assert_eq!(active.iteration, 4, "must fall back to the committed generation");
    }

    #[test]
    fn both_slots_damaged_is_an_error() {
        let (mut file, _path) = create_test_file();
        write_file_header(&mut file, &FileHeader::new()).unwrap();
        write_db_header(&mut file, 0, &DbHeader::EMPTY).unwrap();
        write_db_header(&mut file, 1, &DbHeader::EMPTY).unwrap();

        for slot in 0..2u8 {
            file.seek(SeekFrom::Start(db_header_offset(slot))).unwrap();
            file.write_all(&[0xFF; 32]).unwrap();
        }

        let err = read_db_headers(&mut file).unwrap_err();
        assert!(
            err.to_string().contains("both database header slots are damaged"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn dual_header_alternation() {
        let (mut file, _path) = create_test_file();
        write_file_header(&mut file, &FileHeader::new()).unwrap();
        write_db_header(&mut file, 0, &DbHeader::EMPTY).unwrap();
        write_db_header(&mut file, 1, &DbHeader::EMPTY).unwrap();

        // First checkpoint: write to inactive slot
        let (h0, h1) = read_db_headers(&mut file).unwrap();
        let target_slot = inactive_slot(&h0, &h1);

        let checkpoint1 = DbHeader {
            iteration: 1,
            checksum: 0xAAAA,
            snapshot_length: 100,
            epoch: 1,
            ..DbHeader::EMPTY
        };
        write_db_header(&mut file, target_slot, &checkpoint1).unwrap();

        // Verify checkpoint 1 is active
        let (h0, h1) = read_db_headers(&mut file).unwrap();
        let (active_slot, active) = active_db_header(&h0, &h1);
        assert_eq!(active.iteration, 1);

        // Second checkpoint: write to the other slot
        let target_slot = inactive_slot(&h0, &h1);
        assert_ne!(target_slot, active_slot);

        let checkpoint2 = DbHeader {
            iteration: 2,
            checksum: 0xBBBB,
            snapshot_length: 200,
            epoch: 2,
            ..DbHeader::EMPTY
        };
        write_db_header(&mut file, target_slot, &checkpoint2).unwrap();

        // Verify checkpoint 2 is active
        let (h0, h1) = read_db_headers(&mut file).unwrap();
        let (_, active) = active_db_header(&h0, &h1);
        assert_eq!(active.iteration, 2);
        assert_eq!(active.snapshot_length, 200);
    }
}
