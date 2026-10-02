//! High-level manager for `.grafeo` database files.
//!
//! [`GrafeoFileManager`] owns the file handle and provides create, open,
//! snapshot write/read, and sidecar WAL lifecycle management.

use std::fs::{self, File, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use fs2::FileExt;
use grafeo_common::utils::error::{Error, Result, StorageError};
use parking_lot::Mutex;

use super::format::{DATA_OFFSET, DbHeader, FileHeader};
use super::header;

/// Manages a single `.grafeo` database file.
///
/// # Lifecycle
///
/// 1. [`create`](Self::create) or [`open`](Self::open)
/// 2. Mutations flow through a sidecar WAL (managed externally by the engine)
/// 3. [`write_snapshot`](Self::write_snapshot) checkpoints memory to the file
/// 4. After a successful checkpoint, call [`remove_sidecar_wal`](Self::remove_sidecar_wal)
/// 5. [`close`](Self::close) (or drop) releases the file handle
pub struct GrafeoFileManager {
    /// Path to the `.grafeo` file.
    path: PathBuf,
    /// Open file handle (read/write or read-only).
    file: Mutex<File>,
    /// File header (read once on open, immutable afterwards).
    file_header: FileHeader,
    /// Currently active database header.
    active_header: Mutex<DbHeader>,
    /// Slot index (0 or 1) of the active header.
    active_slot: Mutex<u8>,
    /// Whether this manager was opened in read-only mode.
    read_only: bool,
    /// Encryptor for section data (None = unencrypted).
    #[cfg(feature = "encryption")]
    section_encryptor: Option<grafeo_common::encryption::PageEncryptor>,
}

/// Buffer size interposed between a streaming section and the file.
///
/// Sections emit many small writes (a `u32` length here, a string
/// there), so the raw `File` would see one syscall each. 256 KiB is a
/// bounded, constant staging cost that keeps the syscall count in line
/// with the old one-`write_all`-per-section behaviour.
const SECTION_WRITE_BUFFER_BYTES: usize = 256 * 1024;

/// One section's payload for the shared container-write path.
///
/// Implemented for `(SectionType, &[u8])` (bytes already in hand) and
/// for `&dyn Section` (bytes produced on demand into the sink), which is
/// what lets `write_sections` and `write_sections_streaming` share a
/// single implementation.
pub trait SectionPayload {
    /// The section type this payload will be filed under.
    fn section_type(&self) -> grafeo_common::storage::SectionType;

    /// Writes the payload's bytes into `sink`.
    ///
    /// # Errors
    ///
    /// Returns an error if the payload cannot be produced, or if `sink`
    /// rejects a write.
    fn write_into(&self, sink: &mut dyn Write) -> Result<()>;
}

impl SectionPayload for (grafeo_common::storage::SectionType, &[u8]) {
    fn section_type(&self) -> grafeo_common::storage::SectionType {
        self.0
    }

    fn write_into(&self, sink: &mut dyn Write) -> Result<()> {
        sink.write_all(self.1)?;
        Ok(())
    }
}

impl SectionPayload for &dyn grafeo_common::storage::Section {
    fn section_type(&self) -> grafeo_common::storage::SectionType {
        grafeo_common::storage::Section::section_type(*self)
    }

    fn write_into(&self, sink: &mut dyn Write) -> Result<()> {
        grafeo_common::storage::Section::serialize_into(*self, sink)
    }
}

/// Counts and checksums bytes on their way to an inner writer.
///
/// Replaces the `crc32fast::hash(&data)` + `data.len()` pair that the
/// buffered path could compute only because it held the whole section.
/// `finish` flushes the inner writer before reporting, so the returned
/// length always matches what reached the file.
struct CrcCountingWriter<W: Write> {
    inner: W,
    hasher: crc32fast::Hasher,
    written: u64,
}

/// Writes to a file while refusing to cross a byte bound.
///
/// The out-of-place checkpoint streams the new generation into the gap
/// below the live one without knowing its size up front; this writer is
/// what guarantees the stream can never touch a live byte. A write that
/// would cross `remaining` raises `overflowed` and fails *before* any of
/// its bytes reach the file, so the caller can abandon the gap attempt
/// and restart the stream past the live generation.
struct BoundedFileWriter<'a> {
    file: &'a mut File,
    /// Bytes still available before the bound; `None` means unbounded.
    remaining: Option<u64>,
    overflowed: &'a std::cell::Cell<bool>,
}

impl Write for BoundedFileWriter<'_> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        if let Some(remaining) = self.remaining {
            if (buf.len() as u64) > remaining {
                self.overflowed.set(true);
                return Err(std::io::Error::new(
                    std::io::ErrorKind::WriteZero,
                    "generation stream reached the live-generation bound",
                ));
            }
            self.remaining = Some(remaining - buf.len() as u64);
        }
        self.file.write_all(buf)?;
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.file.flush()
    }
}

impl<W: Write> CrcCountingWriter<W> {
    fn new(inner: W) -> Self {
        Self {
            inner,
            hasher: crc32fast::Hasher::new(),
            written: 0,
        }
    }

    /// Flushes and returns `(length, crc32)` of everything written.
    fn finish(mut self) -> Result<(u64, u32)> {
        self.inner.flush()?;
        Ok((self.written, self.hasher.finalize()))
    }
}

impl<W: Write> Write for CrcCountingWriter<W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let n = self.inner.write(buf)?;
        self.hasher.update(&buf[..n]);
        self.written += n as u64;
        Ok(n)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

impl GrafeoFileManager {
    /// Creates a new `.grafeo` file at `path`.
    ///
    /// Writes the file header and two empty database headers. The file must
    /// not already exist.
    ///
    /// # Errors
    ///
    /// Returns an error if the file already exists or cannot be created.
    pub fn create(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();

        if path.exists() {
            return Err(Error::Internal(format!(
                "file already exists: {}",
                path.display()
            )));
        }

        // Ensure parent directory exists
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent)?;
        }

        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)
            .map_err(|e| {
                if e.kind() == std::io::ErrorKind::AlreadyExists || e.raw_os_error() == Some(183) {
                    Error::Io(std::io::Error::new(
                        std::io::ErrorKind::AlreadyExists,
                        format!(
                            "database file already exists (may be open by another process): {}",
                            path.display()
                        ),
                    ))
                } else {
                    Error::Io(e)
                }
            })?;

        // Acquire an exclusive lock: prevents other processes from opening the same file
        file.try_lock_exclusive().map_err(|_| {
            Error::Internal(format!(
                "database file is locked by another process: {}",
                path.display()
            ))
        })?;

        let file_header = FileHeader::new();
        header::write_file_header(&mut file, &file_header)?;
        header::write_db_header(&mut file, 0, &DbHeader::EMPTY)?;
        header::write_db_header(&mut file, 1, &DbHeader::EMPTY)?;
        file.sync_all()?;

        Ok(Self {
            path,
            file: Mutex::new(file),
            file_header,
            active_header: Mutex::new(DbHeader::EMPTY),
            active_slot: Mutex::new(0),
            read_only: false,
            #[cfg(feature = "encryption")]
            section_encryptor: None,
        })
    }

    /// Opens an existing `.grafeo` file.
    ///
    /// Validates the magic bytes and format version, then selects the
    /// active database header.
    ///
    /// # Errors
    ///
    /// Returns an error if the file does not exist, has invalid magic, or
    /// an unsupported format version.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();

        let mut file = OpenOptions::new().read(true).write(true).open(&path)?;

        // Acquire an exclusive lock: prevents other processes from opening the same file
        file.try_lock_exclusive().map_err(|_| {
            Error::Internal(format!(
                "database file is locked by another process: {}",
                path.display()
            ))
        })?;

        let file_header = header::read_file_header(&mut file)?;
        header::validate_file_header(&file_header)?;

        let (h0, h1) = header::read_db_headers(&mut file)?;
        let (active_slot, active_header) = header::active_db_header(&h0, &h1);

        Ok(Self {
            path,
            file: Mutex::new(file),
            file_header,
            active_header: Mutex::new(active_header),
            active_slot: Mutex::new(active_slot),
            read_only: false,
            #[cfg(feature = "encryption")]
            section_encryptor: None,
        })
    }

    /// Opens an existing `.grafeo` file in read-only mode.
    ///
    /// Uses a **shared** file lock (`try_lock_shared`), allowing multiple
    /// readers to open the same file concurrently, even while a writer holds
    /// an exclusive lock (on platforms with advisory locking).
    ///
    /// The returned manager only supports [`read_snapshot`](Self::read_snapshot)
    /// and other read-only operations. Calling [`write_snapshot`](Self::write_snapshot)
    /// will return an error.
    ///
    /// # Errors
    ///
    /// Returns an error if the file does not exist, has invalid magic, or
    /// an unsupported format version.
    pub fn open_read_only(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();

        let mut file = OpenOptions::new().read(true).open(&path)?;

        // Acquire a shared lock: coexists with other shared locks but
        // blocks if an exclusive lock cannot be shared (platform-dependent).
        file.try_lock_shared().map_err(|_| {
            Error::Internal(format!(
                "database file cannot be locked for reading: {}",
                path.display()
            ))
        })?;

        let file_header = header::read_file_header(&mut file)?;
        header::validate_file_header(&file_header)?;

        let (h0, h1) = header::read_db_headers(&mut file)?;
        let (active_slot, active_header) = header::active_db_header(&h0, &h1);

        Ok(Self {
            path,
            file: Mutex::new(file),
            file_header,
            active_header: Mutex::new(active_header),
            active_slot: Mutex::new(active_slot),
            read_only: true,
            #[cfg(feature = "encryption")]
            section_encryptor: None,
        })
    }

    /// Sets the encryptor for section-level encryption.
    ///
    /// When set, all section data is encrypted on write and decrypted on read.
    /// The GCM authentication tag provides integrity verification, replacing
    /// the CRC-32 checksum for encrypted sections.
    #[cfg(feature = "encryption")]
    pub fn set_section_encryptor(&mut self, encryptor: grafeo_common::encryption::PageEncryptor) {
        self.section_encryptor = Some(encryptor);
    }

    /// Returns `true` if this manager was opened in read-only mode.
    #[must_use]
    pub fn is_read_only(&self) -> bool {
        self.read_only
    }

    /// Writes snapshot data into the file and updates the inactive DB header.
    ///
    /// Steps:
    /// 1. Write `data` at [`DATA_OFFSET`]
    /// 2. Compute CRC-32 checksum
    /// 3. Build a new [`DbHeader`] and write it to the inactive slot
    /// 4. `fsync` the file
    /// 5. Update internal active header/slot state
    ///
    /// # Errors
    ///
    /// Returns an error if any I/O operation fails.
    pub fn write_snapshot(
        &self,
        data: &[u8],
        epoch: u64,
        transaction_id: u64,
        node_count: u64,
        edge_count: u64,
    ) -> Result<()> {
        if self.read_only {
            return Err(Error::Internal(
                "cannot write snapshot: database is open in read-only mode".to_string(),
            ));
        }

        use grafeo_common::testing::crash::maybe_crash;

        let checksum = crc32fast::hash(data);
        // reason: millis since UNIX epoch fits in u64 for ~585 million years
        #[allow(clippy::cast_possible_truncation)]
        let timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let mut file = self.file.lock();
        let active_header = self.active_header.lock();
        let mut active_slot = self.active_slot.lock();

        let new_iteration = active_header.iteration + 1;
        let target_slot = u8::from(*active_slot == 0);

        maybe_crash("write_snapshot:before_data_write");

        // Write snapshot data
        file.seek(SeekFrom::Start(DATA_OFFSET))?;
        file.write_all(data)?;

        maybe_crash("write_snapshot:after_data_write");

        // Truncate file to exact size (remove stale trailing data)
        let file_end = DATA_OFFSET + data.len() as u64;
        file.set_len(file_end)?;

        maybe_crash("write_snapshot:after_truncate");

        // Build and write new header to inactive slot
        let new_header = DbHeader {
            iteration: new_iteration,
            checksum,
            snapshot_length: data.len() as u64,
            epoch,
            transaction_id,
            node_count,
            edge_count,
            timestamp_ms,
            directory_offset: 0,
        };
        header::write_db_header(&mut file, target_slot, &new_header)?;

        maybe_crash("write_snapshot:after_header_write");

        // Ensure everything is on disk before we consider this committed
        file.sync_all()?;

        maybe_crash("write_snapshot:after_fsync");

        // Update internal state: drop the old lock, reacquire to update
        drop(active_header);
        *self.active_header.lock() = new_header;
        *active_slot = target_slot;

        Ok(())
    }

    /// Reads snapshot data from the file using the active database header.
    ///
    /// Returns an empty `Vec` if the database has never been checkpointed
    /// (both headers are empty).
    ///
    /// # Errors
    ///
    /// Returns an error if the read fails or the CRC checksum does not match.
    pub fn read_snapshot(&self) -> Result<Vec<u8>> {
        let active_header = self.active_header.lock();

        if active_header.is_empty() {
            return Ok(Vec::new());
        }

        // v2 files store sections rather than a v1 snapshot blob. They set
        // snapshot_length == 0 and put the directory CRC in the checksum field.
        // Reading 0 bytes here would CRC to 0 and mismatch the directory CRC.
        if active_header.snapshot_length == 0 {
            return Ok(Vec::new());
        }

        // reason: snapshot_length is the size of serialized in-memory data, fits in usize on 64-bit targets;
        // on 32-bit targets the database would OOM long before reaching 4 GiB
        // reason: value bounded by collection size, fits usize
        #[allow(clippy::cast_possible_truncation)]
        let length = active_header.snapshot_length as usize;
        let expected_checksum = active_header.checksum;
        drop(active_header);

        let mut file = self.file.lock();
        file.seek(SeekFrom::Start(DATA_OFFSET))?;

        let mut data = vec![0u8; length];
        std::io::Read::read_exact(&mut *file, &mut data)?;

        // Verify CRC
        let actual_checksum = crc32fast::hash(&data);
        if actual_checksum != expected_checksum {
            return Err(Error::Internal(format!(
                "snapshot checksum mismatch: expected {expected_checksum:#010X}, got {actual_checksum:#010X}"
            )));
        }

        Ok(data)
    }

    /// Returns the path for the sidecar WAL directory.
    ///
    /// For a database at `mydb.grafeo`, the sidecar is `mydb.grafeo.wal/`.
    #[must_use]
    pub fn sidecar_wal_path(&self) -> PathBuf {
        let mut wal_path = self.path.as_os_str().to_owned();
        wal_path.push(".wal");
        PathBuf::from(wal_path)
    }

    /// Returns `true` if a sidecar WAL directory exists.
    #[must_use]
    pub fn has_sidecar_wal(&self) -> bool {
        self.sidecar_wal_path().exists()
    }

    /// Removes the sidecar WAL directory after a successful checkpoint.
    ///
    /// # Errors
    ///
    /// Returns an error if the directory exists but cannot be removed.
    pub fn remove_sidecar_wal(&self) -> Result<()> {
        let wal_path = self.sidecar_wal_path();
        if wal_path.exists() {
            fs::remove_dir_all(&wal_path)?;
        }
        Ok(())
    }

    /// Returns the file path.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Returns a clone of the currently active database header.
    #[must_use]
    pub fn active_header(&self) -> DbHeader {
        self.active_header.lock().clone()
    }

    /// Returns the file header (written at creation, immutable).
    #[must_use]
    pub fn file_header(&self) -> &FileHeader {
        &self.file_header
    }

    /// Returns the total file size on disk.
    ///
    /// # Errors
    ///
    /// Returns an error if the file metadata cannot be read.
    pub fn file_size(&self) -> Result<u64> {
        let file = self.file.lock();
        let metadata = file.metadata()?;
        Ok(metadata.len())
    }

    /// Flushes and syncs the file.
    ///
    /// # Errors
    ///
    /// Returns an error if sync fails.
    pub fn sync(&self) -> Result<()> {
        if !self.read_only {
            let file = self.file.lock();
            file.sync_all()?;
        }
        Ok(())
    }

    // ── Section-based I/O (v2 container format) ─────────────────────

    /// Rounds `value` up to the next container page boundary (4 KiB).
    fn page_align_up(value: u64) -> u64 {
        value.div_ceil(4096) * 4096
    }

    /// Returns the byte offset of the active generation's directory page.
    ///
    /// Headers written before out-of-place checkpoints carry
    /// `directory_offset == 0` (decoded from zero padding), which means the
    /// legacy fixed location.
    fn directory_location(header: &DbHeader) -> u64 {
        if header.directory_offset == 0 {
            crate::container::directory::DIRECTORY_OFFSET
        } else {
            header.directory_offset
        }
    }

    /// Computes the byte extent `[lo, hi)` occupied by the committed (live)
    /// generation, which an in-progress checkpoint must not touch.
    ///
    /// Returns `None` for a store with no committed data. If the live
    /// directory cannot be read or verified (the store is already damaged),
    /// falls back to the whole file so the new generation appends past
    /// everything rather than guessing at reusable space.
    fn live_extent(&self, file: &mut File, active: &DbHeader) -> Result<Option<(u64, u64)>> {
        use crate::container::SectionDirectory;
        use crate::container::directory::SECTION_DATA_OFFSET;

        if active.is_empty() {
            return Ok(None);
        }
        // v1 blob layout: the snapshot occupies [DATA_OFFSET, DATA_OFFSET + len).
        if active.snapshot_length > 0 {
            return Ok(Some((DATA_OFFSET, DATA_OFFSET + active.snapshot_length)));
        }

        let dir_offset = Self::directory_location(active);
        let conservative = {
            let file_len = file.metadata()?.len();
            Some((SECTION_DATA_OFFSET.min(dir_offset), file_len.max(dir_offset + 4096)))
        };

        file.seek(SeekFrom::Start(dir_offset))?;
        let mut buf = vec![0u8; 4096];
        if std::io::Read::read_exact(&mut *file, &mut buf).is_err() {
            return Ok(conservative);
        }
        if crc32fast::hash(&buf) != active.checksum {
            return Ok(conservative);
        }
        let Ok(dir) = SectionDirectory::from_bytes(&buf) else {
            return Ok(conservative);
        };

        let mut lo = dir_offset;
        let mut hi = dir_offset + 4096;
        for entry in dir.entries() {
            lo = lo.min(entry.offset);
            hi = hi.max(entry.offset + entry.length);
        }
        Ok(Some((lo, hi)))
    }

    /// Writes multiple sections to the file using the v2 container format.
    ///
    /// The new generation — section data followed by its directory page —
    /// is written **out of place**, into space not occupied by the live
    /// generation (the gap below it when the new generation fits there,
    /// otherwise appended past it), and made durable with an fsync. Only
    /// then is the new [`DbHeader`] committed to the inactive slot and
    /// fsynced: the header flip is the single atomic commit point, so a
    /// kill at any byte of the checkpoint leaves the store openable at the
    /// previous generation. Space held by the now-dead generation is
    /// reclaimed after the flip by truncating past the new generation's
    /// end when it sits below the dead one (successive generations
    /// ping-pong between the low region and the appended region, so the
    /// file oscillates around one-to-two generations in size).
    ///
    /// # Errors
    ///
    /// Returns an error if write or sync fails.
    pub fn write_sections(
        &self,
        sections: &[(grafeo_common::storage::SectionType, &[u8])],
        epoch: u64,
        transaction_id: u64,
        node_count: u64,
        edge_count: u64,
    ) -> Result<()> {
        self.write_section_payloads(sections, epoch, transaction_id, node_count, edge_count)
    }

    /// Writes sections by streaming each one straight into the container.
    ///
    /// Same on-disk result as [`write_sections`](Self::write_sections),
    /// but no caller-side `Vec<u8>` per section: each [`Section`] emits
    /// itself into the file through
    /// [`Section::serialize_into`](grafeo_common::storage::Section::serialize_into),
    /// so the transient heap during a full-store persist is bounded by
    /// the largest buffer a single section chooses to stage rather than
    /// by the sum of all sections.
    ///
    /// # Errors
    ///
    /// Returns an error if the database is read-only, if a section fails
    /// to serialize, or if write/sync fails.
    pub fn write_sections_streaming(
        &self,
        sections: &[&dyn grafeo_common::storage::Section],
        epoch: u64,
        transaction_id: u64,
        node_count: u64,
        edge_count: u64,
    ) -> Result<()> {
        self.write_section_payloads(sections, epoch, transaction_id, node_count, edge_count)
    }

    /// Shared container-write implementation behind
    /// [`write_sections`](Self::write_sections) and
    /// [`write_sections_streaming`](Self::write_sections_streaming).
    ///
    /// Generic over [`SectionPayload`] so the directory, alignment,
    /// truncation, header-commit and fsync ordering exist exactly once:
    /// the two public entry points differ only in where a section's
    /// bytes come from.
    /// Streams one generation — every section, then its directory page —
    /// into the file starting at `region_start`, never crossing `limit`.
    ///
    /// Returns the directory, its offset, and the region end on success,
    /// and `Ok(None)` when the stream would have crossed `limit`: the bound
    /// is enforced *before* bytes reach the file, so an overflowing attempt
    /// leaves every byte at and past `limit` untouched and the caller can
    /// retry in another region.
    fn stream_generation<P: SectionPayload>(
        &self,
        file: &mut File,
        sections: &[P],
        region_start: u64,
        limit: Option<u64>,
        active_header: &DbHeader,
    ) -> Result<Option<(crate::container::SectionDirectory, u64, u64)>> {
        use crate::container::SectionDirectory;
        use grafeo_common::storage::SectionDirectoryEntry;
        use grafeo_common::testing::crash::maybe_crash;

        let page_size = 4096u64;
        let mut dir = SectionDirectory::new();
        let mut current_offset = region_start;
        let overflowed = std::cell::Cell::new(false);

        // Next checkpoint iteration, used as the high part of the nonce so that
        // the same (section_type, offset) pair produces a different nonce across
        // checkpoints. Without this, identical section layouts would reuse nonces.
        #[cfg(feature = "encryption")]
        // reason: iteration wraps at u32::MAX which takes billions of checkpoints (~100+ years at 1/s)
        #[allow(clippy::cast_possible_truncation)]
        let nonce_iteration = (active_header.iteration + 1) as u32;
        #[cfg(not(feature = "encryption"))]
        let _ = active_header;

        for payload in sections {
            let section_type = payload.section_type();

            file.seek(SeekFrom::Start(current_offset))?;

            // Encryption needs the whole plaintext in hand (AEAD over the
            // section, with the tag appended), so that path still stages
            // one section at a time. Unencrypted containers — the default —
            // stream straight through a CRC-computing writer and never
            // materialise the section at all.
            //
            // Nonce high word: iteration in bits [31:8], section type in bits [7:0].
            // Bit-packing (not XOR) ensures unique high words: XOR is commutative
            // so `iter ^ type` can collide across different (iter, type) pairs,
            // but packing into disjoint bit lanes is injective for type < 256.
            // Nonce low word: page-aligned write offset (unique within a checkpoint).
            // AAD binds the ciphertext to the section type, preventing relocation.
            #[cfg(feature = "encryption")]
            let (length, checksum) = if let Some(ref enc) = self.section_encryptor {
                let mut plaintext = Vec::new();
                payload.write_into(&mut plaintext)?;
                let nonce_high = (nonce_iteration << 8) | (section_type as u32 & 0xFF);
                let nonce = grafeo_common::encryption::build_nonce(nonce_high, current_offset);
                let aad = format!("grafeo-section:{}", section_type as u32);
                let ciphertext = enc
                    .encrypt(&plaintext, &nonce, aad.as_bytes())
                    .map_err(|e| Error::Internal(format!("section encryption failed: {e}")))?;
                if limit.is_some_and(|limit| current_offset + ciphertext.len() as u64 > limit) {
                    return Ok(None);
                }
                file.write_all(&ciphertext)?;
                (ciphertext.len() as u64, crc32fast::hash(&ciphertext))
            } else {
                let bounded = BoundedFileWriter {
                    file: &mut *file,
                    remaining: limit.map(|limit| limit.saturating_sub(current_offset)),
                    overflowed: &overflowed,
                };
                let mut writer = CrcCountingWriter::new(std::io::BufWriter::with_capacity(
                    SECTION_WRITE_BUFFER_BYTES,
                    bounded,
                ));
                match payload.write_into(&mut writer).and_then(|()| writer.finish()) {
                    Ok(result) => result,
                    Err(_) if overflowed.get() => return Ok(None),
                    Err(error) => return Err(error),
                }
            };

            #[cfg(not(feature = "encryption"))]
            let (length, checksum) = {
                let bounded = BoundedFileWriter {
                    file: &mut *file,
                    remaining: limit.map(|limit| limit.saturating_sub(current_offset)),
                    overflowed: &overflowed,
                };
                let mut writer = CrcCountingWriter::new(std::io::BufWriter::with_capacity(
                    SECTION_WRITE_BUFFER_BYTES,
                    bounded,
                ));
                match payload.write_into(&mut writer).and_then(|()| writer.finish()) {
                    Ok(result) => result,
                    Err(_) if overflowed.get() => return Ok(None),
                    Err(error) => return Err(error),
                }
            };

            dir.upsert(SectionDirectoryEntry {
                section_type,
                version: 1,
                flags: section_type.default_flags(),
                offset: current_offset,
                length,
                checksum,
            })?;

            // Align next section to page boundary
            let section_end = current_offset + length;
            current_offset = (section_end + page_size - 1) / page_size * page_size;
        }

        maybe_crash("write_sections:after_data");

        // Write this generation's directory page right after its sections.
        // `current_offset` is already page-aligned by the section loop.
        let directory_offset = current_offset;
        let dir_bytes = dir.to_bytes();
        if limit.is_some_and(|limit| directory_offset + dir_bytes.len() as u64 > limit) {
            return Ok(None);
        }
        file.seek(SeekFrom::Start(directory_offset))?;
        file.write_all(&dir_bytes)?;
        let region_end = directory_offset + dir_bytes.len() as u64;

        Ok(Some((dir, directory_offset, region_end)))
    }

    fn write_section_payloads<P: SectionPayload>(
        &self,
        sections: &[P],
        epoch: u64,
        transaction_id: u64,
        node_count: u64,
        edge_count: u64,
    ) -> Result<()> {
        use crate::container::directory::SECTION_DATA_OFFSET;
        use grafeo_common::testing::crash::maybe_crash;

        if self.read_only {
            return Err(Error::Internal(
                "cannot write sections: database is open in read-only mode".to_string(),
            ));
        }

        let mut file = self.file.lock();
        let active_header = self.active_header.lock();
        let mut active_slot = self.active_slot.lock();

        maybe_crash("write_sections:before_data");

        // Place the new generation outside the live one. Streaming payloads
        // do not know their serialized size up front, so the gap below the
        // live generation is tried optimistically: the stream runs behind a
        // bound that refuses to touch a live byte, and a stream that would
        // cross it is abandoned (the partial bytes are dead space nothing
        // references) and restarted past the live end. Sections re-emit
        // deterministically, so the retry writes the same generation.
        let live = self.live_extent(&mut file, &active_header)?;
        let first_limit = live.map(|(live_lo, _)| live_lo);
        let attempt = self.stream_generation(
            &mut file,
            sections,
            SECTION_DATA_OFFSET,
            first_limit,
            &active_header,
        )?;
        let (dir, directory_offset, region_end) = match (attempt, live) {
            (Some(written), _) => written,
            (None, Some((_, live_hi))) => {
                let region_start = Self::page_align_up(live_hi.max(SECTION_DATA_OFFSET));
                self.stream_generation(&mut file, sections, region_start, None, &active_header)?
                    .ok_or_else(|| {
                        Error::Internal(
                            "unbounded generation stream reported a bound overflow".to_string(),
                        )
                    })?
            }
            (None, None) => {
                return Err(Error::Internal(
                    "generation stream overflowed with no live generation".to_string(),
                ));
            }
        };

        maybe_crash("write_sections:after_directory");

        // Make the whole new generation durable BEFORE the header flip. If
        // the process dies anywhere up to here, the previous header still
        // points at the untouched previous generation.
        file.sync_all()?;

        // Build and write new DbHeader to inactive slot — the atomic commit.
        let new_iteration = active_header.iteration + 1;
        let target_slot = u8::from(*active_slot == 0);
        // reason: millis since UNIX epoch fits in u64 for ~585 million years
        #[allow(clippy::cast_possible_truncation)]
        let timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let new_header = DbHeader {
            iteration: new_iteration,
            checksum: dir.checksum(),
            snapshot_length: 0, // Not used in v2; directory CRC is in checksum field
            epoch,
            transaction_id,
            node_count,
            edge_count,
            timestamp_ms,
            directory_offset,
        };
        header::write_db_header(&mut file, target_slot, &new_header)?;

        // Ensure the flip is on disk before reclaiming the old generation.
        file.sync_all()?;

        maybe_crash("write_sections:after_fsync");

        // Reclaim: everything past the new generation is now dead (either
        // the previous generation, when this one was placed below it, or
        // stale trailing bytes). Truncation only runs after the flip is
        // durable, so a kill anywhere earlier leaves the old bytes intact.
        if file.metadata()?.len() > region_end {
            file.set_len(region_end)?;
        }

        // Update internal state
        drop(active_header);
        *self.active_header.lock() = new_header;
        *active_slot = target_slot;

        Ok(())
    }

    /// Reads the section directory from the file.
    ///
    /// Detects v2 format by checking the `snapshot_length` field in the active
    /// DbHeader: v2 writes set `snapshot_length = 0`, while v1 always has a
    /// non-zero snapshot length when data exists.
    ///
    /// Returns `None` only when the file is unambiguously v1
    /// (`snapshot_length` non-zero) or uninitialized (header iteration is 0).
    /// Once the header asserts v2, any failure to locate or parse the
    /// directory is surfaced as an error: misreporting v2 corruption as a v1
    /// file would cause callers to fall back to v1 read paths and mask the
    /// underlying problem.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - I/O fails
    /// - The header asserts v2 but the file is too short to hold a directory page
    /// - The directory page bytes fail to parse as a `SectionDirectory`
    /// - The directory page CRC does not match the value recorded in the active header
    pub fn read_section_directory(&self) -> Result<Option<crate::container::SectionDirectory>> {
        use crate::container::SectionDirectory;

        let active_header = self.active_header.lock();

        // v1 files have snapshot_length > 0; v2 files set it to 0 and put the
        // directory CRC in the checksum field. An uninitialized header (iteration
        // == 0) means no data has been written yet.
        if active_header.is_empty() || active_header.snapshot_length > 0 {
            return Ok(None);
        }
        let expected_checksum = active_header.checksum;
        // Headers written before out-of-place checkpoints carry 0 here and
        // mean the legacy fixed directory page.
        let directory_offset = Self::directory_location(&active_header);
        drop(active_header);

        // Past this point the header asserts v2: any failure to read or parse
        // the directory is real corruption, not a v1/v2 misdetection. Surface
        // it instead of silently falling through to read_snapshot, where v1 CRC
        // logic would mask the underlying cause.
        let file_size = self.file.lock().metadata()?.len();
        if file_size < directory_offset + 4096 {
            return Err(Error::Storage(StorageError::Corruption(format!(
                "v2 header indicates section directory at offset {directory_offset:#X}, \
                 but file is only {file_size} bytes",
            ))));
        }

        let mut file = self.file.lock();
        file.seek(SeekFrom::Start(directory_offset))?;

        let mut buf = vec![0u8; 4096];
        std::io::Read::read_exact(&mut *file, &mut buf)?;

        let dir = SectionDirectory::from_bytes(&buf).map_err(|e| {
            Error::Storage(StorageError::Corruption(format!(
                "v2 section directory at offset {directory_offset:#X} failed to parse: {e}",
            )))
        })?;

        // Cross-check the directory bytes against the CRC the writer recorded
        // in the active header. A mismatch means the directory page is torn or
        // corrupted (e.g. a partial write from a crashed checkpoint), not a
        // format ambiguity.
        let actual_checksum = crc32fast::hash(&buf);
        if actual_checksum != expected_checksum {
            return Err(Error::Storage(StorageError::Corruption(format!(
                "v2 section directory checksum mismatch: \
                 header recorded {expected_checksum:#010X}, computed {actual_checksum:#010X}",
            ))));
        }

        if dir.is_empty() {
            return Ok(None);
        }
        Ok(Some(dir))
    }

    /// Reads a single section's data from the file.
    ///
    /// Uses the section directory entry to locate and verify the data.
    ///
    /// # Errors
    ///
    /// Returns an error if read fails or CRC checksum doesn't match.
    pub fn read_section_data(
        &self,
        entry: &grafeo_common::storage::SectionDirectoryEntry,
    ) -> Result<Vec<u8>> {
        let mut file = self.file.lock();
        file.seek(SeekFrom::Start(entry.offset))?;

        // reason: section length is bounded by file size, which fits in usize on 64-bit targets;
        // on 32-bit targets sections would OOM long before reaching 4 GiB
        // reason: value bounded by collection size, fits usize
        #[allow(clippy::cast_possible_truncation)]
        let mut data = vec![0u8; entry.length as usize];
        std::io::Read::read_exact(&mut *file, &mut data)?;

        // Verify CRC on the raw bytes (encrypted or plaintext)
        let actual_crc = crc32fast::hash(&data);
        if actual_crc != entry.checksum {
            return Err(Error::Storage(StorageError::Corruption(format!(
                "section {:?} CRC mismatch: expected {:#010X}, got {actual_crc:#010X}",
                entry.section_type, entry.checksum
            ))));
        }

        // Decrypt if encryption is enabled
        #[cfg(feature = "encryption")]
        if let Some(ref enc) = self.section_encryptor {
            let aad = format!("grafeo-section:{}", entry.section_type as u32);
            return enc.decrypt(&data, aad.as_bytes()).map_err(|_| {
                Error::Internal(format!(
                    "section {:?} decryption failed: wrong key or corrupted data",
                    entry.section_type
                ))
            });
        }

        Ok(data)
    }

    /// Memory-maps a single section for zero-copy read access.
    ///
    /// The section's CRC-32 is verified against the mmap'd bytes before
    /// returning, which also warms the OS page cache. Only sections with
    /// `flags.mmap_able = true` can be mapped (index sections).
    ///
    /// The returned [`MmapSection`](crate::container::MmapSection) is
    /// independent of the file mutex: multiple mmaps can coexist. However,
    /// all `MmapSection` handles **must be dropped before writing** (via
    /// `write_sections()` or `write_snapshot()`). On Windows the OS rejects
    /// writes to a file with active mappings; on Linux/macOS stale mappings
    /// would read outdated data. See [`MmapSection`](crate::container::MmapSection)
    /// for the full lifecycle.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The section is not mmap-able (data section)
    /// - The mmap system call fails
    /// - The CRC-32 checksum does not match (corrupt data)
    #[allow(unsafe_code)]
    pub fn mmap_section(
        &self,
        entry: &grafeo_common::storage::SectionDirectoryEntry,
    ) -> Result<crate::container::MmapSection> {
        if !entry.flags.mmap_able {
            return Err(Error::Internal(format!(
                "section {:?} is not mmap-able (data sections must be deserialized)",
                entry.section_type
            )));
        }

        if entry.length == 0 {
            return Err(Error::Internal(format!(
                "section {:?} has zero length, cannot mmap",
                entry.section_type
            )));
        }

        let file = self.file.lock();

        // SAFETY: We hold an exclusive lock on the `.grafeo` file, preventing
        // concurrent modification by other processes. The mapping is read-only.
        // The section region [offset .. offset+length] was written by
        // write_sections() and its CRC is verified below before the mmap
        // is exposed to callers.
        // reason: section length is bounded by file size, fits in usize on 64-bit targets
        #[allow(clippy::cast_possible_truncation)]
        let section_len = entry.length as usize;
        let mmap = unsafe {
            memmap2::MmapOptions::new()
                .offset(entry.offset)
                .len(section_len)
                .map(&*file)
        }
        .map_err(Error::Io)?;

        drop(file);

        // Verify CRC on the mmap'd bytes. This reads through the mapping,
        // which triggers page faults and warms the OS page cache: a free
        // prefetch disguised as an integrity check.
        let actual_crc = crc32fast::hash(&mmap);
        if actual_crc != entry.checksum {
            return Err(Error::Storage(StorageError::Corruption(format!(
                "section {:?} CRC mismatch: expected {:#010X}, got {actual_crc:#010X}",
                entry.section_type, entry.checksum
            ))));
        }

        Ok(crate::container::MmapSection::new(
            mmap,
            entry.section_type,
            entry.checksum,
        ))
    }

    /// Whether section payloads in this container are encrypted at rest.
    ///
    /// [`mmap_section`](Self::mmap_section) hands back the raw on-disk
    /// bytes, so a zero-copy reader must fall back to
    /// [`read_section_data`](Self::read_section_data) — which decrypts —
    /// when this returns `true`.
    #[must_use]
    pub fn section_encryption_enabled(&self) -> bool {
        #[cfg(feature = "encryption")]
        {
            self.section_encryptor.is_some()
        }
        #[cfg(not(feature = "encryption"))]
        {
            false
        }
    }

    /// Copies the database file to `dest` using the already-locked file handle.
    ///
    /// `std::fs::copy()` opens the source with a new handle, which fails on
    /// Windows when an exclusive lock is held. This method reads through the
    /// existing handle, avoiding lock conflicts.
    ///
    /// # Errors
    ///
    /// Returns an error if the read or write fails.
    pub fn copy_to(&self, dest: &Path) -> Result<u64> {
        let mut file = self.file.lock();
        file.seek(SeekFrom::Start(0))?;

        let mut dest_file = fs::File::create(dest)?;
        let bytes = std::io::copy(&mut *file, &mut dest_file).map_err(Error::Io)?;
        dest_file.sync_all()?;
        Ok(bytes)
    }

    /// Releases the file lock and syncs.
    ///
    /// # Errors
    ///
    /// Returns an error if sync or unlock fails.
    pub fn close(&self) -> Result<()> {
        let file = self.file.lock();
        if !self.read_only {
            file.sync_all()?;
        }
        file.unlock()
            .map_err(|e| Error::Internal(format!("failed to unlock database file: {e}")))?;
        Ok(())
    }
}

impl Drop for GrafeoFileManager {
    fn drop(&mut self) {
        let file = self.file.lock();
        let _ = file.unlock();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn test_dir() -> TempDir {
        TempDir::new().expect("create temp dir")
    }

    /// Section that emits its payload in many small writes — the shape a
    /// real streaming serializer produces, and the case where a naive
    /// length/CRC accumulator would go wrong.
    struct ChattySection {
        section_type: grafeo_common::storage::SectionType,
        payload: Vec<u8>,
    }

    impl grafeo_common::storage::Section for ChattySection {
        fn section_type(&self) -> grafeo_common::storage::SectionType {
            self.section_type
        }
        fn serialize(&self) -> Result<Vec<u8>> {
            Ok(self.payload.clone())
        }
        fn serialize_into(&self, sink: &mut dyn Write) -> Result<()> {
            // Deliberately awkward chunking, and not a divisor of the
            // payload length.
            for chunk in self.payload.chunks(7) {
                sink.write_all(chunk)?;
            }
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

    /// The streaming writer must land the same directory (offsets,
    /// lengths, CRCs) and the same payload bytes as the buffered writer.
    /// Anything else is a silent format fork.
    #[test]
    fn alix_streaming_and_buffered_writes_produce_identical_containers() {
        use grafeo_common::storage::{Section, SectionType};

        let sections = [
            ChattySection {
                section_type: SectionType::Catalog,
                payload: (0..5_000u32).flat_map(u32::to_le_bytes).collect(),
            },
            ChattySection {
                section_type: SectionType::LpgStore,
                // Crosses several page boundaries so alignment is exercised.
                payload: vec![0x5A; 40_000],
            },
            ChattySection {
                section_type: SectionType::VectorStore,
                payload: b"short".to_vec(),
            },
        ];

        let dir = test_dir();

        let buffered_path = dir.path().join("buffered.grafeo");
        {
            let manager = GrafeoFileManager::create(&buffered_path).unwrap();
            let owned: Vec<(SectionType, Vec<u8>)> = sections
                .iter()
                .map(|s| (s.section_type, s.serialize().unwrap()))
                .collect();
            let refs: Vec<(SectionType, &[u8])> =
                owned.iter().map(|(t, d)| (*t, d.as_slice())).collect();
            manager.write_sections(&refs, 7, 9, 11, 13).unwrap();
            manager.close().unwrap();
        }

        let streamed_path = dir.path().join("streamed.grafeo");
        {
            let manager = GrafeoFileManager::create(&streamed_path).unwrap();
            let refs: Vec<&dyn Section> = sections.iter().map(|s| s as &dyn Section).collect();
            manager
                .write_sections_streaming(&refs, 7, 9, 11, 13)
                .unwrap();
            manager.close().unwrap();
        }

        let buffered = GrafeoFileManager::open(&buffered_path).unwrap();
        let streamed = GrafeoFileManager::open(&streamed_path).unwrap();

        let buffered_dir = buffered.read_section_directory().unwrap().unwrap();
        let streamed_dir = streamed.read_section_directory().unwrap().unwrap();

        for section in &sections {
            let b = buffered_dir.find(section.section_type).expect("buffered");
            let s = streamed_dir.find(section.section_type).expect("streamed");
            assert_eq!(b.offset, s.offset, "{:?} offset", section.section_type);
            assert_eq!(b.length, s.length, "{:?} length", section.section_type);
            assert_eq!(
                b.checksum, s.checksum,
                "{:?} checksum",
                section.section_type
            );
            assert_eq!(
                buffered.read_section_data(b).unwrap(),
                streamed.read_section_data(s).unwrap(),
                "{:?} payload",
                section.section_type
            );
            assert_eq!(
                streamed.read_section_data(s).unwrap(),
                section.payload,
                "{:?} payload must survive the streaming write",
                section.section_type
            );
        }

        // Same layout end-to-end, so the files must also be the same size.
        assert_eq!(buffered.file_size().unwrap(), streamed.file_size().unwrap());
    }

    /// A section that fails mid-stream must surface the error rather
    /// than committing a truncated section to the directory.
    #[test]
    fn gus_streaming_write_propagates_section_errors() {
        use grafeo_common::storage::{Section, SectionType};

        struct FailingSection;
        impl Section for FailingSection {
            fn section_type(&self) -> SectionType {
                SectionType::LpgStore
            }
            fn serialize(&self) -> Result<Vec<u8>> {
                Err(Error::Internal("boom".to_string()))
            }
            fn deserialize(&mut self, _data: &[u8]) -> Result<()> {
                Ok(())
            }
            fn is_dirty(&self) -> bool {
                true
            }
            fn mark_clean(&self) {}
            fn memory_usage(&self) -> usize {
                0
            }
        }

        let dir = test_dir();
        let path = dir.path().join("failing.grafeo");
        let manager = GrafeoFileManager::create(&path).unwrap();
        let section = FailingSection;
        let refs: Vec<&dyn Section> = vec![&section];
        let err = manager
            .write_sections_streaming(&refs, 1, 1, 0, 0)
            .expect_err("section failure must surface");
        assert!(err.to_string().contains("boom"), "got {err}");
    }

    #[test]
    fn create_and_open() {
        let dir = test_dir();
        let path = dir.path().join("test.grafeo");

        // Create
        let manager = GrafeoFileManager::create(&path).unwrap();
        assert!(path.exists());
        assert!(manager.active_header().is_empty());
        drop(manager);

        // Open
        let manager = GrafeoFileManager::open(&path).unwrap();
        assert!(manager.active_header().is_empty());
    }

    #[test]
    fn create_fails_if_exists() {
        let dir = test_dir();
        let path = dir.path().join("test.grafeo");

        GrafeoFileManager::create(&path).unwrap();
        let result = GrafeoFileManager::create(&path);
        assert!(result.is_err());
    }

    #[test]
    fn open_fails_if_not_exists() {
        let dir = test_dir();
        let path = dir.path().join("nonexistent.grafeo");

        let result = GrafeoFileManager::open(&path);
        assert!(result.is_err());
    }

    #[test]
    fn write_and_read_snapshot() {
        let dir = test_dir();
        let path = dir.path().join("test.grafeo");

        let manager = GrafeoFileManager::create(&path).unwrap();

        let snapshot_data = b"hello grafeo snapshot data";
        manager.write_snapshot(snapshot_data, 1, 1, 10, 20).unwrap();

        let loaded = manager.read_snapshot().unwrap();
        assert_eq!(loaded, snapshot_data);

        // Verify header was updated
        let header = manager.active_header();
        assert_eq!(header.iteration, 1);
        assert_eq!(header.snapshot_length, snapshot_data.len() as u64);
        assert_eq!(header.epoch, 1);
        assert_eq!(header.node_count, 10);
        assert_eq!(header.edge_count, 20);
    }

    #[test]
    fn snapshot_persists_across_reopen() {
        let dir = test_dir();
        let path = dir.path().join("test.grafeo");

        let snapshot_data = b"persistent data across reopen";

        // Write
        {
            let manager = GrafeoFileManager::create(&path).unwrap();
            manager
                .write_snapshot(snapshot_data, 5, 3, 100, 200)
                .unwrap();
        }

        // Reopen and read
        {
            let manager = GrafeoFileManager::open(&path).unwrap();
            let loaded = manager.read_snapshot().unwrap();
            assert_eq!(loaded, snapshot_data);

            let header = manager.active_header();
            assert_eq!(header.iteration, 1);
            assert_eq!(header.epoch, 5);
            assert_eq!(header.node_count, 100);
        }
    }

    #[test]
    fn alternating_snapshots() {
        let dir = test_dir();
        let path = dir.path().join("test.grafeo");

        let manager = GrafeoFileManager::create(&path).unwrap();

        // First checkpoint
        let data1 = b"snapshot version 1";
        manager.write_snapshot(data1, 1, 1, 10, 5).unwrap();
        assert_eq!(manager.active_header().iteration, 1);

        // Second checkpoint (alternates to other slot)
        let data2 = b"snapshot version 2 with more data";
        manager.write_snapshot(data2, 2, 2, 20, 10).unwrap();
        assert_eq!(manager.active_header().iteration, 2);

        let loaded = manager.read_snapshot().unwrap();
        assert_eq!(loaded, data2);
    }

    #[test]
    fn read_empty_snapshot() {
        let dir = test_dir();
        let path = dir.path().join("test.grafeo");

        let manager = GrafeoFileManager::create(&path).unwrap();
        let data = manager.read_snapshot().unwrap();
        assert!(data.is_empty());
    }

    #[test]
    fn read_snapshot_returns_empty_on_v2_header() {
        // After write_sections, snapshot_length == 0 in the active header and the
        // checksum field holds the section-directory CRC. The pre-fix v1 reader
        // would read 0 bytes, CRC empty data to 0, and mismatch the directory CRC.
        // The fix early-returns Ok(Vec::new()) when snapshot_length == 0.
        use grafeo_common::storage::SectionType;

        let dir = test_dir();
        let path = dir.path().join("v2.grafeo");

        let manager = GrafeoFileManager::create(&path).unwrap();
        manager
            .write_sections(&[(SectionType::LpgStore, b"section payload")], 1, 1, 0, 0)
            .unwrap();

        // Pre-fix: this returned Err("snapshot checksum mismatch").
        // Post-fix: returns Ok(Vec::new()), letting engine fall through to v2 dispatch.
        let data = manager.read_snapshot().unwrap();
        assert!(
            data.is_empty(),
            "v2 file should produce empty snapshot vec, not an error"
        );

        // Sanity: header confirms this is a v2 file (snapshot_length == 0 with non-zero checksum).
        let header = manager.active_header();
        assert_eq!(header.snapshot_length, 0);
        assert!(!header.is_empty());
    }

    #[test]
    fn read_section_directory_surfaces_parse_error_on_v2_header() {
        // A v2 header with a corrupted directory page must not silently
        // degrade to "this is a v1 file" — that masking is what made the
        // GRAFEO-X001 in #323 surface as a misleading snapshot CRC error
        // instead of pointing at the real directory corruption.
        use grafeo_common::storage::SectionType;

        let dir = test_dir();
        let path = dir.path().join("corrupt_dir.grafeo");

        let directory_offset;
        {
            let manager = GrafeoFileManager::create(&path).unwrap();
            manager
                .write_sections(&[(SectionType::LpgStore, b"section payload")], 1, 1, 0, 0)
                .unwrap();
            directory_offset = manager.active_header().directory_offset;
        }

        // Overwrite the directory page count field with a value above MAX_SECTIONS
        // so SectionDirectory::from_bytes rejects it as malformed.
        {
            let mut file = OpenOptions::new().write(true).open(&path).unwrap();
            file.seek(SeekFrom::Start(directory_offset)).unwrap();
            file.write_all(&u32::MAX.to_le_bytes()).unwrap();
        }

        let manager = GrafeoFileManager::open(&path).unwrap();
        let err = manager
            .read_section_directory()
            .expect_err("corrupt v2 directory must surface as Err, not Ok(None)");
        let msg = err.to_string();
        assert!(
            msg.contains("v2 section directory") && msg.contains("failed to parse"),
            "error should name the v2 directory and the parse failure, got: {msg}"
        );
    }

    #[test]
    fn read_section_directory_surfaces_checksum_mismatch_on_v2_header() {
        // A torn write (e.g. a crashed checkpoint) can leave the directory page
        // bytes inconsistent with the CRC the writer recorded in the active
        // header. The pre-fix wildcard match swallowed this, falling through to
        // v1 read logic that reported a misleading snapshot checksum mismatch.
        use grafeo_common::storage::SectionType;

        let dir = test_dir();
        let path = dir.path().join("torn_dir.grafeo");

        let directory_offset;
        {
            let manager = GrafeoFileManager::create(&path).unwrap();
            manager
                .write_sections(&[(SectionType::LpgStore, b"section payload")], 1, 1, 0, 0)
                .unwrap();
            directory_offset = manager.active_header().directory_offset;
        }

        // Flip a byte in the reserved area of the directory page (bytes 4-7).
        // The page still parses (count is intact, no entries change) but the
        // CRC over the page no longer matches the value in the active header.
        {
            let mut file = OpenOptions::new().write(true).open(&path).unwrap();
            file.seek(SeekFrom::Start(directory_offset + 4)).unwrap();
            file.write_all(&[0xAA]).unwrap();
        }

        let manager = GrafeoFileManager::open(&path).unwrap();
        let err = manager
            .read_section_directory()
            .expect_err("torn v2 directory must surface as Err, not Ok(None)");
        let msg = err.to_string();
        assert!(
            msg.contains("v2 section directory checksum mismatch"),
            "error should identify the directory CRC mismatch, got: {msg}"
        );
    }

    #[test]
    fn sidecar_wal_path_computation() {
        let dir = test_dir();
        let path = dir.path().join("mydb.grafeo");

        let manager = GrafeoFileManager::create(&path).unwrap();
        let wal_path = manager.sidecar_wal_path();

        assert_eq!(
            wal_path.file_name().unwrap().to_str().unwrap(),
            "mydb.grafeo.wal"
        );
        assert!(!manager.has_sidecar_wal());
    }

    #[test]
    fn sidecar_wal_detect_and_remove() {
        let dir = test_dir();
        let path = dir.path().join("test.grafeo");

        let manager = GrafeoFileManager::create(&path).unwrap();
        assert!(!manager.has_sidecar_wal());

        // Create sidecar directory manually (simulating engine behavior)
        fs::create_dir_all(manager.sidecar_wal_path()).unwrap();
        assert!(manager.has_sidecar_wal());

        // Remove it
        manager.remove_sidecar_wal().unwrap();
        assert!(!manager.has_sidecar_wal());
    }

    #[test]
    fn file_size_grows_with_data() {
        let dir = test_dir();
        let path = dir.path().join("test.grafeo");

        let manager = GrafeoFileManager::create(&path).unwrap();
        let empty_size = manager.file_size().unwrap();

        // Empty file should be at least 12 KiB (3 headers)
        assert!(empty_size >= DATA_OFFSET, "empty size: {empty_size}");

        let big_data = vec![0xAB; 100_000];
        manager.write_snapshot(&big_data, 1, 1, 0, 0).unwrap();

        let full_size = manager.file_size().unwrap();
        assert!(full_size > empty_size);
        assert_eq!(full_size, DATA_OFFSET + big_data.len() as u64);
    }

    #[test]
    fn exclusive_lock_prevents_second_open() {
        let dir = test_dir();
        let path = dir.path().join("locked.grafeo");

        let _manager1 = GrafeoFileManager::create(&path).unwrap();

        // Second open should fail
        let result = GrafeoFileManager::open(&path);
        assert!(result.is_err());
        assert!(result.err().unwrap().to_string().contains("locked"));
    }

    #[test]
    fn lock_released_after_close() {
        let dir = test_dir();
        let path = dir.path().join("lockclose.grafeo");

        let manager = GrafeoFileManager::create(&path).unwrap();
        manager.write_snapshot(b"data", 1, 1, 0, 0).unwrap();
        manager.close().unwrap();

        // Should succeed after close
        let manager2 = GrafeoFileManager::open(&path).unwrap();
        let data = manager2.read_snapshot().unwrap();
        assert_eq!(data, b"data");
    }

    #[test]
    fn lock_released_on_drop() {
        let dir = test_dir();
        let path = dir.path().join("lockdrop.grafeo");

        {
            let _manager = GrafeoFileManager::create(&path).unwrap();
            // Drop without explicit close
        }

        // Should succeed after drop
        let _manager2 = GrafeoFileManager::open(&path).unwrap();
    }

    #[test]
    fn checksum_mismatch_detected() {
        let dir = test_dir();
        let path = dir.path().join("test.grafeo");

        let manager = GrafeoFileManager::create(&path).unwrap();
        manager.write_snapshot(b"valid data", 1, 1, 0, 0).unwrap();
        drop(manager);

        // Corrupt the snapshot data in the file
        {
            let mut file = OpenOptions::new().write(true).open(&path).unwrap();
            file.seek(SeekFrom::Start(DATA_OFFSET)).unwrap();
            file.write_all(b"CORRUPT!!!").unwrap();
        }

        let manager = GrafeoFileManager::open(&path).unwrap();
        let result = manager.read_snapshot();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("checksum"));
    }

    #[test]
    fn open_read_only_reads_snapshot() {
        let dir = test_dir();
        let path = dir.path().join("ro.grafeo");

        // Create and write snapshot, then close
        {
            let manager = GrafeoFileManager::create(&path).unwrap();
            manager
                .write_snapshot(b"read-only test data", 3, 2, 5, 10)
                .unwrap();
            manager.close().unwrap();
        }

        // Open read-only
        let ro = GrafeoFileManager::open_read_only(&path).unwrap();
        assert!(ro.is_read_only());
        let data = ro.read_snapshot().unwrap();
        assert_eq!(data, b"read-only test data");

        let header = ro.active_header();
        assert_eq!(header.epoch, 3);
        assert_eq!(header.node_count, 5);
        assert_eq!(header.edge_count, 10);
    }

    #[test]
    fn read_only_rejects_write_snapshot() {
        let dir = test_dir();
        let path = dir.path().join("ro_write.grafeo");

        {
            let manager = GrafeoFileManager::create(&path).unwrap();
            manager.close().unwrap();
        }

        let ro = GrafeoFileManager::open_read_only(&path).unwrap();
        let result = ro.write_snapshot(b"nope", 1, 1, 0, 0);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("read-only"));
    }

    #[test]
    fn read_only_coexists_with_exclusive_after_close() {
        let dir = test_dir();
        let path = dir.path().join("coexist.grafeo");

        // Create, write, close
        {
            let manager = GrafeoFileManager::create(&path).unwrap();
            manager.write_snapshot(b"coexist data", 1, 1, 1, 1).unwrap();
            manager.close().unwrap();
        }

        // Two read-only opens should coexist
        let ro1 = GrafeoFileManager::open_read_only(&path).unwrap();
        let ro2 = GrafeoFileManager::open_read_only(&path).unwrap();

        assert_eq!(ro1.read_snapshot().unwrap(), b"coexist data");
        assert_eq!(ro2.read_snapshot().unwrap(), b"coexist data");
    }

    // ── Mmap section tests ─────────────────────────────────────────

    #[test]
    fn mmap_section_roundtrip() {
        use grafeo_common::storage::SectionType;

        let dir = test_dir();
        let path = dir.path().join("mmap.grafeo");

        let manager = GrafeoFileManager::create(&path).unwrap();

        // Write two sections: one data (LPG), one index (VectorStore)
        let lpg_data = b"lpg node data here";
        let vector_data = vec![0x42u8; 8192]; // 8 KiB of vector embeddings

        manager
            .write_sections(
                &[
                    (SectionType::LpgStore, lpg_data.as_slice()),
                    (SectionType::VectorStore, &vector_data),
                ],
                1,
                1,
                10,
                5,
            )
            .unwrap();

        // Read the directory to get entries
        let section_dir = manager.read_section_directory().unwrap().unwrap();

        // Mmap the VectorStore section (mmap-able)
        let vector_entry = section_dir.find(SectionType::VectorStore).unwrap();
        let mmap = manager.mmap_section(vector_entry).unwrap();

        assert_eq!(mmap.section_type(), SectionType::VectorStore);
        assert_eq!(mmap.len(), vector_data.len());
        assert_eq!(mmap.as_bytes(), &vector_data);
        assert!(!mmap.is_empty());
    }

    #[test]
    fn mmap_rejects_data_sections() {
        use grafeo_common::storage::SectionType;

        let dir = test_dir();
        let path = dir.path().join("mmap_reject.grafeo");

        let manager = GrafeoFileManager::create(&path).unwrap();
        manager
            .write_sections(&[(SectionType::LpgStore, b"data")], 1, 1, 1, 0)
            .unwrap();

        let section_dir = manager.read_section_directory().unwrap().unwrap();
        let lpg_entry = section_dir.find(SectionType::LpgStore).unwrap();

        let result = manager.mmap_section(lpg_entry);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not mmap-able"));
    }

    #[test]
    fn mmap_detects_corruption() {
        use grafeo_common::storage::SectionType;
        use std::io::Write as IoWrite;

        let dir = test_dir();
        let path = dir.path().join("mmap_corrupt.grafeo");

        let manager = GrafeoFileManager::create(&path).unwrap();
        let vector_data = vec![0xAB; 4096];
        manager
            .write_sections(&[(SectionType::VectorStore, &vector_data)], 1, 1, 0, 0)
            .unwrap();

        let section_dir = manager.read_section_directory().unwrap().unwrap();
        let entry = section_dir.find(SectionType::VectorStore).unwrap().clone();

        // Corrupt the section data by writing directly to the file
        drop(manager);
        {
            let mut file = OpenOptions::new().write(true).open(&path).unwrap();
            file.seek(SeekFrom::Start(entry.offset)).unwrap();
            file.write_all(b"CORRUPTED!").unwrap();
        }

        let manager = GrafeoFileManager::open(&path).unwrap();
        let result = manager.mmap_section(&entry);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("CRC mismatch"));
    }

    #[test]
    fn mmap_multiple_sections_coexist() {
        use grafeo_common::storage::SectionType;

        let dir = test_dir();
        let path = dir.path().join("mmap_multi.grafeo");

        let manager = GrafeoFileManager::create(&path).unwrap();

        let vector_data = vec![0x11; 4096];
        let text_data = vec![0x22; 2048];

        manager
            .write_sections(
                &[
                    (SectionType::VectorStore, &vector_data),
                    (SectionType::TextIndex, &text_data),
                ],
                1,
                1,
                0,
                0,
            )
            .unwrap();

        let section_dir = manager.read_section_directory().unwrap().unwrap();

        // Mmap both index sections simultaneously
        let vec_entry = section_dir.find(SectionType::VectorStore).unwrap();
        let text_entry = section_dir.find(SectionType::TextIndex).unwrap();

        let vec_mmap = manager.mmap_section(vec_entry).unwrap();
        let text_mmap = manager.mmap_section(text_entry).unwrap();

        // Both are valid and independent
        assert_eq!(vec_mmap.as_bytes(), &vector_data);
        assert_eq!(text_mmap.as_bytes(), &text_data);
        assert_eq!(vec_mmap.section_type(), SectionType::VectorStore);
        assert_eq!(text_mmap.section_type(), SectionType::TextIndex);
    }

    #[test]
    fn mmap_drop_then_checkpoint_lifecycle() {
        use grafeo_common::storage::SectionType;

        let dir = test_dir();
        let path = dir.path().join("mmap_lifecycle.grafeo");

        let manager = GrafeoFileManager::create(&path).unwrap();

        // Checkpoint 1: write vector section
        let vector_v1 = vec![0x11; 4096];
        manager
            .write_sections(&[(SectionType::VectorStore, &vector_v1)], 1, 1, 0, 0)
            .unwrap();

        // Mmap the section, read it
        let section_dir = manager.read_section_directory().unwrap().unwrap();
        let entry = section_dir.find(SectionType::VectorStore).unwrap();
        let mmap = manager.mmap_section(entry).unwrap();
        assert_eq!(mmap.as_bytes(), &vector_v1);

        // Drop the mmap before next checkpoint.
        // On Windows, writes fail if mmaps are still active (error 1224).
        // On all platforms, the intended lifecycle is: drop mmaps, checkpoint,
        // re-mmap. This keeps the flow simple and cross-platform.
        drop(mmap);

        // Checkpoint 2: write updated vector section
        let vector_v2 = vec![0x22; 8192];
        manager
            .write_sections(&[(SectionType::VectorStore, &vector_v2)], 2, 2, 0, 0)
            .unwrap();

        // Re-mmap the new section
        let section_dir = manager.read_section_directory().unwrap().unwrap();
        let entry = section_dir.find(SectionType::VectorStore).unwrap();
        let mmap = manager.mmap_section(entry).unwrap();
        assert_eq!(mmap.as_bytes(), &vector_v2);
        assert_eq!(mmap.len(), 8192);
    }

    #[test]
    fn mmap_section_debug_format() {
        use grafeo_common::storage::SectionType;

        let dir = test_dir();
        let path = dir.path().join("mmap_debug.grafeo");

        let manager = GrafeoFileManager::create(&path).unwrap();
        manager
            .write_sections(&[(SectionType::VectorStore, &[1, 2, 3, 4])], 1, 1, 0, 0)
            .unwrap();

        let section_dir = manager.read_section_directory().unwrap().unwrap();
        let entry = section_dir.find(SectionType::VectorStore).unwrap();
        let mmap = manager.mmap_section(entry).unwrap();

        let debug = format!("{mmap:?}");
        assert!(debug.contains("MmapSection"));
        assert!(debug.contains("VectorStore"));
    }

    #[test]
    fn path_returns_database_file_path() {
        let dir = test_dir();
        let path = dir.path().join("alix.grafeo");
        let manager = GrafeoFileManager::create(&path).unwrap();
        assert_eq!(manager.path(), path);
    }

    #[test]
    fn file_header_returns_valid_header() {
        use crate::file::format;
        let dir = test_dir();
        let path = dir.path().join("gus.grafeo");
        let manager = GrafeoFileManager::create(&path).unwrap();
        let header = manager.file_header();
        assert_eq!(header.magic, format::MAGIC);
        assert_eq!(header.format_version, format::FORMAT_VERSION);
    }

    #[test]
    fn sync_succeeds_for_writable_manager() {
        let dir = test_dir();
        let path = dir.path().join("vincent.grafeo");
        let manager = GrafeoFileManager::create(&path).unwrap();
        manager.write_snapshot(b"sync test", 1, 1, 5, 3).unwrap();
        manager.sync().unwrap();
    }

    #[test]
    fn sync_skips_for_read_only_manager() {
        let dir = test_dir();
        let path = dir.path().join("jules.grafeo");
        {
            let manager = GrafeoFileManager::create(&path).unwrap();
            manager.write_snapshot(b"ro sync", 1, 1, 0, 0).unwrap();
            manager.close().unwrap();
        }
        let ro = GrafeoFileManager::open_read_only(&path).unwrap();
        ro.sync().unwrap();
    }

    #[test]
    fn close_succeeds_for_read_only_manager() {
        let dir = test_dir();
        let path = dir.path().join("mia.grafeo");
        {
            let manager = GrafeoFileManager::create(&path).unwrap();
            manager.close().unwrap();
        }
        let ro = GrafeoFileManager::open_read_only(&path).unwrap();
        ro.close().unwrap();
    }

    #[test]
    fn remove_sidecar_wal_no_op_when_absent() {
        let dir = test_dir();
        let path = dir.path().join("django.grafeo");
        let manager = GrafeoFileManager::create(&path).unwrap();
        assert!(!manager.has_sidecar_wal());
        manager.remove_sidecar_wal().unwrap();
        assert!(!manager.has_sidecar_wal());
    }

    #[test]
    fn multiple_snapshots_alternate_slots() {
        let dir = test_dir();
        let path = dir.path().join("shosanna.grafeo");
        let manager = GrafeoFileManager::create(&path).unwrap();

        manager.write_snapshot(b"epoch one", 1, 1, 1, 0).unwrap();
        assert_eq!(manager.active_header().iteration, 1);

        manager.write_snapshot(b"epoch two", 2, 2, 2, 1).unwrap();
        assert_eq!(manager.active_header().iteration, 2);

        manager
            .write_snapshot(b"epoch three, longer data", 3, 3, 3, 2)
            .unwrap();
        assert_eq!(manager.active_header().iteration, 3);

        let loaded = manager.read_snapshot().unwrap();
        assert_eq!(loaded, b"epoch three, longer data");

        let header = manager.active_header();
        assert_eq!(header.epoch, 3);
        assert_eq!(header.node_count, 3);
        assert!(header.timestamp_ms > 0);
    }

    #[test]
    fn snapshot_truncates_stale_trailing_data() {
        let dir = test_dir();
        let path = dir.path().join("hans.grafeo");
        let manager = GrafeoFileManager::create(&path).unwrap();

        let large_data = vec![0xAA; 50_000];
        manager.write_snapshot(&large_data, 1, 1, 0, 0).unwrap();
        let size_after_large = manager.file_size().unwrap();

        let small_data = b"tiny";
        manager.write_snapshot(small_data, 2, 2, 0, 0).unwrap();
        let size_after_small = manager.file_size().unwrap();

        assert!(
            size_after_small < size_after_large,
            "file should shrink: {size_after_small} >= {size_after_large}"
        );
        assert_eq!(manager.read_snapshot().unwrap(), small_data);
    }

    #[test]
    fn open_read_only_fails_for_nonexistent_file() {
        let dir = test_dir();
        let path = dir.path().join("beatrix_missing.grafeo");
        assert!(GrafeoFileManager::open_read_only(&path).is_err());
    }

    #[test]
    fn copy_to_produces_identical_file() {
        let dir = test_dir();
        let src = dir.path().join("copy_src.grafeo");
        let dest = dir.path().join("copy_dest.grafeo");

        let manager = GrafeoFileManager::create(&src).unwrap();
        manager
            .write_snapshot(b"copy test payload", 5, 3, 10, 20)
            .unwrap();

        // copy_to reads through the locked handle (no new open)
        let bytes = manager.copy_to(&dest).unwrap();
        assert!(bytes > 0);

        // The original is still usable
        let snap = manager.read_snapshot().unwrap();
        assert_eq!(snap, b"copy test payload");
        manager.close().unwrap();

        // The copy is a valid .grafeo file
        let copy = GrafeoFileManager::open(&dest).unwrap();
        let snap = copy.read_snapshot().unwrap();
        assert_eq!(snap, b"copy test payload");

        let header = copy.active_header();
        assert_eq!(header.epoch, 5);
        assert_eq!(header.node_count, 10);
        assert_eq!(header.edge_count, 20);
        copy.close().unwrap();
    }

    #[test]
    fn copy_to_from_read_only_manager() {
        let dir = test_dir();
        let src = dir.path().join("ro_copy_src.grafeo");
        let dest = dir.path().join("ro_copy_dest.grafeo");

        {
            let manager = GrafeoFileManager::create(&src).unwrap();
            manager
                .write_snapshot(b"read-only copy data", 7, 4, 3, 1)
                .unwrap();
            manager.close().unwrap();
        }

        let ro = GrafeoFileManager::open_read_only(&src).unwrap();
        let bytes = ro.copy_to(&dest).unwrap();
        assert!(bytes > 0);

        let copy = GrafeoFileManager::open(&dest).unwrap();
        assert_eq!(copy.read_snapshot().unwrap(), b"read-only copy data");
        copy.close().unwrap();
    }

    #[test]
    #[cfg(all(feature = "encryption", not(miri)))]
    fn encrypted_section_roundtrip() {
        use grafeo_common::encryption::KeyChain;
        use grafeo_common::storage::SectionType;

        let dir = test_dir();
        let path = dir.path().join("encrypted.grafeo");

        let kc = KeyChain::new([0xAB; 32]);

        let section_data = b"sensitive graph data that must be encrypted";

        // Write with encryption
        {
            let mut manager = GrafeoFileManager::create(&path).unwrap();
            manager.set_section_encryptor(kc.encryptor_for("section", b"test"));
            manager
                .write_sections(&[(SectionType::LpgStore, &section_data[..])], 1, 0, 0, 0)
                .unwrap();
            manager.close().unwrap();
        }

        // Read back with same key
        {
            let mut manager = GrafeoFileManager::open(&path).unwrap();
            manager.set_section_encryptor(kc.encryptor_for("section", b"test"));
            let dir_opt = manager.read_section_directory().unwrap();
            let section_dir = dir_opt.expect("directory should exist");
            let entry = section_dir
                .entries()
                .iter()
                .find(|e| e.section_type == SectionType::LpgStore)
                .expect("LpgStore section should exist");
            let decrypted = manager.read_section_data(entry).unwrap();
            assert_eq!(decrypted, section_data);
        }
    }

    #[test]
    #[cfg(all(feature = "encryption", not(miri)))]
    fn encrypted_section_wrong_key_fails() {
        use grafeo_common::encryption::KeyChain;
        use grafeo_common::storage::SectionType;

        let dir = test_dir();
        let path = dir.path().join("wrong_key.grafeo");

        let kc_a = KeyChain::new([0xAA; 32]);
        let kc_b = KeyChain::new([0xBB; 32]);

        // Write with key A
        {
            let mut manager = GrafeoFileManager::create(&path).unwrap();
            manager.set_section_encryptor(kc_a.encryptor_for("section", b"test"));
            manager
                .write_sections(&[(SectionType::LpgStore, b"secret data")], 1, 0, 0, 0)
                .unwrap();
            manager.close().unwrap();
        }

        // Read with key B: CRC passes (computed on encrypted bytes), but decryption fails
        {
            let mut manager = GrafeoFileManager::open(&path).unwrap();
            manager.set_section_encryptor(kc_b.encryptor_for("section", b"test"));
            let dir_opt = manager.read_section_directory().unwrap();
            let section_dir = dir_opt.expect("directory should exist");
            let entry = section_dir
                .entries()
                .iter()
                .find(|e| e.section_type == SectionType::LpgStore)
                .expect("section should exist");
            let result = manager.read_section_data(entry);
            assert!(result.is_err(), "decryption with wrong key should fail");
        }
    }
}
