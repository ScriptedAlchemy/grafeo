//! Page checksums for a mapped section, verified the first time a page
//! is read.
//!
//! A sealed section is checksummed once, at write time, in fixed
//! `PAGE_SIZE` pages, and the page table travels in the section footer.
//! An open verifies the footer and the pages its parse reads; every other
//! page is verified on the first read that touches it. Opening a large
//! sealed container therefore costs what its first queries read, not a hash
//! of the whole file.
//!
//! A page that fails its checksum latches a [`fault`](SectionPages::fault):
//! the read that found it returns nothing, and every holder of the store can
//! observe the typed corruption instead of a silently missing value.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

use bytes::Bytes;

/// Checksum granularity. Small enough that a point read verifies little
/// beyond what it reads, large enough that the page table stays a few
/// kilobytes per hundred megabytes.
pub(crate) const PAGE_SIZE: usize = 64 * 1024;

/// Footer after the page table: `page_size u32, page_count u32, crc u32`,
/// the CRC covering the page table and the two counts.
const FOOTER_BYTES: usize = 12;

/// Folds section bytes into per-page CRCs as they are written.
pub(crate) struct PageCrcWriter {
    current: crc32fast::Hasher,
    filled: usize,
    crcs: Vec<u32>,
}

impl PageCrcWriter {
    pub(crate) fn new() -> Self {
        Self {
            current: crc32fast::Hasher::new(),
            filled: 0,
            crcs: Vec::new(),
        }
    }

    pub(crate) fn update(&mut self, mut bytes: &[u8]) {
        while !bytes.is_empty() {
            let take = (PAGE_SIZE - self.filled).min(bytes.len());
            self.current.update(&bytes[..take]);
            self.filled += take;
            bytes = &bytes[take..];
            if self.filled == PAGE_SIZE {
                let page = std::mem::replace(&mut self.current, crc32fast::Hasher::new());
                self.crcs.push(page.finalize());
                self.filled = 0;
            }
        }
    }

    /// The page table and footer covering everything passed to
    /// [`update`](Self::update).
    ///
    /// # Errors
    ///
    /// `Error::Internal` when the body needs more pages than the footer can
    /// count.
    pub(crate) fn finish(mut self) -> grafeo_common::utils::error::Result<Vec<u8>> {
        if self.filled > 0 {
            self.crcs.push(self.current.finalize());
        }
        let count = u32::try_from(self.crcs.len()).map_err(|_| {
            grafeo_common::utils::error::Error::Internal(format!(
                "compact section needs {} checksum pages, more than a u32 counts",
                self.crcs.len()
            ))
        })?;
        let mut out = Vec::with_capacity(self.crcs.len() * 4 + FOOTER_BYTES);
        for crc in &self.crcs {
            out.extend_from_slice(&crc.to_le_bytes());
        }
        // reason: PAGE_SIZE is a small constant
        #[allow(clippy::cast_possible_truncation)]
        out.extend_from_slice(&(PAGE_SIZE as u32).to_le_bytes());
        out.extend_from_slice(&count.to_le_bytes());
        let crc = crc32fast::hash(&out);
        out.extend_from_slice(&crc.to_le_bytes());
        Ok(out)
    }
}

/// The page table of a mapped section and which pages have been verified.
#[derive(Debug)]
pub struct SectionPages {
    /// Section bytes the pages cover (everything before the page table).
    body: Bytes,
    crcs: Bytes,
    verified: Box<[AtomicU64]>,
    fault: OnceLock<String>,
}

impl SectionPages {
    /// Reads the footer of `section` and returns the page table plus the
    /// covered body.
    ///
    /// # Errors
    ///
    /// A truncated footer, a footer CRC mismatch, or a page table that does
    /// not cover the body exactly.
    pub(crate) fn from_section(section: &Bytes) -> Result<Self, String> {
        let len = section.len();
        if len < FOOTER_BYTES {
            return Err("section too short for its page footer".into());
        }
        let footer = &section[len - FOOTER_BYTES..];
        let word = |at: usize| {
            u32::from_le_bytes([footer[at], footer[at + 1], footer[at + 2], footer[at + 3]])
        };
        let page_size = word(0) as usize;
        let page_count = word(4) as usize;
        let stored = word(8);
        if page_size != PAGE_SIZE {
            return Err(format!("unsupported section page size {page_size}"));
        }
        let table_len = page_count
            .checked_mul(4)
            .filter(|table| table + FOOTER_BYTES <= len)
            .ok_or("section page table exceeds the section")?;
        let table_start = len - FOOTER_BYTES - table_len;
        let computed = crc32fast::hash(&section[table_start..len - 4]);
        if computed != stored {
            return Err(format!(
                "section page table CRC mismatch: stored {stored:#010X}, computed {computed:#010X}"
            ));
        }
        if table_start.div_ceil(PAGE_SIZE) != page_count {
            return Err(format!(
                "section page table names {page_count} pages for a {table_start}-byte body"
            ));
        }
        Ok(Self {
            body: section.slice(..table_start),
            crcs: section.slice(table_start..table_start + table_len),
            verified: (0..page_count.div_ceil(64))
                .map(|_| AtomicU64::new(0))
                .collect(),
            fault: OnceLock::new(),
        })
    }

    /// The checksummed section body.
    pub(crate) fn body(&self) -> &Bytes {
        &self.body
    }

    /// Verifies every page overlapping `start..end` of the body that has not
    /// been verified yet. Returns `false`, latching the fault, when one does
    /// not match its checksum or the range leaves the body.
    pub(crate) fn verify(&self, start: usize, end: usize) -> bool {
        if start >= end {
            return true;
        }
        if end > self.body.len() {
            self.latch(format!(
                "read of bytes {start}..{end} past the {}-byte section body",
                self.body.len()
            ));
            return false;
        }
        for page in start / PAGE_SIZE..=(end - 1) / PAGE_SIZE {
            let (word, bit) = (page / 64, 1u64 << (page % 64));
            if self.verified[word].load(Ordering::Acquire) & bit != 0 {
                continue;
            }
            let from = page * PAGE_SIZE;
            let to = (from + PAGE_SIZE).min(self.body.len());
            let computed = crc32fast::hash(&self.body[from..to]);
            let at = page * 4;
            let stored = u32::from_le_bytes([
                self.crcs[at],
                self.crcs[at + 1],
                self.crcs[at + 2],
                self.crcs[at + 3],
            ]);
            if computed != stored {
                self.latch(format!(
                    "section page {page} CRC mismatch: stored {stored:#010X}, computed {computed:#010X}"
                ));
                return false;
            }
            self.verified[word].fetch_or(bit, Ordering::AcqRel);
        }
        true
    }

    /// Verifies the pages under `slice`, which must be a view into the body.
    pub(crate) fn verify_slice(&self, slice: &[u8]) -> bool {
        let base = self.body.as_ptr().addr();
        let start = slice.as_ptr().addr().wrapping_sub(base);
        match start.checked_add(slice.len()) {
            Some(end) if start <= self.body.len() => self.verify(start, end),
            _ => {
                self.latch("read outside the checksummed section body".into());
                false
            }
        }
    }

    /// The first checksum failure a read found, if any.
    #[must_use]
    pub fn fault(&self) -> Option<&str> {
        self.fault.get().map(String::as_str)
    }

    /// How many pages have been verified so far.
    #[must_use]
    pub fn verified_pages(&self) -> usize {
        self.verified
            .iter()
            .map(|word| word.load(Ordering::Acquire).count_ones() as usize)
            .sum()
    }

    /// Total pages in the table.
    #[must_use]
    pub fn page_count(&self) -> usize {
        self.crcs.len() / 4
    }

    /// Latches a fault found while decoding verified bytes. The first
    /// fault wins; later ones describe the same broken section.
    pub(crate) fn latch(&self, message: String) {
        let _ = self.fault.set(message);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sealed(body: &[u8]) -> Bytes {
        let mut writer = PageCrcWriter::new();
        writer.update(body);
        let mut out = body.to_vec();
        out.extend_from_slice(&writer.finish().unwrap());
        Bytes::from(out)
    }

    #[test]
    fn a_read_verifies_only_the_pages_it_touches() {
        let body: Vec<u8> = (0..(PAGE_SIZE * 3 + 17))
            .map(|i| u8::try_from(i % 251).unwrap())
            .collect();
        let pages = SectionPages::from_section(&sealed(&body)).unwrap();
        assert_eq!(pages.page_count(), 4);
        assert!(pages.verify(PAGE_SIZE + 5, PAGE_SIZE + 9));
        assert_eq!(pages.verified_pages(), 1);
        assert!(pages.verify(PAGE_SIZE * 3, PAGE_SIZE * 3 + 17));
        assert_eq!(pages.verified_pages(), 2);
        assert_eq!(pages.fault(), None);
    }

    #[test]
    fn a_flipped_byte_fails_the_read_that_touches_its_page_and_latches() {
        let body: Vec<u8> = (0..(PAGE_SIZE * 2))
            .map(|i| u8::try_from(i % 13).unwrap())
            .collect();
        let mut bytes = sealed(&body).to_vec();
        bytes[PAGE_SIZE + 100] ^= 0x40;
        let pages = SectionPages::from_section(&Bytes::from(bytes)).unwrap();
        assert!(
            pages.verify(0, 10),
            "the untouched first page still verifies"
        );
        assert!(!pages.verify(PAGE_SIZE + 99, PAGE_SIZE + 101));
        assert_eq!(
            pages
                .fault()
                .map(|fault| fault.starts_with("section page 1 CRC mismatch")),
            Some(true)
        );
    }

    #[test]
    fn a_corrupt_page_table_is_refused_at_open() {
        let body = vec![7u8; PAGE_SIZE + 1];
        let mut bytes = sealed(&body).to_vec();
        bytes[PAGE_SIZE + 2] ^= 1;
        let error = SectionPages::from_section(&Bytes::from(bytes)).unwrap_err();
        assert!(
            error.starts_with("section page table CRC mismatch"),
            "{error}"
        );
    }
}
