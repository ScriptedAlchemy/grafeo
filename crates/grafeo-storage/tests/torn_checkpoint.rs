//! Torn-checkpoint crash integrity for the v2 section container.
//!
//! A checkpoint writes the new generation's section data and directory
//! out-of-place and only then flips the header slot, so a process kill at
//! ANY byte of the checkpoint must leave the store openable at the
//! previous consistent generation with no CRC error. These tests replay
//! that guarantee byte-for-byte: they capture the file after generation A
//! commits and after generation B commits, then splice together the exact
//! on-disk state a kill at a given byte of B's data phase would leave
//! behind, and assert the store still opens at generation A.

#![cfg(feature = "grafeo-file")]

use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use grafeo_common::storage::{SectionDirectoryEntry, SectionType};
use grafeo_storage::container::SectionDirectory;
use grafeo_storage::container::directory::{DIRECTORY_OFFSET, SECTION_DATA_OFFSET};
use grafeo_storage::file::GrafeoFileManager;
use grafeo_storage::file::format::{DB_HEADER_SIZE, FILE_HEADER_SIZE};

/// Section types assigned per index; the directory dedupes by type, so each
/// section of a test generation needs a distinct one.
const SECTION_TYPES: [SectionType; 4] = [
    SectionType::Catalog,
    SectionType::LpgStore,
    SectionType::VectorStore,
    SectionType::TextIndex,
];

/// Deterministic section payload: `len` bytes seeded from `tag`.
fn payload(tag: u8, len: usize) -> Vec<u8> {
    // reason: `i % 251` always fits in u8
    #[allow(clippy::cast_possible_truncation)]
    (0..len).map(|i| tag.wrapping_add((i % 251) as u8)).collect()
}

/// Commits one generation via the production checkpoint path.
fn write_generation(mgr: &GrafeoFileManager, tag: u8, sizes: &[usize], epoch: u64) {
    let payloads: Vec<Vec<u8>> = sizes.iter().map(|&len| payload(tag, len)).collect();
    let sections: Vec<(SectionType, &[u8])> = payloads
        .iter()
        .enumerate()
        .map(|(i, data)| (SECTION_TYPES[i], data.as_slice()))
        .collect();
    mgr.write_sections(&sections, epoch, epoch, 0, 0).unwrap();
}

/// Asserts the store at `path` opens cleanly at the generation carrying
/// `tag`/`sizes`, with every section passing its CRC.
fn assert_opens_at_generation(path: &Path, expected_iteration: u64, tag: u8, sizes: &[usize]) {
    let mgr = GrafeoFileManager::open(path).unwrap();
    assert_eq!(
        mgr.active_header().iteration,
        expected_iteration,
        "store must open at the expected generation"
    );
    let dir = mgr
        .read_section_directory()
        .expect("directory must read without CRC error")
        .expect("committed generation must have a directory");
    assert_eq!(dir.len(), sizes.len());
    for (i, &len) in sizes.iter().enumerate() {
        let section_type = SECTION_TYPES[i];
        let entry = dir.find(section_type).unwrap().clone();
        let data = mgr.read_section_data(&entry).unwrap();
        assert_eq!(data, payload(tag, len), "section {section_type:?} content");
    }
    mgr.close().unwrap();
}

/// Byte extent `[lo, end)` of the active generation's region (sections plus
/// directory page), read through the production directory path.
fn generation_region(path: &Path) -> (u64, u64) {
    let mgr = GrafeoFileManager::open(path).unwrap();
    let dir_offset = mgr.active_header().directory_offset;
    assert_ne!(dir_offset, 0, "current writer records the directory offset");
    let dir = mgr.read_section_directory().unwrap().unwrap();
    let lo = dir
        .entries()
        .iter()
        .map(|e: &SectionDirectoryEntry| e.offset)
        .min()
        .unwrap()
        .min(dir_offset);
    let end = dir_offset + 4096;
    mgr.close().unwrap();
    (lo, end)
}

/// Two committed generations of distinct shape, returned as
/// `(dir, path, file_after_a, file_after_b)`.
fn two_generations() -> (tempfile::TempDir, PathBuf, Vec<u8>, Vec<u8>) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("store.grafeo");

    let mgr = GrafeoFileManager::create(&path).unwrap();
    write_generation(&mgr, 0xA1, &[3000, 9000], 1);
    mgr.close().unwrap();
    let file_a = std::fs::read(&path).unwrap();

    let mgr = GrafeoFileManager::open(&path).unwrap();
    write_generation(&mgr, 0xB2, &[5000, 12000, 700], 2);
    mgr.close().unwrap();
    let file_b = std::fs::read(&path).unwrap();

    (dir, path, file_a, file_b)
}

/// Reconstructs the on-disk state of a kill after exactly `torn_bytes` of
/// generation B's region (sections + directory) reached the file: the
/// post-A file plus a prefix of B's out-of-place region, headers untouched.
fn splice_tear(path: &Path, file_a: &[u8], file_b: &[u8], region_lo: u64, torn_bytes: u64) {
    std::fs::write(path, file_a).unwrap();
    if torn_bytes > 0 {
        let lo = usize::try_from(region_lo).unwrap();
        let torn = usize::try_from(torn_bytes).unwrap();
        let mut file = std::fs::OpenOptions::new().write(true).open(path).unwrap();
        file.seek(SeekFrom::Start(region_lo)).unwrap();
        file.write_all(&file_b[lo..lo + torn]).unwrap();
        file.sync_all().unwrap();
    }
}

#[test]
fn kill_at_any_byte_of_the_data_phase_reopens_at_previous_generation() {
    let (_dir, path, file_a, file_b) = two_generations();

    // Generation B was placed out-of-place (appended past A), so its region
    // is disjoint from A's live bytes by construction.
    std::fs::write(&path, &file_b).unwrap();
    let (b_lo, b_end) = generation_region(&path);
    assert!(
        b_lo >= file_a.len() as u64,
        "generation B ({b_lo:#X}..{b_end:#X}) must not overlap generation A \
         (file was {:#X} bytes after A committed)",
        file_a.len()
    );

    let region_len = b_end - b_lo;
    // Early / intra-section / mid / late / directory-page / everything-but-
    // the-header-flip tear points inside the section-data phase.
    let tear_points = [
        0,
        1,
        4096,
        region_len / 2,
        region_len - 4097, // last byte before B's directory page
        region_len - 4096, // directory page not yet written
        region_len - 1,    // directory page torn one byte short
        region_len,        // data + directory complete, header flip never ran
    ];
    for &torn_bytes in &tear_points {
        splice_tear(&path, &file_a, &file_b, b_lo, torn_bytes);
        assert_opens_at_generation(&path, 1, 0xA1, &[3000, 9000]);
    }
}

#[test]
fn torn_header_flip_falls_back_to_previous_generation() {
    let (_dir, path, _file_a, file_b) = two_generations();

    // Locate the slot generation B's flip wrote (iteration 2) and damage
    // its front while leaving its authentication tail intact — the state a
    // partial header-page write leaves behind.
    std::fs::write(&path, &file_b).unwrap();
    let slot0 = FILE_HEADER_SIZE;
    let slot1 = FILE_HEADER_SIZE + DB_HEADER_SIZE;
    let b_slot = [slot0, slot1]
        .into_iter()
        .find(|&off| {
            let bytes = &file_b[usize::try_from(off).unwrap()..];
            let (h, _): (grafeo_storage::file::DbHeader, _) =
                bincode::serde::decode_from_slice(bytes, bincode::config::standard()).unwrap();
            h.iteration == 2
        })
        .expect("one slot holds generation B's header");

    {
        let mut file = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        file.seek(SeekFrom::Start(b_slot)).unwrap();
        file.write_all(&[0xFF; 64]).unwrap();
        file.sync_all().unwrap();
    }

    // Generation B was appended past A, so A's bytes are still intact in
    // the post-B file; the damaged flip must fall back to them.
    assert_opens_at_generation(&path, 1, 0xA1, &[3000, 9000]);

    // A store that lost the torn generation must still accept the next
    // checkpoint and commit it durably.
    let mgr = GrafeoFileManager::open(&path).unwrap();
    write_generation(&mgr, 0xC3, &[2000], 3);
    mgr.close().unwrap();
    assert_opens_at_generation(&path, 2, 0xC3, &[2000]);
}

#[test]
fn kill_after_the_flip_keeps_the_new_generation() {
    let (_dir, path, _file_a, file_b) = two_generations();

    // A kill between the durable header flip and the reclaim truncation
    // leaves dead trailing bytes; the committed generation must win.
    let mut with_dead_tail = file_b.clone();
    with_dead_tail.extend_from_slice(&[0xEE; 8192]);
    std::fs::write(&path, &with_dead_tail).unwrap();
    assert_opens_at_generation(&path, 2, 0xB2, &[5000, 12000, 700]);
}

#[test]
fn generations_ping_pong_and_file_size_stays_bounded() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("pingpong.grafeo");

    let mgr = GrafeoFileManager::create(&path).unwrap();
    let sizes = [4000usize, 15000];
    for generation in 1..=6u64 {
        // reason: test generation counter is tiny
        #[allow(clippy::cast_possible_truncation)]
        write_generation(&mgr, generation as u8, &sizes, generation);
    }
    mgr.close().unwrap();

    // Each generation is sections (page-aligned) + one directory page; the
    // file must oscillate around one-to-two generations, not grow with
    // every checkpoint.
    let generation_size: u64 = sizes
        .iter()
        .map(|&len| (len as u64).div_ceil(4096) * 4096)
        .sum::<u64>()
        + 4096;
    let file_len = std::fs::metadata(&path).unwrap().len();
    assert!(
        file_len <= SECTION_DATA_OFFSET + 3 * generation_size,
        "file grew unbounded: {file_len} bytes after 6 checkpoints of {generation_size}-byte generations"
    );

    assert_opens_at_generation(&path, 6, 6, &sizes);
}

#[test]
fn store_with_legacy_fixed_directory_layout_still_opens() {
    // Replicates, byte for byte, a v2 store written by the previous
    // revision: sections in place from SECTION_DATA_OFFSET, the directory
    // at the fixed DIRECTORY_OFFSET page, and a header slot without the
    // authentication tail whose encoding predates `directory_offset`.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("legacy.grafeo");

    // Lay down the standard file header and empty slots.
    GrafeoFileManager::create(&path).unwrap().close().unwrap();

    let catalog = payload(0x11, 2500);
    let lpg = payload(0x11, 6000);

    let mut directory = SectionDirectory::new();
    let mut offset = SECTION_DATA_OFFSET;
    for (section_type, data) in [(SectionType::Catalog, &catalog), (SectionType::LpgStore, &lpg)] {
        directory
            .upsert(SectionDirectoryEntry {
                section_type,
                version: 1,
                flags: section_type.default_flags(),
                offset,
                length: data.len() as u64,
                checksum: crc32fast::hash(data),
            })
            .unwrap();
        offset = (offset + data.len() as u64).div_ceil(4096) * 4096;
    }

    // The legacy header encoding is the current struct minus the trailing
    // `directory_offset` field; an equally-typed tuple encodes identically.
    let legacy_header: (u64, u32, u64, u64, u64, u64, u64, u64) =
        (1, directory.checksum(), 0, 7, 7, 2, 0, 1_700_000_000_000);
    let encoded =
        bincode::serde::encode_to_vec(legacy_header, bincode::config::standard()).unwrap();
    let mut slot = vec![0u8; usize::try_from(DB_HEADER_SIZE).unwrap()];
    slot[..encoded.len()].copy_from_slice(&encoded);

    {
        let mut file = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        file.seek(SeekFrom::Start(DIRECTORY_OFFSET)).unwrap();
        file.write_all(&directory.to_bytes()).unwrap();
        for entry in directory.entries() {
            let data = if entry.section_type == SectionType::Catalog {
                &catalog
            } else {
                &lpg
            };
            file.seek(SeekFrom::Start(entry.offset)).unwrap();
            file.write_all(data).unwrap();
        }
        // Legacy writer targeted slot 1 for the first checkpoint.
        file.seek(SeekFrom::Start(FILE_HEADER_SIZE + DB_HEADER_SIZE))
            .unwrap();
        file.write_all(&slot).unwrap();
        file.sync_all().unwrap();
    }

    // The legacy store must open and read through the fixed-layout path.
    assert_opens_at_generation(&path, 1, 0x11, &[2500, 6000]);

    // A checkpoint on the legacy store must go out-of-place (never over
    // the live fixed-layout generation) and land readable.
    let mgr = GrafeoFileManager::open(&path).unwrap();
    write_generation(&mgr, 0x33, &[4000], 8);
    assert_ne!(mgr.active_header().directory_offset, 0);
    mgr.close().unwrap();
    assert_opens_at_generation(&path, 2, 0x33, &[4000]);
}

#[test]
fn full_round_trip_on_the_new_path() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("roundtrip.grafeo");

    let mgr = GrafeoFileManager::create(&path).unwrap();
    write_generation(&mgr, 0x77, &[1234, 56789, 42], 1);
    mgr.close().unwrap();

    assert_opens_at_generation(&path, 1, 0x77, &[1234, 56789, 42]);
}
