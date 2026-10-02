//! [`Section`](grafeo_common::storage::section::Section) implementation for [`CompactStore`].
//!
//! Serializes/deserializes a CompactStore to/from the `.grafeo` container
//! format with versioned headers and CRC32 integrity.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use bytes::Bytes;
use grafeo_common::storage::section::{Section, SectionType};
use grafeo_common::types::{EdgeId, NodeId, PropertyKey};
use grafeo_common::utils::error::Error;
use grafeo_common::utils::hash::FxHashMap;
use parking_lot::RwLock;

use super::CompactStore;
use super::column::{ColumnCodec, MappedDictionary, write_dictionary_regions};
use super::csr::CsrAdjacency;
use super::id_map::{IdMap, IdRecord, RECORD_BYTES};
use super::node_table::NodeTable;
use super::rel_table::RelTable;
use super::schema::{ColumnDef, ColumnType, EdgeSchema, TableSchema};
use super::value_order::RowOrder;
use super::zone_map::ZoneMap;
use crate::codec::pages::{PageCrcWriter, SectionPages};
use crate::codec::{BitVector, SectionSpan};
use crate::statistics::{EdgeTypeStatistics, LabelStatistics, Statistics};

/// Magic bytes identifying a CompactStore section.
const MAGIC: [u8; 4] = *b"GCST";

/// Current section format version. v4 versions the dictionary-entry
/// mapping (`dict_value`): in a v4 section, a dictionary entry beginning
/// with the marker prefix is a typed payload — a `Value::Bytes` hex body
/// or an escaped string — never a raw user string. The column byte
/// layout is identical to v3.
///
/// v5 adds a per-column null mask (flag byte + [`BitVector`]) ahead of
/// each node and edge column body, so a row the builder padded because
/// its node never had the property reads back as absent instead of as
/// `0`, `""`, or `false`. Column bodies are unchanged from v3/v4.
///
/// v6 is laid out to be served in place from a mapping: every bulk part
/// (column body, dictionary entries, row order, adjacency, id map) is a
/// region, described by a metadata block at the end, and the section is
/// checksummed in pages ([`SectionPages`]) instead of by one trailing CRC.
/// An open verifies the metadata and the regions it decodes; dictionary
/// entries, id maps, and the row orders of indexed columns are read in
/// place and verified page by page as reads first touch them.
pub(crate) const FORMAT_VERSION: u8 = 6;

/// v5 layout: v3/v4 columns with null masks, one trailing CRC.
const FORMAT_VERSION_V5: u8 = 5;

/// Section header: magic, version, flags.
const HEADER_BYTES: usize = 6;

/// v6 tail before the page table: metadata offset and length, LE u64.
const TAIL_BYTES: usize = 16;

/// First version that stores per-column null masks. Columns read from an
/// older section have no mask, so their padded rows still decode as the
/// padding value, exactly as they always did.
const NULL_MASKS_SINCE_VERSION: u8 = 5;

/// v4 layout: v3 column layout with marked dictionary entries.
const FORMAT_VERSION_V4: u8 = 4;

/// First version whose dictionaries use the marked-entry mapping.
///
/// Older sections stored every dictionary entry as a raw string, so a
/// legacy entry colliding with the marker prefix must not be trusted as
/// a marker: the reader escapes it at load instead
/// ([`ColumnCodec::escape_legacy_dict_markers`]).
const DICT_MARKERS_SINCE_VERSION: u8 = 4;

/// v3 (Phase 2c) layout: per-block zone maps in the column index for
/// skip pruning. Same column layout as v4, pre-marker dictionaries.
/// Files written by published 0.5.42 carry this byte.
const FORMAT_VERSION_V3: u8 = 3;

/// v2 (Phase 2b) layout: per-block index + bodies, no per-block stats.
/// Retained as a read-only compat path for one release.
const FORMAT_VERSION_V2: u8 = 2;

/// v1 layout: flat columns, no blocks. Retained as a read-only compat
/// path for one release. Files written by 0.5.41 and earlier carry
/// this byte.
const FORMAT_VERSION_V1: u8 = 1;

/// Wraps a [`CompactStore`] as a container [`Section`].
pub struct CompactStoreSection {
    store: RwLock<Option<Arc<CompactStore>>>,
    dirty: AtomicBool,
    /// Whether the store's section buffer is the heap copy
    /// [`Section::deserialize`] made, rather than bytes the caller handed
    /// to [`deserialize_from_bytes`](Self::deserialize_from_bytes).
    owns_section_copy: bool,
}

impl CompactStoreSection {
    /// Creates a new section wrapping an existing store.
    #[must_use]
    pub fn new(store: Arc<CompactStore>) -> Self {
        Self {
            store: RwLock::new(Some(store)),
            dirty: AtomicBool::new(false),
            owns_section_copy: false,
        }
    }

    /// Creates an empty section (for deserialization).
    #[must_use]
    pub fn empty() -> Self {
        Self {
            store: RwLock::new(None),
            dirty: AtomicBool::new(false),
            owns_section_copy: false,
        }
    }

    /// Whether `data` is a section that carries its own page checksums, so
    /// a mapped open may skip hashing it whole: reads verify each page the
    /// first time they touch it.
    #[must_use]
    pub fn is_page_checksummed(data: &[u8]) -> bool {
        data.starts_with(&MAGIC) && data.get(4) == Some(&FORMAT_VERSION)
    }

    /// Marks this section as dirty.
    pub fn mark_dirty(&self) {
        self.dirty.store(true, Ordering::Release);
    }

    /// Returns a reference to the inner store, if any.
    #[must_use]
    pub fn store(&self) -> Option<Arc<CompactStore>> {
        self.store.read().clone()
    }

    /// Deserializes from a refcounted [`Bytes`] buffer (Phase 3c).
    ///
    /// This is the zero-copy entry point: when `data` wraps a mmap
    /// region (via [`bytes::Bytes::from_owner`]), column codec storage
    /// is constructed via `data.slice(range)` rather than copying. The
    /// trait [`Section::deserialize`] entry point still works on
    /// `&[u8]` and incurs one heap copy (a single `Bytes::copy_from_slice`
    /// at the boundary).
    ///
    /// # Errors
    ///
    /// `Error::Storage(StorageError::Corruption)` for a section that does
    /// not decode or fails a checksum it verifies at open.
    pub fn deserialize_from_bytes(
        &mut self,
        data: bytes::Bytes,
    ) -> grafeo_common::utils::error::Result<()> {
        let store = deserialize_compact_store(&data).map_err(|e| {
            Error::Storage(grafeo_common::utils::error::StorageError::Corruption(
                format!("CompactStore deserialization failed: {e}"),
            ))
        })?;
        *self.store.write() = Some(Arc::new(store));
        self.owns_section_copy = false;
        Ok(())
    }

    /// Serializes at the requested format version.
    ///
    /// The default [`Section::serialize`] always writes [`FORMAT_VERSION`].
    /// This entry point is kept (test-only outside this crate) so the
    /// legacy compat readers can be exercised without keeping any
    /// externally committed fixtures. Legacy versions predate the marked
    /// dictionary-entry mapping, so a store whose Dict columns carry
    /// marked entries (any `Value::Bytes` property) is not meaningfully
    /// representable below [`DICT_MARKERS_SINCE_VERSION`]: the reader
    /// will escape those entries back into raw strings.
    pub(crate) fn serialize_with_version(
        &self,
        version: u8,
    ) -> grafeo_common::utils::error::Result<Vec<u8>> {
        // Size hint preserved from the pre-streaming implementation so
        // the `Vec` path still allocates once instead of doubling.
        let capacity = self
            .store
            .read()
            .as_ref()
            .map_or(0, |store| store.heap_bytes() + store.section_bytes());
        let mut out = Vec::with_capacity(capacity);
        self.serialize_into_with_version(&mut out, version)?;
        Ok(out)
    }

    /// Streams the section at the requested format version.
    ///
    /// The encoder still writes through `&mut Vec<u8>` helpers
    /// ([`write_len`], [`ColumnCodec::write_to_v3`],
    /// [`CsrAdjacency::write_to`]) that live in sibling modules, so a
    /// scratch buffer stays. What changed is its lifetime: it is drained
    /// to `sink` at every table boundary and reused, so the resident
    /// peak is the largest single table rather than the whole store.
    ///
    /// The trailing CRC is folded incrementally over each drained chunk,
    /// which covers exactly the same payload bytes as the previous
    /// one-shot `crc32fast::hash(&buf)`, so files written through either
    /// entry point are byte-identical.
    ///
    /// # Errors
    ///
    /// `Error::Internal` when the section holds no store, `Error::Io`
    /// when the sink rejects a write.
    pub(crate) fn serialize_into_with_version(
        &self,
        sink: &mut dyn std::io::Write,
        version: u8,
    ) -> grafeo_common::utils::error::Result<()> {
        let guard = self.store.read();
        let store = guard.as_ref().ok_or_else(|| {
            grafeo_common::utils::error::Error::Internal("no CompactStore to serialize".into())
        })?;
        if version == FORMAT_VERSION {
            return write_store(store, sink);
        }

        let mut stream = SectionStream::begin(sink, version, store.preserves_ids());

        stream.write_len(store.node_tables_by_id.len());
        for nt in &store.node_tables_by_id {
            let columns = sorted_by_key(nt.columns());
            stream.node_table(nt.label(), nt.len(), columns.len())?;
            let zone_maps = nt.zone_maps();
            for (key, codec) in columns {
                stream.node_column(
                    key,
                    zone_maps.get(key),
                    nt.null_mask(key),
                    codec,
                    nt.block_zone_maps().get(key).map(Vec::as_slice),
                )?;
            }
        }

        stream.write_len(store.rel_tables_by_id.len());
        for rt in &store.rel_tables_by_id {
            let properties = sorted_by_key(rt.properties());
            stream.rel_table(
                rt.edge_type().as_str(),
                rt.src_table_id(),
                rt.dst_table_id(),
                rt.fwd(),
                rt.bwd(),
                properties.len(),
            )?;
            for (key, codec) in properties {
                stream.rel_column(key, rt.null_mask(key), codec)?;
            }
        }

        if let Some((nodes, edges)) = store.id_maps() {
            stream.id_map(&id_records(nodes)?)?;
            stream.id_map(&id_records(edges)?)?;
        }

        stream.finish()
    }
}

/// An id map's records in ascending id order.
fn id_records(map: &IdMap) -> grafeo_common::utils::error::Result<Vec<IdRecord>> {
    (0..map.len())
        .map(|index| map.record(index).ok_or_else(unreadable_region))
        .collect()
}

/// A table's reverse ids in position order.
fn reverse_ids(map: &IdMap, table: usize) -> grafeo_common::utils::error::Result<Vec<u64>> {
    let table_id = u16::try_from(table).map_err(|_| unreadable_region())?;
    (0..map.table_len(table))
        .map(|position| {
            map.original(table_id, position as u64)
                .ok_or_else(unreadable_region)
        })
        .collect()
}

fn unreadable_region() -> Error {
    Error::Internal("a mapped compact store region failed its page checksum".into())
}

/// Writes a resident store as a current-version section.
fn write_store(
    store: &CompactStore,
    sink: &mut dyn std::io::Write,
) -> grafeo_common::utils::error::Result<()> {
    let mut writer = SectionWriter::begin(sink, store.preserves_ids())?;
    let indexes = store.property_value_indexes.read();

    writer.count(store.node_tables_by_id.len());
    for nt in &store.node_tables_by_id {
        let columns = sorted_by_key(nt.columns());
        writer.node_table(nt.label(), nt.len(), columns.len())?;
        for (key, codec) in columns {
            let order = indexes.get(key).and_then(|index| {
                index
                    .iter()
                    .find(|(table_id, _)| *table_id == nt.table_id())
                    .map(|(_, order)| order)
            });
            writer.node_column(
                key,
                nt.zone_maps().get(key),
                nt.null_mask(key),
                codec,
                nt.block_zone_maps().get(key).map(Vec::as_slice),
                order,
            )?;
        }
    }

    writer.count(store.rel_tables_by_id.len());
    for rt in &store.rel_tables_by_id {
        let properties = sorted_by_key(rt.properties());
        writer.rel_table(
            rt.edge_type().as_str(),
            rt.src_table_id(),
            rt.dst_table_id(),
            rt.fwd(),
            rt.bwd(),
            properties.len(),
        )?;
        for (key, codec) in properties {
            writer.rel_column(key, rt.null_mask(key), codec)?;
        }
    }

    if let Some((nodes, edges)) = store.id_maps() {
        for map in [nodes, edges] {
            writer.id_records(&id_records(map)?)?;
            writer.count(map.table_count());
            for table in 0..map.table_count() {
                writer.reverse_ids(&reverse_ids(map, table)?)?;
            }
        }
    }
    drop(indexes);
    writer.finish()
}

/// The encoder of the current section layout.
///
/// Bulk parts are written to the sink as regions the moment they are
/// encoded; the metadata that names them is small and collected until
/// [`finish`](Self::finish) appends it, the tail that locates it, and the
/// page table. A resident [`CompactStore`] and the incremental builder's
/// spooled rows both serialize through it, so a section written either
/// way is the same bytes, and the resident scratch is one column.
pub(crate) struct SectionWriter<'a> {
    sink: &'a mut dyn std::io::Write,
    pages: PageCrcWriter,
    offset: u64,
    meta: Vec<u8>,
    scratch: Vec<u8>,
}

impl<'a> SectionWriter<'a> {
    /// Writes the section header.
    pub(crate) fn begin(
        sink: &'a mut dyn std::io::Write,
        preserves_ids: bool,
    ) -> grafeo_common::utils::error::Result<Self> {
        let mut writer = Self {
            sink,
            pages: PageCrcWriter::new(),
            offset: 0,
            meta: Vec::new(),
            scratch: Vec::with_capacity(CHUNK_TARGET_BYTES),
        };
        let mut header = MAGIC.to_vec();
        header.push(FORMAT_VERSION);
        header.push(u8::from(preserves_ids));
        writer.write_raw(&header)?;
        Ok(writer)
    }

    fn write_raw(&mut self, bytes: &[u8]) -> grafeo_common::utils::error::Result<()> {
        self.pages.update(bytes);
        self.sink.write_all(bytes)?;
        self.offset += bytes.len() as u64;
        Ok(())
    }

    /// Writes `bytes` as a region and names it in the metadata.
    fn region(&mut self, bytes: &[u8]) -> grafeo_common::utils::error::Result<()> {
        write_u64(&mut self.meta, self.offset);
        write_u64(&mut self.meta, bytes.len() as u64);
        self.write_raw(bytes)
    }

    /// Writes the scratch buffer as a region and clears it.
    fn scratch_region(&mut self) -> grafeo_common::utils::error::Result<()> {
        let scratch = std::mem::take(&mut self.scratch);
        let written = self.region(&scratch);
        self.scratch = scratch;
        self.scratch.clear();
        written
    }

    /// Writes a table or entry count into the metadata.
    pub(crate) fn count(&mut self, count: usize) {
        write_len(&mut self.meta, count);
    }

    /// Opens a node table whose `column_count` columns follow in ascending
    /// key order.
    pub(crate) fn node_table(
        &mut self,
        label: &str,
        rows: usize,
        column_count: usize,
    ) -> grafeo_common::utils::error::Result<()> {
        write_name(&mut self.meta, label)?;
        write_len(&mut self.meta, rows);
        write_len(&mut self.meta, column_count);
        Ok(())
    }

    /// Writes one node column, with the row order that indexes it when the
    /// property is indexed.
    pub(crate) fn node_column(
        &mut self,
        key: &PropertyKey,
        zone_map: Option<&ZoneMap>,
        null_mask: Option<&BitVector>,
        codec: &ColumnCodec,
        block_zone_maps: Option<&[ZoneMap]>,
        order: Option<&RowOrder>,
    ) -> grafeo_common::utils::error::Result<()> {
        write_name(&mut self.meta, key.as_str())?;
        if let Some(zm) = zone_map {
            self.meta.push(1);
            write_zone_map(&mut self.meta, zm);
        } else {
            self.meta.push(0);
        }
        self.column_body(null_mask, codec, block_zone_maps)?;
        match order {
            Some(order) => {
                self.meta.push(1);
                order.write_to(&mut self.scratch);
                self.scratch_region()
            }
            None => {
                self.meta.push(0);
                Ok(())
            }
        }
    }

    /// A column's body region (null mask and codec), then a Dict column's
    /// entry regions.
    fn column_body(
        &mut self,
        null_mask: Option<&BitVector>,
        codec: &ColumnCodec,
        block_zone_maps: Option<&[ZoneMap]>,
    ) -> grafeo_common::utils::error::Result<()> {
        write_null_mask(&mut self.scratch, FORMAT_VERSION, null_mask)?;
        codec.write_blocked(&mut self.scratch, block_zone_maps, false);
        self.scratch_region()?;
        match codec {
            ColumnCodec::Dict(dict) => {
                self.meta.push(1);
                let mut starts = Vec::with_capacity(dict.dictionary_size() * 4);
                write_dictionary_regions(dict, &mut starts, &mut self.scratch)?;
                self.region(&starts)?;
                self.scratch_region()
            }
            _ => {
                self.meta.push(0);
                Ok(())
            }
        }
    }

    /// Writes a relationship table's adjacency and opens its
    /// `property_count` columns, which follow in ascending key order.
    pub(crate) fn rel_table(
        &mut self,
        edge_type: &str,
        src_table_id: u16,
        dst_table_id: u16,
        fwd: &CsrAdjacency,
        bwd: Option<&CsrAdjacency>,
        property_count: usize,
    ) -> grafeo_common::utils::error::Result<()> {
        write_name(&mut self.meta, edge_type)?;
        write_u16(&mut self.meta, src_table_id);
        write_u16(&mut self.meta, dst_table_id);
        fwd.write_to(&mut self.scratch);
        if let Some(bwd) = bwd {
            self.scratch.push(1);
            bwd.write_to(&mut self.scratch);
        } else {
            self.scratch.push(0);
        }
        self.scratch_region()?;
        write_len(&mut self.meta, property_count);
        Ok(())
    }

    /// Writes one relationship property column.
    pub(crate) fn rel_column(
        &mut self,
        key: &PropertyKey,
        null_mask: Option<&BitVector>,
        codec: &ColumnCodec,
    ) -> grafeo_common::utils::error::Result<()> {
        write_name(&mut self.meta, key.as_str())?;
        self.column_body(null_mask, codec, None)
    }

    /// Writes an id map's records, ascending by id, as one region.
    pub(crate) fn id_records(
        &mut self,
        records: &[IdRecord],
    ) -> grafeo_common::utils::error::Result<()> {
        write_u64(&mut self.meta, self.offset);
        write_u64(&mut self.meta, (records.len() * RECORD_BYTES) as u64);
        for &(id, table, position) in records {
            write_u64(&mut self.scratch, id);
            write_u16(&mut self.scratch, table);
            write_u64(&mut self.scratch, position);
            self.drain_scratch(false)?;
        }
        self.drain_scratch(true)
    }

    /// Writes one table's ids in position order as one region.
    pub(crate) fn reverse_ids(&mut self, ids: &[u64]) -> grafeo_common::utils::error::Result<()> {
        write_u64(&mut self.meta, self.offset);
        write_u64(&mut self.meta, (ids.len() * 8) as u64);
        for &id in ids {
            write_u64(&mut self.scratch, id);
            self.drain_scratch(false)?;
        }
        self.drain_scratch(true)
    }

    /// Writes the scratch buffer as part of the open region once it is
    /// full, or whenever `force` is set.
    fn drain_scratch(&mut self, force: bool) -> grafeo_common::utils::error::Result<()> {
        if self.scratch.is_empty() || (!force && self.scratch.len() < CHUNK_TARGET_BYTES) {
            return Ok(());
        }
        let scratch = std::mem::take(&mut self.scratch);
        let written = self.write_raw(&scratch);
        self.scratch = scratch;
        self.scratch.clear();
        written
    }

    /// Appends the metadata, the tail locating it, and the page table.
    pub(crate) fn finish(mut self) -> grafeo_common::utils::error::Result<()> {
        let meta = std::mem::take(&mut self.meta);
        let mut tail = Vec::with_capacity(TAIL_BYTES);
        write_u64(&mut tail, self.offset);
        write_u64(&mut tail, meta.len() as u64);
        self.write_raw(&meta)?;
        self.write_raw(&tail)?;
        let footer = self.pages.finish()?;
        self.sink.write_all(&footer)?;
        Ok(())
    }
}

/// The one encoder of the CompactStore section layout, driven in layout
/// order: header, node tables with their columns, relationship tables with
/// their adjacency and columns, id maps, CRC.
///
/// A resident [`CompactStore`] and the incremental builder's spooled rows
/// both serialize through it, so a section written either way is the same
/// bytes. Each column is drained to the sink as soon as it is written, so
/// the resident scratch is one column, and a caller that produces columns
/// one at a time never holds more than the column it is writing.
pub(crate) struct SectionStream<'a> {
    sink: &'a mut dyn std::io::Write,
    buf: Vec<u8>,
    crc: crc32fast::Hasher,
    version: u8,
}

impl<'a> SectionStream<'a> {
    /// Writes the section header.
    pub(crate) fn begin(
        sink: &'a mut dyn std::io::Write,
        version: u8,
        preserves_ids: bool,
    ) -> Self {
        // Starts small and grows to the largest column; `drain_chunk`
        // clears without releasing capacity, so later columns reuse it.
        let mut buf: Vec<u8> = Vec::with_capacity(CHUNK_TARGET_BYTES);
        buf.extend_from_slice(&MAGIC);
        buf.push(version);
        buf.push(u8::from(preserves_ids));
        Self {
            sink,
            buf,
            crc: crc32fast::Hasher::new(),
            version,
        }
    }

    /// Writes a table count.
    pub(crate) fn write_len(&mut self, len: usize) {
        write_len(&mut self.buf, len);
    }

    /// Opens a node table whose `column_count` columns follow in ascending
    /// key order.
    pub(crate) fn node_table(
        &mut self,
        label: &str,
        rows: usize,
        column_count: usize,
    ) -> grafeo_common::utils::error::Result<()> {
        write_name(&mut self.buf, label)?;
        write_len(&mut self.buf, rows);
        write_len(&mut self.buf, column_count);
        Ok(())
    }

    /// Writes one node column and drains it.
    pub(crate) fn node_column(
        &mut self,
        key: &PropertyKey,
        zone_map: Option<&ZoneMap>,
        null_mask: Option<&BitVector>,
        codec: &ColumnCodec,
        block_zone_maps: Option<&[ZoneMap]>,
    ) -> grafeo_common::utils::error::Result<()> {
        write_name(&mut self.buf, key.as_str())?;
        if let Some(zm) = zone_map {
            self.buf.push(1);
            write_zone_map(&mut self.buf, zm);
        } else {
            self.buf.push(0);
        }
        write_null_mask(&mut self.buf, self.version, null_mask)?;
        write_codec(codec, &mut self.buf, self.version, block_zone_maps);
        // Column granularity: a single wide column is the smallest unit
        // this encoder can emit without rewriting the sibling-module
        // writers.
        drain_chunk(&mut self.buf, &mut *self.sink, &mut self.crc, false)
    }

    /// Writes a relationship table's adjacency, drains it, and opens its
    /// `property_count` columns, which follow in ascending key order.
    pub(crate) fn rel_table(
        &mut self,
        edge_type: &str,
        src_table_id: u16,
        dst_table_id: u16,
        fwd: &CsrAdjacency,
        bwd: Option<&CsrAdjacency>,
        property_count: usize,
    ) -> grafeo_common::utils::error::Result<()> {
        write_name(&mut self.buf, edge_type)?;
        write_u16(&mut self.buf, src_table_id);
        write_u16(&mut self.buf, dst_table_id);
        fwd.write_to(&mut self.buf);
        if let Some(bwd) = bwd {
            self.buf.push(1);
            bwd.write_to(&mut self.buf);
        } else {
            self.buf.push(0);
        }
        drain_chunk(&mut self.buf, &mut *self.sink, &mut self.crc, false)?;
        write_len(&mut self.buf, property_count);
        Ok(())
    }

    /// Writes one relationship property column and drains it. Edge
    /// columns carry no per-block zone maps.
    pub(crate) fn rel_column(
        &mut self,
        key: &PropertyKey,
        null_mask: Option<&BitVector>,
        codec: &ColumnCodec,
    ) -> grafeo_common::utils::error::Result<()> {
        write_name(&mut self.buf, key.as_str())?;
        write_null_mask(&mut self.buf, self.version, null_mask)?;
        write_codec(codec, &mut self.buf, self.version, None);
        drain_chunk(&mut self.buf, &mut *self.sink, &mut self.crc, false)
    }

    /// Writes one id map: `(id, table id, offset)` entries in ascending id
    /// order.
    pub(crate) fn id_map(
        &mut self,
        entries: &[(u64, u16, u64)],
    ) -> grafeo_common::utils::error::Result<()> {
        write_len(&mut self.buf, entries.len());
        for &(id, table, offset) in entries {
            write_u64(&mut self.buf, id);
            write_u16(&mut self.buf, table);
            write_u64(&mut self.buf, offset);
            drain_chunk(&mut self.buf, &mut *self.sink, &mut self.crc, false)?;
        }
        Ok(())
    }

    /// Flushes what is left, then the CRC over everything written.
    pub(crate) fn finish(mut self) -> grafeo_common::utils::error::Result<()> {
        drain_chunk(&mut self.buf, &mut *self.sink, &mut self.crc, true)?;
        let crc = std::mem::take(&mut self.crc).finalize();
        self.sink.write_all(&crc.to_le_bytes())?;
        Ok(())
    }
}

/// A table's columns in ascending key order.
///
/// The column maps hash with a per-instance random seed, so writing them in
/// walk order would make two serializations of the same store differ byte
/// for byte; the reader rebuilds a map, so any order decodes identically.
fn sorted_by_key(
    columns: &FxHashMap<PropertyKey, ColumnCodec>,
) -> Vec<(&PropertyKey, &ColumnCodec)> {
    let mut sorted: Vec<(&PropertyKey, &ColumnCodec)> = columns.iter().collect();
    sorted.sort_unstable_by(|(a, _), (b, _)| a.as_str().cmp(b.as_str()));
    sorted
}

/// Target scratch size before a drain. Large enough that a per-row
/// `drain_chunk` call in the id-map loops is a predictable-branch no-op,
/// small enough that the peak stays bounded.
const CHUNK_TARGET_BYTES: usize = 1 << 20;

/// Moves `buf` into `sink`, folding it into `crc` on the way, and clears
/// `buf` without releasing its capacity.
///
/// A no-op unless `force` is set or the buffer has reached
/// [`CHUNK_TARGET_BYTES`], so callers can invoke it on every row.
fn drain_chunk(
    buf: &mut Vec<u8>,
    sink: &mut dyn std::io::Write,
    crc: &mut crc32fast::Hasher,
    force: bool,
) -> grafeo_common::utils::error::Result<()> {
    if buf.is_empty() || (!force && buf.len() < CHUNK_TARGET_BYTES) {
        return Ok(());
    }
    crc.update(buf);
    sink.write_all(buf)?;
    buf.clear();
    Ok(())
}

/// Writes a single column codec body using the layout matching the
/// section's format version.
///
/// - v1 = flat columns (legacy)
/// - v2 = per-block index + concatenated bodies, no stats
/// - v3+ = v2 layout + inline per-block zone map per index entry (v4
///   shares the byte layout and differs only in dictionary-entry
///   semantics, marked by the header version byte)
///
/// `block_stats_hint` is consulted only at v3+; when `None` or with a
/// mismatched length, [`ColumnCodec::write_to_v3`] computes the stats
/// from the column itself.
fn write_codec(
    codec: &ColumnCodec,
    buf: &mut Vec<u8>,
    version: u8,
    block_stats_hint: Option<&[ZoneMap]>,
) {
    match version {
        FORMAT_VERSION_V1 => codec.write_to(buf),
        FORMAT_VERSION_V2 => codec.write_to_v2(buf),
        _ => codec.write_to_v3(buf, block_stats_hint),
    }
}

/// Writes a column's null mask as a presence byte followed, when present,
/// by the [`BitVector`] bytes. Nothing is written below
/// [`NULL_MASKS_SINCE_VERSION`]: those layouts have no slot for it, and a
/// reader of that version pads exactly as it did before masks existed.
fn write_null_mask(
    buf: &mut Vec<u8>,
    version: u8,
    mask: Option<&BitVector>,
) -> grafeo_common::utils::error::Result<()> {
    if version < NULL_MASKS_SINCE_VERSION {
        return Ok(());
    }
    match mask {
        Some(mask) => {
            buf.push(1);
            buf.extend_from_slice(&mask.to_bytes()?);
        }
        None => buf.push(0),
    }
    Ok(())
}

/// Reads the null mask [`write_null_mask`] wrote, when the section version
/// carries one.
fn read_null_mask(data: &[u8], pos: &mut usize, version: u8) -> Result<Option<BitVector>, String> {
    if version < NULL_MASKS_SINCE_VERSION {
        return Ok(None);
    }
    let present = *data.get(*pos).ok_or("truncated null mask flag")?;
    *pos += 1;
    match present {
        0 => Ok(None),
        1 => {
            let mask = BitVector::from_bytes(&data[*pos..]).map_err(|e| e.to_string())?;
            *pos += 4 + mask.word_count() * 8;
            Ok(Some(mask))
        }
        other => Err(format!("invalid null mask flag {other}")),
    }
}

impl Section for CompactStoreSection {
    fn section_type(&self) -> SectionType {
        SectionType::CompactStore
    }

    fn version(&self) -> u8 {
        FORMAT_VERSION
    }

    fn serialize(&self) -> grafeo_common::utils::error::Result<Vec<u8>> {
        self.serialize_with_version(FORMAT_VERSION)
    }

    fn serialize_into(
        &self,
        sink: &mut dyn std::io::Write,
    ) -> grafeo_common::utils::error::Result<()> {
        self.serialize_into_with_version(sink, FORMAT_VERSION)
    }

    fn deserialize(&mut self, data: &[u8]) -> grafeo_common::utils::error::Result<()> {
        // Heap-copy entry point (Section trait). Phase 3c adds
        // [`deserialize_from_bytes`](Self::deserialize_from_bytes) which
        // skips the copy on the mmap path.
        let owned = bytes::Bytes::copy_from_slice(data);
        self.deserialize_from_bytes(owned)?;
        self.owns_section_copy = true;
        Ok(())
    }

    fn is_dirty(&self) -> bool {
        self.dirty.load(Ordering::Acquire)
    }

    fn mark_clean(&self) {
        self.dirty.store(false, Ordering::Release);
    }

    fn memory_usage(&self) -> usize {
        self.store.read().as_ref().map_or(0, |store| {
            store.heap_bytes()
                + if self.owns_section_copy {
                    store.section_bytes()
                } else {
                    0
                }
        })
    }
}

/// A CompactStore section written straight from an
/// [`IncrementalCompactStoreBuilder`], one column at a time, without first
/// building the [`CompactStore`] it describes.
///
/// Write-only and single-use: the first serialization consumes the rows, so
/// a second one — or a read — is refused rather than answered with a
/// different or empty section.
pub struct IncrementalCompactStoreSection {
    builder: parking_lot::Mutex<Option<super::IncrementalCompactStoreBuilder>>,
    indexed_properties: Vec<PropertyKey>,
}

impl IncrementalCompactStoreSection {
    /// Wraps the rows to be written, storing a row order beside each node
    /// column of `indexed_properties` so the written store opens with
    /// those properties already indexed.
    #[must_use]
    pub fn new(
        builder: super::IncrementalCompactStoreBuilder,
        indexed_properties: Vec<PropertyKey>,
    ) -> Self {
        Self {
            builder: parking_lot::Mutex::new(Some(builder)),
            indexed_properties,
        }
    }
}

impl Section for IncrementalCompactStoreSection {
    fn section_type(&self) -> SectionType {
        SectionType::CompactStore
    }

    fn version(&self) -> u8 {
        FORMAT_VERSION
    }

    fn serialize(&self) -> grafeo_common::utils::error::Result<Vec<u8>> {
        let mut out = Vec::new();
        self.serialize_into(&mut out)?;
        Ok(out)
    }

    fn serialize_into(
        &self,
        sink: &mut dyn std::io::Write,
    ) -> grafeo_common::utils::error::Result<()> {
        let builder = self.builder.lock().take().ok_or_else(|| {
            grafeo_common::utils::error::Error::Internal(
                "incremental compact section was already written".into(),
            )
        })?;
        builder.write_section(sink, &self.indexed_properties)
    }

    fn deserialize(&mut self, _data: &[u8]) -> grafeo_common::utils::error::Result<()> {
        Err(grafeo_common::utils::error::Error::Internal(
            "incremental compact section is write-only".into(),
        ))
    }

    fn is_dirty(&self) -> bool {
        self.builder.lock().is_some()
    }

    fn mark_clean(&self) {}

    fn memory_usage(&self) -> usize {
        0
    }
}

// ── Deserialization ────────────────────────────────────────────────

/// Reads a single column codec body, dispatching by section version.
///
/// Dictionaries read from sections older than
/// [`DICT_MARKERS_SINCE_VERSION`] are normalized on the way in: their
/// entries are raw strings, so any entry colliding with the marker prefix
/// is escaped before the always-on marker decoding can retype it.
///
/// - v1 → [`ColumnCodec::read_from`] (flat layout, no per-block stats)
/// - v2 → [`ColumnCodec::read_from_v2`] (block index, no stats)
/// - v3/v4 → [`ColumnCodec::read_from_v3`] (block index + per-block stats)
///
/// Returns the codec and an `Option<Vec<ZoneMap>>` carrying per-block
/// stats when the v3/v4 path was taken.
fn read_codec(
    data: &Bytes,
    pos: &mut usize,
    version: u8,
) -> Result<(ColumnCodec, Option<Vec<ZoneMap>>), String> {
    let (mut codec, stats) = match version {
        FORMAT_VERSION_V1 => ColumnCodec::read_from(data, pos)
            .map(|c| (c, None))
            .map_err(|e| e.to_string())?,
        FORMAT_VERSION_V2 => ColumnCodec::read_from_v2(data, pos)
            .map(|c| (c, None))
            .map_err(|e| e.to_string())?,
        FORMAT_VERSION_V3 | FORMAT_VERSION_V4 | FORMAT_VERSION_V5 => {
            ColumnCodec::read_from_v3(data, pos)
                .map(|(c, stats)| (c, Some(stats)))
                .map_err(|e| e.to_string())?
        }
        _ => return Err(format!("unsupported CompactStore version {version}")),
    };
    if version < DICT_MARKERS_SINCE_VERSION {
        codec.escape_legacy_dict_markers();
    }
    Ok((codec, stats))
}

fn deserialize_compact_store(data_bytes: &bytes::Bytes) -> Result<CompactStore, String> {
    match data_bytes.get(4) {
        Some(&FORMAT_VERSION) if data_bytes.starts_with(&MAGIC) => deserialize_paged(data_bytes),
        _ => deserialize_legacy(data_bytes),
    }
}

/// One column as a section describes it.
struct ParsedColumn {
    key: String,
    zone_map: Option<ZoneMap>,
    null_mask: Option<BitVector>,
    codec: ColumnCodec,
    block_stats: Option<Vec<ZoneMap>>,
    order: Option<RowOrder>,
}

struct ParsedNodeTable {
    label: String,
    rows: usize,
    columns: Vec<ParsedColumn>,
}

struct ParsedRelTable {
    edge_type: String,
    src_tid: u16,
    dst_tid: u16,
    fwd: CsrAdjacency,
    bwd: Option<CsrAdjacency>,
    columns: Vec<ParsedColumn>,
}

/// Builds the store the parsed tables describe: tables, lookups,
/// statistics, and the property indexes whose row orders the section
/// carried.
fn assemble(node_tables: Vec<ParsedNodeTable>, rel_tables: Vec<ParsedRelTable>) -> CompactStore {
    let mut tables = Vec::with_capacity(node_tables.len());
    let mut label_to_table_id: FxHashMap<arcstr::ArcStr, u16> = FxHashMap::default();
    let mut table_id_to_label: Vec<arcstr::ArcStr> = Vec::with_capacity(node_tables.len());
    let mut indexes: FxHashMap<PropertyKey, Vec<(u16, RowOrder)>> = FxHashMap::default();

    for (table_idx, parsed) in node_tables.into_iter().enumerate() {
        let table_id = u16::try_from(table_idx).unwrap_or(0);
        let label = arcstr::ArcStr::from(parsed.label.as_str());
        let mut columns: FxHashMap<PropertyKey, ColumnCodec> = FxHashMap::default();
        let mut zone_maps: FxHashMap<PropertyKey, ZoneMap> = FxHashMap::default();
        let mut block_zone_maps: FxHashMap<PropertyKey, Vec<ZoneMap>> = FxHashMap::default();
        let mut null_masks: FxHashMap<PropertyKey, BitVector> = FxHashMap::default();
        let mut col_defs = Vec::with_capacity(parsed.columns.len());
        for column in parsed.columns {
            let key = PropertyKey::new(&column.key);
            if let Some(zm) = column.zone_map {
                zone_maps.insert(key.clone(), zm);
            }
            if let Some(mask) = column.null_mask {
                null_masks.insert(key.clone(), mask);
            }
            if let Some(stats) = column.block_stats {
                block_zone_maps.insert(key.clone(), stats);
            }
            if let Some(order) = column.order {
                indexes
                    .entry(key.clone())
                    .or_default()
                    .push((table_id, order));
            }
            col_defs.push(ColumnDef::new(
                &column.key,
                infer_column_type_from_codec(&column.codec),
            ));
            columns.insert(key, column.codec);
        }
        let schema = TableSchema::new(label.as_str(), table_id, col_defs);
        let table = NodeTable::from_columns_with_block_stats(
            schema,
            columns,
            zone_maps,
            block_zone_maps,
            parsed.rows,
        )
        .with_null_masks(null_masks);
        tables.push(table);
        label_to_table_id.insert(label.clone(), table_id);
        table_id_to_label.push(label);
    }

    let mut rels = Vec::with_capacity(rel_tables.len());
    let mut edge_type_to_rel_id: FxHashMap<arcstr::ArcStr, Vec<u16>> = FxHashMap::default();
    let mut rel_table_id_to_type: Vec<arcstr::ArcStr> = Vec::with_capacity(rel_tables.len());
    for (rel_idx, parsed) in rel_tables.into_iter().enumerate() {
        let rel_table_id = u16::try_from(rel_idx).unwrap_or(0);
        let edge_type = arcstr::ArcStr::from(parsed.edge_type.as_str());
        let mut properties: FxHashMap<PropertyKey, ColumnCodec> = FxHashMap::default();
        let mut null_masks: FxHashMap<PropertyKey, BitVector> = FxHashMap::default();
        let mut prop_defs = Vec::with_capacity(parsed.columns.len());
        for column in parsed.columns {
            let key = PropertyKey::new(&column.key);
            if let Some(mask) = column.null_mask {
                null_masks.insert(key.clone(), mask);
            }
            prop_defs.push(ColumnDef::new(
                &column.key,
                infer_column_type_from_codec(&column.codec),
            ));
            properties.insert(key, column.codec);
        }
        let src_label = table_id_to_label
            .get(parsed.src_tid as usize)
            .cloned()
            .unwrap_or_default();
        let dst_label = table_id_to_label
            .get(parsed.dst_tid as usize)
            .cloned()
            .unwrap_or_default();
        let schema = EdgeSchema::new(
            edge_type.as_str(),
            rel_table_id,
            src_label.as_str(),
            dst_label.as_str(),
            prop_defs,
        );
        let table = RelTable::new(
            schema,
            parsed.fwd,
            parsed.bwd,
            properties,
            parsed.src_tid,
            parsed.dst_tid,
        )
        .with_null_masks(null_masks);
        edge_type_to_rel_id
            .entry(edge_type.clone())
            .or_default()
            .push(rel_table_id);
        rel_table_id_to_type.push(edge_type);
        rels.push(table);
    }

    let mut stats = Statistics::new();
    let mut total_nodes = 0u64;
    let mut total_edges = 0u64;
    for (idx, nt) in tables.iter().enumerate() {
        let c = nt.len() as u64;
        total_nodes += c;
        stats.update_label(table_id_to_label[idx].as_str(), LabelStatistics::new(c));
    }
    let mut edge_counts: FxHashMap<&str, u64> = FxHashMap::default();
    for (idx, rt) in rels.iter().enumerate() {
        let c = rt.num_edges() as u64;
        total_edges += c;
        *edge_counts
            .entry(rel_table_id_to_type[idx].as_str())
            .or_default() += c;
    }
    for (et, count) in edge_counts {
        stats.update_edge_type(et, EdgeTypeStatistics::new(count, 0.0, 0.0));
    }
    stats.total_nodes = total_nodes;
    stats.total_edges = total_edges;

    let mut store = CompactStore::new(
        tables,
        label_to_table_id,
        rels,
        edge_type_to_rel_id,
        table_id_to_label,
        rel_table_id_to_type,
        stats,
    );
    store.attach_property_indexes(indexes);
    store
}

/// Reads a section written before the paged layout: one trailing CRC over
/// everything, columns and id maps inline.
fn deserialize_legacy(data_bytes: &bytes::Bytes) -> Result<CompactStore, String> {
    let data: &[u8] = data_bytes.as_ref();
    if data.len() < 10 {
        return Err("data too short for CompactStore section".into());
    }

    // Verify CRC32.
    let payload = &data[..data.len() - 4];
    let stored_crc = u32::from_le_bytes([
        data[data.len() - 4],
        data[data.len() - 3],
        data[data.len() - 2],
        data[data.len() - 1],
    ]);
    let computed_crc = crc32fast::hash(payload);
    if stored_crc != computed_crc {
        return Err(format!(
            "CRC32 mismatch: stored {stored_crc:#010X}, computed {computed_crc:#010X}"
        ));
    }

    let mut pos = 0;

    // Header.
    if data[pos..pos + 4] != MAGIC {
        return Err("bad magic".into());
    }
    pos += 4;
    let version = data[pos];
    pos += 1;
    if !matches!(
        version,
        FORMAT_VERSION_V5
            | FORMAT_VERSION_V4
            | FORMAT_VERSION_V3
            | FORMAT_VERSION_V2
            | FORMAT_VERSION_V1
    ) {
        return Err(format!(
            "unsupported CompactStore section version {version} (supported: {FORMAT_VERSION_V1}, {FORMAT_VERSION_V2}, {FORMAT_VERSION_V3}, {FORMAT_VERSION_V4}, {FORMAT_VERSION_V5}, {FORMAT_VERSION})"
        ));
    }
    let flags = data[pos];
    pos += 1;
    let preserves_ids = flags & 0x01 != 0;

    let num_node_tables = read_u32(data, &mut pos)? as usize;
    let mut node_tables = Vec::with_capacity(num_node_tables);
    for _ in 0..num_node_tables {
        let label = read_string(data, &mut pos)?;
        let rows = read_u32(data, &mut pos)? as usize;
        let num_cols = read_u32(data, &mut pos)? as usize;
        let mut columns = Vec::with_capacity(num_cols);
        for _ in 0..num_cols {
            let key = read_string(data, &mut pos)?;
            let has_zm = *data.get(pos).ok_or("truncated zone map flag")?;
            pos += 1;
            let zone_map = if has_zm == 1 {
                Some(read_zone_map(data, &mut pos)?)
            } else {
                None
            };
            let null_mask = read_null_mask(data, &mut pos, version)?;
            let (codec, block_stats) =
                read_codec(data_bytes, &mut pos, version).map_err(|e| format!("codec: {e}"))?;
            columns.push(ParsedColumn {
                key,
                zone_map,
                null_mask,
                codec,
                block_stats,
                order: None,
            });
        }
        node_tables.push(ParsedNodeTable {
            label,
            rows,
            columns,
        });
    }

    let num_rel_tables = read_u32(data, &mut pos)? as usize;
    let mut rel_tables = Vec::with_capacity(num_rel_tables);
    for _ in 0..num_rel_tables {
        let edge_type = read_string(data, &mut pos)?;
        let src_tid = read_u16(data, &mut pos)?;
        let dst_tid = read_u16(data, &mut pos)?;
        let fwd = CsrAdjacency::read_from(data, &mut pos).map_err(|e| format!("fwd CSR: {e}"))?;
        let has_bwd = *data.get(pos).ok_or("truncated bwd flag")?;
        pos += 1;
        let bwd = if has_bwd == 1 {
            Some(CsrAdjacency::read_from(data, &mut pos).map_err(|e| format!("bwd CSR: {e}"))?)
        } else {
            None
        };
        let num_props = read_u32(data, &mut pos)? as usize;
        let mut columns = Vec::with_capacity(num_props);
        for _ in 0..num_props {
            let key = read_string(data, &mut pos)?;
            let null_mask = read_null_mask(data, &mut pos, version)?;
            let (codec, _block_stats) = read_codec(data_bytes, &mut pos, version)
                .map_err(|e| format!("edge codec: {e}"))?;
            columns.push(ParsedColumn {
                key,
                zone_map: None,
                null_mask,
                codec,
                block_stats: None,
                order: None,
            });
        }
        rel_tables.push(ParsedRelTable {
            edge_type,
            src_tid,
            dst_tid,
            fwd,
            bwd,
            columns,
        });
    }

    let mut store = assemble(node_tables, rel_tables);
    if preserves_ids {
        let nodes = read_legacy_id_records(data, &mut pos)?;
        let edges = read_legacy_id_records(data, &mut pos)?;
        store.attach_id_maps(
            IdMap::from_records(
                nodes,
                store.node_tables_by_id.len(),
                NodeId::INVALID.as_u64(),
            ),
            IdMap::from_records(
                edges,
                store.rel_tables_by_id.len(),
                EdgeId::INVALID.as_u64(),
            ),
        );
    }
    store.section = SectionSpan::of(data_bytes);
    Ok(store)
}

/// Reads one inline id map: a count, then `(id, table, position)` records
/// in ascending id order.
fn read_legacy_id_records(data: &[u8], pos: &mut usize) -> Result<Vec<IdRecord>, String> {
    let count = read_u32(data, pos)? as usize;
    let mut records = Vec::with_capacity(count);
    for _ in 0..count {
        let id = read_u64(data, pos)?;
        let table = read_u16(data, pos)?;
        let position = read_u64(data, pos)?;
        records.push((id, table, position));
    }
    Ok(records)
}

/// Reads a paged section in place.
///
/// The footer, header, and metadata are verified first; then each region
/// the open decodes (column bodies, null masks, adjacency) is verified
/// before it is read. Dictionary entries, row orders, and id maps are
/// handed to the store as unverified views and checked page by page as
/// reads first touch them.
fn deserialize_paged(data_bytes: &bytes::Bytes) -> Result<CompactStore, String> {
    let pages = Arc::new(SectionPages::from_section(data_bytes)?);
    let body = pages.body().clone();
    let verify = |start: usize, end: usize| -> Result<(), String> {
        if pages.verify(start, end) {
            Ok(())
        } else {
            Err(pages
                .fault()
                .unwrap_or("section page check failed")
                .to_owned())
        }
    };
    if body.len() < HEADER_BYTES + TAIL_BYTES {
        return Err("paged CompactStore section is too short".into());
    }
    verify(0, HEADER_BYTES)?;
    if body[..4] != MAGIC || body[4] != FORMAT_VERSION {
        return Err("bad paged CompactStore section header".into());
    }
    let preserves_ids = body[5] & 0x01 != 0;

    let tail = body.len() - TAIL_BYTES;
    verify(tail, body.len())?;
    let mut tail_pos = tail;
    let meta_start = usize::try_from(read_u64(&body, &mut tail_pos)?)
        .map_err(|_| "metadata offset exceeds the address space")?;
    let meta_len = usize::try_from(read_u64(&body, &mut tail_pos)?)
        .map_err(|_| "metadata length exceeds the address space")?;
    if meta_start < HEADER_BYTES || meta_start.checked_add(meta_len) != Some(tail) {
        return Err("paged CompactStore metadata is not where the tail places it".into());
    }
    verify(meta_start, tail)?;
    let meta = &body[meta_start..tail];
    let mut pos = 0;

    // A region named by the metadata: its bytes, unverified.
    let region = |pos: &mut usize| -> Result<Bytes, String> {
        let start = usize::try_from(read_u64(meta, pos)?).map_err(|_| "region offset overflow")?;
        let len = usize::try_from(read_u64(meta, pos)?).map_err(|_| "region length overflow")?;
        match start.checked_add(len) {
            Some(end) if start >= HEADER_BYTES && end <= meta_start => Ok(body.slice(start..end)),
            _ => Err("region outside the section body".into()),
        }
    };
    // A region the open decodes now: verified before it is read.
    let eager = |pos: &mut usize| -> Result<Bytes, String> {
        let bytes = region(pos)?;
        if pages.verify_slice(&bytes) {
            Ok(bytes)
        } else {
            Err(pages
                .fault()
                .unwrap_or("section page check failed")
                .to_owned())
        }
    };
    let column =
        |pos: &mut usize, key: String, zone_map: Option<ZoneMap>| -> Result<ParsedColumn, String> {
            let body = eager(pos)?;
            let has_dictionary = *meta.get(*pos).ok_or("truncated dictionary flag")?;
            *pos += 1;
            let mapped_dictionary = match has_dictionary {
                0 => None,
                1 => {
                    let starts = region(pos)?;
                    let entries = region(pos)?;
                    Some(MappedDictionary {
                        entries,
                        starts,
                        pages: Arc::clone(&pages),
                    })
                }
                other => return Err(format!("invalid dictionary flag {other}")),
            };
            let mut at = 0;
            let null_mask = read_null_mask(&body, &mut at, FORMAT_VERSION)?;
            let (codec, block_stats) = ColumnCodec::read_blocked(&body, &mut at, mapped_dictionary)
                .map_err(|e| format!("codec {key}: {e}"))?;
            if at != body.len() {
                return Err(format!(
                    "column {key} body has {} trailing bytes",
                    body.len() - at
                ));
            }
            Ok(ParsedColumn {
                key,
                zone_map,
                null_mask,
                codec,
                block_stats: Some(block_stats),
                order: None,
            })
        };

    let num_node_tables = read_u32(meta, &mut pos)? as usize;
    let mut node_tables = Vec::with_capacity(num_node_tables);
    for _ in 0..num_node_tables {
        let label = read_string(meta, &mut pos)?;
        let rows = read_u32(meta, &mut pos)? as usize;
        let num_cols = read_u32(meta, &mut pos)? as usize;
        let mut columns = Vec::with_capacity(num_cols);
        for _ in 0..num_cols {
            let key = read_string(meta, &mut pos)?;
            let has_zm = *meta.get(pos).ok_or("truncated zone map flag")?;
            pos += 1;
            let zone_map = match has_zm {
                0 => None,
                1 => Some(read_zone_map(meta, &mut pos)?),
                other => return Err(format!("invalid zone map flag {other}")),
            };
            let mut parsed = column(&mut pos, key, zone_map)?;
            let has_order = *meta.get(pos).ok_or("truncated row order flag")?;
            pos += 1;
            parsed.order = match has_order {
                0 => None,
                1 => {
                    let order = RowOrder::mapped(region(&mut pos)?, &pages)?;
                    if order.len() > rows {
                        return Err(format!(
                            "row order of {} names more rows than {rows}",
                            parsed.key
                        ));
                    }
                    Some(order)
                }
                other => return Err(format!("invalid row order flag {other}")),
            };
            columns.push(parsed);
        }
        node_tables.push(ParsedNodeTable {
            label,
            rows,
            columns,
        });
    }

    let num_rel_tables = read_u32(meta, &mut pos)? as usize;
    let mut rel_tables = Vec::with_capacity(num_rel_tables);
    for _ in 0..num_rel_tables {
        let edge_type = read_string(meta, &mut pos)?;
        let src_tid = read_u16(meta, &mut pos)?;
        let dst_tid = read_u16(meta, &mut pos)?;
        let adjacency = eager(&mut pos)?;
        let mut at = 0;
        let fwd =
            CsrAdjacency::read_from(&adjacency, &mut at).map_err(|e| format!("fwd CSR: {e}"))?;
        let has_bwd = *adjacency.get(at).ok_or("truncated bwd flag")?;
        at += 1;
        let bwd = match has_bwd {
            0 => None,
            1 => Some(
                CsrAdjacency::read_from(&adjacency, &mut at)
                    .map_err(|e| format!("bwd CSR: {e}"))?,
            ),
            other => return Err(format!("invalid bwd flag {other}")),
        };
        if at != adjacency.len() {
            return Err(format!("adjacency of {edge_type} has trailing bytes"));
        }
        let num_props = read_u32(meta, &mut pos)? as usize;
        let mut columns = Vec::with_capacity(num_props);
        for _ in 0..num_props {
            let key = read_string(meta, &mut pos)?;
            let mut parsed = column(&mut pos, key, None)?;
            parsed.block_stats = None;
            columns.push(parsed);
        }
        rel_tables.push(ParsedRelTable {
            edge_type,
            src_tid,
            dst_tid,
            fwd,
            bwd,
            columns,
        });
    }

    let mut store = assemble(node_tables, rel_tables);
    if preserves_ids {
        let id_map = |pos: &mut usize, tables: usize| -> Result<IdMap, String> {
            let records = region(pos)?;
            let reverse_count = read_u32(meta, pos)? as usize;
            if reverse_count != tables {
                return Err(format!(
                    "id map covers {reverse_count} tables, the section holds {tables}"
                ));
            }
            let reverse = (0..reverse_count)
                .map(|_| region(pos))
                .collect::<Result<Vec<_>, _>>()?;
            IdMap::mapped(records, reverse, &pages)
        };
        let nodes = id_map(&mut pos, store.node_tables_by_id.len())?;
        let edges = id_map(&mut pos, store.rel_tables_by_id.len())?;
        store.attach_id_maps(nodes, edges);
    }
    if pos != meta.len() {
        return Err(format!(
            "section metadata has {} trailing bytes",
            meta.len() - pos
        ));
    }
    store.section = SectionSpan::of(data_bytes);
    store.pages = Some(pages);
    Ok(store)
}

// ── Write helpers ──────────────────────────────────────────────────

fn write_u16(buf: &mut Vec<u8>, v: u16) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn write_u64(buf: &mut Vec<u8>, v: u64) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn write_len(buf: &mut Vec<u8>, v: usize) {
    let n = u32::try_from(v).expect("length exceeds u32::MAX in compact section");
    buf.extend_from_slice(&n.to_le_bytes());
}

/// Writes a table label, property key, or edge type behind its `u16` length.
///
/// # Errors
///
/// Returns [`Error::InvalidValue`] for a name longer than `u16::MAX` bytes.
fn write_name(buf: &mut Vec<u8>, name: &str) -> grafeo_common::utils::error::Result<()> {
    let len = u16::try_from(name.len()).map_err(|_| {
        Error::InvalidValue(format!(
            "a compact section name is limited to {} bytes, got {}",
            u16::MAX,
            name.len()
        ))
    })?;
    write_u16(buf, len);
    buf.extend_from_slice(name.as_bytes());
    Ok(())
}

fn write_zone_map(buf: &mut Vec<u8>, zm: &ZoneMap) {
    write_len(buf, zm.null_count);
    write_len(buf, zm.row_count);
    // Encode min/max as (tag, value) pairs.
    write_optional_value(buf, &zm.min);
    write_optional_value(buf, &zm.max);
}

fn write_optional_value(buf: &mut Vec<u8>, v: &Option<grafeo_common::types::Value>) {
    match v {
        None => buf.push(0),
        Some(grafeo_common::types::Value::Int64(n)) => {
            buf.push(1);
            // Store as raw i64 bytes to avoid sign-loss lint.
            buf.extend_from_slice(&n.to_le_bytes());
        }
        Some(grafeo_common::types::Value::Bool(b)) => {
            buf.push(2);
            buf.push(u8::from(*b));
        }
        Some(grafeo_common::types::Value::String(s)) => match u16::try_from(s.len()) {
            Ok(len) => {
                buf.push(3);
                write_u16(buf, len);
                buf.extend_from_slice(s.as_bytes());
            }
            // The column holds the value whole; only this bound, which the
            // u16 length cannot carry, is written absent, and an absent
            // bound never prunes.
            Err(_) => buf.push(0),
        },
        Some(_) => {
            // Unsupported type for zone map: write as absent.
            buf.push(0);
        }
    }
}

// ── Read helpers ───────────────────────────────────────────────────

fn read_u16(data: &[u8], pos: &mut usize) -> Result<u16, String> {
    if *pos + 2 > data.len() {
        return Err("truncated u16".into());
    }
    let v = u16::from_le_bytes([data[*pos], data[*pos + 1]]);
    *pos += 2;
    Ok(v)
}

fn read_u32(data: &[u8], pos: &mut usize) -> Result<u32, String> {
    if *pos + 4 > data.len() {
        return Err("truncated u32".into());
    }
    let v = u32::from_le_bytes([data[*pos], data[*pos + 1], data[*pos + 2], data[*pos + 3]]);
    *pos += 4;
    Ok(v)
}

fn read_u64(data: &[u8], pos: &mut usize) -> Result<u64, String> {
    if *pos + 8 > data.len() {
        return Err("truncated u64".into());
    }
    let v = u64::from_le_bytes(data[*pos..*pos + 8].try_into().unwrap());
    *pos += 8;
    Ok(v)
}

fn read_string(data: &[u8], pos: &mut usize) -> Result<String, String> {
    let slen = read_u16(data, pos)? as usize;
    if *pos + slen > data.len() {
        return Err("truncated string".into());
    }
    let s =
        std::str::from_utf8(&data[*pos..*pos + slen]).map_err(|_| "invalid UTF-8".to_string())?;
    *pos += slen;
    Ok(s.to_string())
}

fn read_zone_map(data: &[u8], pos: &mut usize) -> Result<ZoneMap, String> {
    let null_count = read_u32(data, pos)? as usize;
    let row_count = read_u32(data, pos)? as usize;
    let min = read_optional_value(data, pos)?;
    let max = read_optional_value(data, pos)?;
    Ok(ZoneMap {
        min,
        max,
        null_count,
        row_count,
    })
}

fn read_optional_value(
    data: &[u8],
    pos: &mut usize,
) -> Result<Option<grafeo_common::types::Value>, String> {
    let tag = *data.get(*pos).ok_or("truncated value tag")?;
    *pos += 1;
    match tag {
        0 => Ok(None),
        1 => {
            // Read raw i64 bytes (written via i64::to_le_bytes).
            if *pos + 8 > data.len() {
                return Err("truncated i64 value".into());
            }
            let v = i64::from_le_bytes(data[*pos..*pos + 8].try_into().unwrap());
            *pos += 8;
            Ok(Some(grafeo_common::types::Value::Int64(v)))
        }
        2 => {
            let b = *data.get(*pos).ok_or("truncated bool")?;
            *pos += 1;
            Ok(Some(grafeo_common::types::Value::Bool(b != 0)))
        }
        3 => {
            let s = read_string(data, pos)?;
            Ok(Some(grafeo_common::types::Value::String(
                arcstr::ArcStr::from(s.as_str()),
            )))
        }
        _ => Err(format!("unknown value tag {tag}")),
    }
}

fn infer_column_type_from_codec(codec: &ColumnCodec) -> ColumnType {
    match codec {
        ColumnCodec::BitPacked(bp) => ColumnType::UInt {
            bits: bp.bits_per_value(),
        },
        ColumnCodec::Dict(_) => ColumnType::DictString,
        ColumnCodec::Bitmap(_) => ColumnType::Bool,
        ColumnCodec::Int8Vector { dimensions, .. } => ColumnType::Int8Vector {
            dimensions: *dimensions,
        },
        ColumnCodec::Float64(_) => ColumnType::Float64,
        ColumnCodec::Float32Vector { dimensions, .. } => ColumnType::Float32Vector {
            dimensions: *dimensions,
        },
        ColumnCodec::RawI64(_) => ColumnType::Int64,
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::compact::from_graph_store_preserving_ids;
    use crate::graph::lpg::LpgStore;
    use crate::graph::traits::GraphStore;
    use grafeo_common::types::Value;
    use grafeo_common::utils::hash::FxHashSet;

    /// Builds a store big enough that the streaming encoder has to drain
    /// its scratch buffer several times, so the chunk boundaries — and
    /// therefore the incremental CRC — are actually exercised.
    fn multi_chunk_store() -> CompactStore {
        let store = LpgStore::new().unwrap();
        // ~512 B of string per node over 8k nodes ≈ 4 MiB of column data,
        // comfortably past CHUNK_TARGET_BYTES.
        let filler = "x".repeat(512);
        let mut ids = Vec::new();
        for i in 0..8_192i64 {
            let id = store.create_node(&["Person"]);
            store.set_node_property(id, "name", Value::from(format!("{filler}-{i}").as_str()));
            store.set_node_property(id, "age", Value::Int64(i));
            ids.push(id);
        }
        for w in ids.windows(2) {
            store.create_edge(w[0], w[1], "KNOWS");
        }
        from_graph_store_preserving_ids(&store).unwrap()
    }

    /// A container written through the sink path must be bit-for-bit
    /// what the `Vec` path would have produced, chunk boundaries and
    /// incrementally-folded CRC included.
    #[test]
    fn compact_sink_and_vec_paths_are_byte_identical() {
        let section = CompactStoreSection::new(Arc::new(multi_chunk_store()));

        let via_vec = section.serialize().expect("serialize");
        let mut via_sink = Vec::new();
        section
            .serialize_into(&mut via_sink)
            .expect("serialize_into");

        assert!(
            via_vec.len() > CHUNK_TARGET_BYTES,
            "fixture must exceed one chunk to prove anything: {} bytes",
            via_vec.len()
        );
        assert_eq!(via_sink.len(), via_vec.len(), "byte counts must match");
        assert!(via_sink == via_vec, "sink and Vec bytes must be identical");

        // The streamed bytes must still deserialize, which also
        // re-verifies the trailing CRC.
        let mut restored = CompactStoreSection::empty();
        restored.deserialize(&via_sink).expect("deserialize");
        assert_eq!(restored.store().unwrap().node_count(), 8_192);
    }

    #[test]
    fn preserved_edge_ids_keep_native_endpoints_after_reopen() {
        let source = LpgStore::new().unwrap();
        let from = source.create_node(&["Entity"]);
        let lower_target = source.create_node(&["Entity"]);
        let higher_target = source.create_node(&["Entity"]);
        let higher_edge = source.create_edge(from, higher_target, "RELATES_TO");
        let lower_edge = source.create_edge(from, lower_target, "RELATES_TO");
        source.set_edge_property(higher_edge, "rank", Value::Int64(2));
        source.set_edge_property(lower_edge, "rank", Value::Int64(1));

        let section =
            CompactStoreSection::new(Arc::new(from_graph_store_preserving_ids(&source).unwrap()));
        for version in [
            FORMAT_VERSION_V1,
            FORMAT_VERSION_V2,
            FORMAT_VERSION_V3,
            FORMAT_VERSION_V4,
            FORMAT_VERSION,
        ] {
            let bytes = section.serialize_with_version(version).expect("serialize");
            let mut restored = CompactStoreSection::empty();
            restored.deserialize(&bytes).expect("deserialize");
            let compact = restored.store().expect("restored store");

            let higher = compact.get_edge(higher_edge).unwrap();
            assert_eq!((higher.src, higher.dst), (from, higher_target));
            assert_eq!(
                higher.properties.get(&PropertyKey::new("rank")),
                Some(&Value::Int64(2)),
            );
            let lower = compact.get_edge(lower_edge).unwrap();
            assert_eq!((lower.src, lower.dst), (from, lower_target));
            assert_eq!(
                lower.properties.get(&PropertyKey::new("rank")),
                Some(&Value::Int64(1)),
            );
        }
    }

    #[test]
    fn generation_scale_reopen_preserves_edge_endpoints_and_traversal() {
        const SOURCE_COUNT: usize = 7_500;
        const TARGET_COUNT: usize = 22_500;

        let source = LpgStore::new().unwrap();
        let sources: Vec<_> = (0..SOURCE_COUNT)
            .map(|_| source.create_node(&["Source"]))
            .collect();
        let targets: Vec<_> = (0..TARGET_COUNT)
            .map(|_| source.create_node(&["Target"]))
            .collect();
        let mut expected_by_source = Vec::with_capacity(SOURCE_COUNT);
        for (index, &from) in sources.iter().enumerate() {
            let lower_target = targets[index * 2];
            let higher_target = targets[index * 2 + 1];
            let higher_edge = source.create_edge(from, higher_target, "RELATES_TO");
            let lower_edge = source.create_edge(from, lower_target, "RELATES_TO");
            expected_by_source.push([(higher_target, higher_edge), (lower_target, lower_edge)]);
        }
        assert_eq!(
            source.node_count() + source.edge_count(),
            45_000,
            "fixture must retain the audited generation scale",
        );

        let section =
            CompactStoreSection::new(Arc::new(from_graph_store_preserving_ids(&source).unwrap()));
        let bytes = section.serialize().expect("serialize");
        let mut restored = CompactStoreSection::empty();
        restored.deserialize(&bytes).expect("deserialize");
        let compact = restored.store().expect("restored store");

        for (&from, expected) in sources.iter().zip(&expected_by_source) {
            for &(target, edge_id) in expected {
                let edge = compact.get_edge(edge_id).expect("preserved edge resolves");
                assert_eq!((edge.src, edge.dst), (from, target));
            }
        }
        for (&from, expected) in sources.iter().zip(expected_by_source) {
            let actual: FxHashSet<_> = compact
                .edges_from(from, crate::graph::Direction::Outgoing)
                .into_iter()
                .collect();
            assert_eq!(actual, FxHashSet::from_iter(expected));
        }
    }

    /// Byte identity has to hold at every format version the writer can
    /// still emit, not just the current one.
    #[test]
    fn compact_sink_matches_vec_for_legacy_versions() {
        let store = LpgStore::new().unwrap();
        for i in 0..64i64 {
            let id = store.create_node(&["Person"]);
            store.set_node_property(id, "age", Value::Int64(i));
        }
        let section =
            CompactStoreSection::new(Arc::new(from_graph_store_preserving_ids(&store).unwrap()));

        for version in [
            FORMAT_VERSION_V1,
            FORMAT_VERSION_V2,
            FORMAT_VERSION_V3,
            FORMAT_VERSION_V4,
            FORMAT_VERSION,
        ] {
            let via_vec = section.serialize_with_version(version).unwrap();
            let mut via_sink = Vec::new();
            section
                .serialize_into_with_version(&mut via_sink, version)
                .unwrap();
            assert!(
                via_sink == via_vec,
                "version {version}: sink and Vec bytes must be identical"
            );
        }
    }

    /// A pre-v4 dictionary stores every entry as a raw string, including
    /// one that collides with the v4 marker prefix. Loading such a
    /// section must hand back exactly the original string — never retype
    /// it as a `Value::Bytes` payload — and equality lookups (which
    /// encode the query through the marker-aware path) must keep finding
    /// the row.
    #[test]
    fn legacy_marker_colliding_string_survives_reopen() {
        // Same byte length as the raw colliding string, so it can be
        // spliced over in the serialized section. The builder would
        // escape the colliding string at build time (that is v4
        // behavior), so a faithful legacy fixture has to be forged in
        // the serialized bytes, the way a pre-marker writer laid it out.
        let placeholder = "Xgfo1:b:00";
        let tricky = "\u{0}gfo1:b:00";
        assert_eq!(placeholder.len(), tricky.len());

        let store = LpgStore::new().unwrap();
        let id = store.create_node(&["Item"]);
        store.set_node_property(id, "s", Value::from(placeholder));

        let compact = from_graph_store_preserving_ids(&store).unwrap();
        let section = CompactStoreSection::new(Arc::new(compact));
        let mut bytes = section
            .serialize_with_version(FORMAT_VERSION_V3)
            .expect("serialize legacy section");

        // The string is stored in several places — the dictionary entry and
        // the zone-map min/max copies — and a legacy writer would have the
        // raw form in all of them.
        let mut search_from = 0;
        let mut replaced = 0;
        while let Some(found) = bytes[search_from..]
            .windows(placeholder.len())
            .position(|w| w == placeholder.as_bytes())
        {
            let at = search_from + found;
            bytes[at..at + tricky.len()].copy_from_slice(tricky.as_bytes());
            search_from = at + tricky.len();
            replaced += 1;
        }
        assert!(
            replaced >= 1,
            "placeholder entry not found in serialized section"
        );
        let crc_pos = bytes.len() - 4;
        let crc = crc32fast::hash(&bytes[..crc_pos]);
        bytes[crc_pos..].copy_from_slice(&crc.to_le_bytes());

        let mut section2 = CompactStoreSection::empty();
        section2.deserialize(&bytes).expect("deserialize legacy");
        let restored = section2.store().unwrap();

        assert_eq!(
            restored.get_node_property(id, &PropertyKey::new("s")),
            Some(Value::from(tricky)),
            "legacy raw string was retyped instead of escaped"
        );
        assert_eq!(
            restored
                .find_nodes_by_property("s", &Value::from(tricky))
                .len(),
            1,
            "equality lookup lost the legacy raw string"
        );
    }

    #[test]
    fn test_round_trip_empty() {
        let store = LpgStore::new().unwrap();
        let compact = from_graph_store_preserving_ids(&store).unwrap();
        let section = CompactStoreSection::new(Arc::new(compact));

        let bytes = section.serialize().unwrap();
        let mut section2 = CompactStoreSection::empty();
        section2.deserialize(&bytes).unwrap();

        let restored = section2.store().unwrap();
        assert_eq!(restored.node_count(), 0);
        assert_eq!(restored.edge_count(), 0);
    }

    #[test]
    fn test_round_trip_nodes_and_edges() {
        let store = LpgStore::new().unwrap();
        let alix = store.create_node(&["Person"]);
        store.set_node_property(alix, "name", Value::from("Alix"));
        store.set_node_property(alix, "age", Value::Int64(30));

        let gus = store.create_node(&["Person"]);
        store.set_node_property(gus, "name", Value::from("Gus"));
        store.set_node_property(gus, "age", Value::Int64(25));

        let amsterdam = store.create_node(&["City"]);
        store.set_node_property(amsterdam, "name", Value::from("Amsterdam"));

        store.create_edge(alix, amsterdam, "LIVES_IN");
        store.create_edge(gus, amsterdam, "LIVES_IN");

        let compact = from_graph_store_preserving_ids(&store).unwrap();
        assert!(compact.preserves_ids());

        let section = CompactStoreSection::new(Arc::new(compact));
        let bytes = section.serialize().unwrap();

        let mut section2 = CompactStoreSection::empty();
        section2.deserialize(&bytes).unwrap();
        let restored = section2.store().unwrap();

        assert!(restored.preserves_ids());
        assert_eq!(restored.node_count(), 3);
        assert_eq!(restored.edge_count(), 2);

        // Verify original IDs survive.
        let alix_node = restored.get_node(alix).expect("Alix by original ID");
        assert_eq!(
            alix_node.properties.get(&PropertyKey::new("name")),
            Some(&Value::String(arcstr::ArcStr::from("Alix")))
        );
        assert_eq!(
            alix_node.properties.get(&PropertyKey::new("age")),
            Some(&Value::Int64(30))
        );

        // Verify edge traversal.
        let neighbors = restored.neighbors(alix, crate::graph::Direction::Outgoing);
        assert_eq!(neighbors.len(), 1);
        assert_eq!(neighbors[0], amsterdam);
    }

    #[test]
    fn test_round_trip_without_id_preservation() {
        use crate::graph::compact::from_graph_store;

        let lpg = LpgStore::new().unwrap();
        let a = lpg.create_node(&["Node"]);
        lpg.set_node_property(a, "val", Value::Int64(42));
        let b = lpg.create_node(&["Node"]);
        lpg.set_node_property(b, "val", Value::Int64(99));
        lpg.create_edge(a, b, "LINK");

        let compact = from_graph_store(&lpg).unwrap();
        assert!(!compact.preserves_ids());

        let section = CompactStoreSection::new(Arc::new(compact));
        let bytes = section.serialize().unwrap();

        let mut section2 = CompactStoreSection::empty();
        section2.deserialize(&bytes).unwrap();
        let restored = section2.store().unwrap();

        assert!(!restored.preserves_ids());
        assert_eq!(restored.node_count(), 2);
        assert_eq!(restored.edge_count(), 1);
    }

    #[test]
    fn test_crc_integrity() {
        let store = LpgStore::new().unwrap();
        store.create_node(&["Test"]);
        let compact = from_graph_store_preserving_ids(&store).unwrap();

        let section = CompactStoreSection::new(Arc::new(compact));
        let mut bytes = section.serialize().unwrap();

        // Corrupt a byte in the middle.
        if bytes.len() > 10 {
            bytes[10] ^= 0xFF;
        }

        let mut section2 = CompactStoreSection::empty();
        assert!(section2.deserialize(&bytes).is_err());
    }

    #[test]
    fn test_section_type_and_version() {
        let section = CompactStoreSection::empty();
        assert_eq!(section.section_type(), SectionType::CompactStore);
        assert_eq!(section.version(), FORMAT_VERSION);
        assert!(!section.is_dirty());
        assert_eq!(section.memory_usage(), 0);
    }

    #[test]
    fn test_dirty_tracking() {
        let section = CompactStoreSection::empty();
        assert!(!section.is_dirty());
        section.mark_dirty();
        assert!(section.is_dirty());
        section.mark_clean();
        assert!(!section.is_dirty());
    }

    /// Phase 2b: confirm the v1 (flat-column) on-disk format still
    /// round-trips through the v2-aware deserializer, exercising the
    /// compat path users on 0.5.41 and earlier rely on for one release.
    #[test]
    fn nelson_v1_section_reads_through_v2_aware_deserializer() {
        let store = LpgStore::new().unwrap();
        let alix = store.create_node(&["Person"]);
        store.set_node_property(alix, "name", Value::from("Alix"));
        store.set_node_property(alix, "age", Value::Int64(30));

        let gus = store.create_node(&["Person"]);
        store.set_node_property(gus, "name", Value::from("Gus"));
        store.set_node_property(gus, "age", Value::Int64(25));

        store.create_edge(alix, gus, "KNOWS");

        let compact = from_graph_store_preserving_ids(&store).unwrap();
        let section = CompactStoreSection::new(Arc::new(compact));

        // Force v1 layout (flat columns, version byte = 1).
        let v1_bytes = section.serialize_with_version(FORMAT_VERSION_V1).unwrap();
        // First byte after MAGIC must be the v1 marker.
        assert_eq!(
            v1_bytes[4], FORMAT_VERSION_V1,
            "expected v1 marker in version byte"
        );

        // The v2-aware deserializer must handle both versions.
        let mut section2 = CompactStoreSection::empty();
        section2.deserialize(&v1_bytes).unwrap();
        let restored = section2.store().unwrap();

        assert_eq!(restored.node_count(), 2);
        assert_eq!(restored.edge_count(), 1);
        assert_eq!(
            restored.get_node_property(alix, &PropertyKey::new("name")),
            Some(Value::String(arcstr::ArcStr::from("Alix")))
        );
        assert_eq!(
            restored.get_node_property(alix, &PropertyKey::new("age")),
            Some(Value::Int64(30))
        );
    }

    // ── Phase 2c: per-block zone maps ────────────────────────────────

    /// The builder must populate per-block zone maps for every column,
    /// one ZoneMap per block. `1024` rows per block (DEFAULT_BLOCK_ROWS).
    #[test]
    fn alix_builder_populates_per_block_zone_maps() {
        let store = LpgStore::new().unwrap();
        // 3000 nodes → 3 blocks (1024 + 1024 + 952).
        for i in 0i64..3000 {
            let n = store.create_node(&["Person"]);
            store.set_node_property(n, "age", Value::Int64(i));
        }
        let compact = from_graph_store_preserving_ids(&store).unwrap();
        let table = &compact.node_tables_by_id[0];
        let block_zms = table
            .block_zone_maps_for(&PropertyKey::new("age"))
            .expect("per-block stats present");
        assert_eq!(block_zms.len(), 3, "3000 rows should produce 3 blocks");
        assert_eq!(block_zms[0].row_count, 1024);
        assert_eq!(block_zms[1].row_count, 1024);
        assert_eq!(block_zms[2].row_count, 952);
        assert_eq!(block_zms[0].min, Some(Value::Int64(0)));
        assert_eq!(block_zms[0].max, Some(Value::Int64(1023)));
        assert_eq!(block_zms[1].min, Some(Value::Int64(1024)));
        assert_eq!(block_zms[1].max, Some(Value::Int64(2047)));
        assert_eq!(block_zms[2].min, Some(Value::Int64(2048)));
        assert_eq!(block_zms[2].max, Some(Value::Int64(2999)));
    }

    /// v3 round-trip preserves per-block zone maps verbatim.
    #[test]
    fn gus_v3_round_trip_preserves_block_zone_maps() {
        let store = LpgStore::new().unwrap();
        for i in 0i64..2500 {
            let n = store.create_node(&["Item"]);
            store.set_node_property(n, "score", Value::Int64(i));
        }
        let compact = from_graph_store_preserving_ids(&store).unwrap();
        let original = &compact.node_tables_by_id[0];
        let original_zms = original
            .block_zone_maps_for(&PropertyKey::new("score"))
            .expect("original block stats")
            .to_vec();

        let section = CompactStoreSection::new(Arc::new(compact));
        let bytes = section.serialize().unwrap();
        let mut section2 = CompactStoreSection::empty();
        section2.deserialize(&bytes).unwrap();
        let restored = section2.store().unwrap();
        let restored_table = &restored.node_tables_by_id[0];
        let restored_zms = restored_table
            .block_zone_maps_for(&PropertyKey::new("score"))
            .expect("restored block stats");

        assert_eq!(restored_zms.len(), original_zms.len());
        for (i, (orig, rest)) in original_zms.iter().zip(restored_zms.iter()).enumerate() {
            assert_eq!(orig.row_count, rest.row_count, "row_count mismatch at {i}");
            assert_eq!(
                orig.null_count, rest.null_count,
                "null_count mismatch at {i}"
            );
            assert_eq!(orig.min, rest.min, "min mismatch at {i}");
            assert_eq!(orig.max, rest.max, "max mismatch at {i}");
        }
    }

    /// v2 sections (Phase 2b) carry no per-block zone maps; the v3 reader
    /// must accept them and leave `block_zone_maps_for` returning `None`.
    #[test]
    fn vincent_v2_section_round_trip_leaves_block_zone_maps_empty() {
        let store = LpgStore::new().unwrap();
        for i in 0i64..1500 {
            let n = store.create_node(&["Item"]);
            store.set_node_property(n, "score", Value::Int64(i));
        }
        let compact = from_graph_store_preserving_ids(&store).unwrap();
        let section = CompactStoreSection::new(Arc::new(compact));
        let v2_bytes = section.serialize_with_version(FORMAT_VERSION_V2).unwrap();
        assert_eq!(v2_bytes[4], FORMAT_VERSION_V2);

        let mut section2 = CompactStoreSection::empty();
        section2.deserialize(&v2_bytes).unwrap();
        let restored = section2.store().unwrap();
        let table = &restored.node_tables_by_id[0];
        assert!(
            table
                .block_zone_maps_for(&PropertyKey::new("score"))
                .is_none(),
            "v2 stream must not populate block_zone_maps"
        );
        // But the column data still survives.
        assert_eq!(table.len(), 1500);
    }

    /// v1 sections likewise carry no per-block stats.
    #[test]
    fn jules_v1_section_round_trip_leaves_block_zone_maps_empty() {
        let store = LpgStore::new().unwrap();
        for i in 0i64..1500 {
            let n = store.create_node(&["Item"]);
            store.set_node_property(n, "score", Value::Int64(i));
        }
        let compact = from_graph_store_preserving_ids(&store).unwrap();
        let section = CompactStoreSection::new(Arc::new(compact));
        let v1_bytes = section.serialize_with_version(FORMAT_VERSION_V1).unwrap();
        assert_eq!(v1_bytes[4], FORMAT_VERSION_V1);

        let mut section2 = CompactStoreSection::empty();
        section2.deserialize(&v1_bytes).unwrap();
        let restored = section2.store().unwrap();
        let table = &restored.node_tables_by_id[0];
        assert!(
            table
                .block_zone_maps_for(&PropertyKey::new("score"))
                .is_none(),
            "v1 stream must not populate block_zone_maps"
        );
        assert_eq!(table.len(), 1500);
    }

    /// String columns also get per-block min/max.
    #[test]
    fn mia_block_zone_maps_for_string_column() {
        let store = LpgStore::new().unwrap();
        // Use enough nodes to force >= 2 blocks.
        for i in 0u32..1100 {
            let n = store.create_node(&["Tag"]);
            store.set_node_property(n, "name", Value::from(format!("tag_{i:04}")));
        }
        let compact = from_graph_store_preserving_ids(&store).unwrap();
        let table = &compact.node_tables_by_id[0];
        let block_zms = table
            .block_zone_maps_for(&PropertyKey::new("name"))
            .expect("string column block stats");
        assert_eq!(block_zms.len(), 2);
        assert_eq!(
            block_zms[0].min,
            Some(Value::String(arcstr::ArcStr::from("tag_0000")))
        );
        assert_eq!(
            block_zms[0].max,
            Some(Value::String(arcstr::ArcStr::from("tag_1023")))
        );
        assert_eq!(
            block_zms[1].min,
            Some(Value::String(arcstr::ArcStr::from("tag_1024")))
        );
        assert_eq!(
            block_zms[1].max,
            Some(Value::String(arcstr::ArcStr::from("tag_1099")))
        );
    }

    /// Phase 2b: an unsupported version byte must produce a clean error,
    /// not panic or silently misread the section.
    #[test]
    fn rita_unknown_version_returns_clear_error() {
        let store = LpgStore::new().unwrap();
        let _ = store.create_node(&["Item"]);
        let compact = from_graph_store_preserving_ids(&store).unwrap();
        let section = CompactStoreSection::new(Arc::new(compact));
        let mut bytes = section.serialize().unwrap();
        // Strip CRC, flip version byte to a future v9, recompute CRC.
        let crc_pos = bytes.len() - 4;
        bytes[4] = 9;
        let crc = crc32fast::hash(&bytes[..crc_pos]);
        bytes[crc_pos..].copy_from_slice(&crc.to_le_bytes());

        let mut section2 = CompactStoreSection::empty();
        let err = section2
            .deserialize(&bytes)
            .expect_err("expected version error");
        let msg = err.to_string();
        assert!(
            msg.contains("unsupported CompactStore section version"),
            "unexpected error message: {msg}"
        );
    }

    #[test]
    fn test_round_trip_bool_column() {
        let store = LpgStore::new().unwrap();
        let a = store.create_node(&["Item"]);
        store.set_node_property(a, "active", Value::Bool(true));
        let b = store.create_node(&["Item"]);
        store.set_node_property(b, "active", Value::Bool(false));

        let compact = from_graph_store_preserving_ids(&store).unwrap();
        let section = CompactStoreSection::new(Arc::new(compact));
        let bytes = section.serialize().unwrap();

        let mut section2 = CompactStoreSection::empty();
        section2.deserialize(&bytes).unwrap();
        let restored = section2.store().unwrap();

        assert_eq!(
            restored.get_node_property(a, &PropertyKey::new("active")),
            Some(Value::Bool(true))
        );
        assert_eq!(
            restored.get_node_property(b, &PropertyKey::new("active")),
            Some(Value::Bool(false))
        );
    }

    #[test]
    fn test_round_trip_edge_properties() {
        let store = LpgStore::new().unwrap();
        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        let e = store.create_edge(a, b, "LINK");
        store.set_edge_property(e, "weight", Value::Int64(5));

        let compact = from_graph_store_preserving_ids(&store).unwrap();
        let section = CompactStoreSection::new(Arc::new(compact));
        let bytes = section.serialize().unwrap();

        let mut section2 = CompactStoreSection::empty();
        section2.deserialize(&bytes).unwrap();
        let restored = section2.store().unwrap();

        // Find the edge via traversal.
        let edges = restored.edges_from(a, crate::graph::Direction::Outgoing);
        assert_eq!(edges.len(), 1);
        let edge = restored.get_edge(edges[0].1).unwrap();
        assert_eq!(
            edge.properties.get(&PropertyKey::new("weight")),
            Some(&Value::Int64(5))
        );
    }
}
