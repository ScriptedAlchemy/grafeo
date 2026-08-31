//! [`Section`](grafeo_common::storage::section::Section) implementation for [`CompactStore`].
//!
//! Serializes/deserializes a CompactStore to/from the `.grafeo` container
//! format with versioned headers and CRC32 integrity.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use bytes::Bytes;
use grafeo_common::storage::section::{Section, SectionType};
use grafeo_common::types::{EdgeId, NodeId, PropertyKey};
use grafeo_common::utils::hash::FxHashMap;
use parking_lot::RwLock;

use super::CompactStore;
use super::column::ColumnCodec;
use super::csr::CsrAdjacency;
use super::node_table::NodeTable;
use super::rel_table::RelTable;
use super::schema::{ColumnDef, ColumnType, EdgeSchema, TableSchema};
use super::zone_map::ZoneMap;
use crate::statistics::{EdgeTypeStatistics, LabelStatistics, Statistics};

/// Magic bytes identifying a CompactStore section.
const MAGIC: [u8; 4] = *b"GCST";

/// Current section format version. v4 versions the dictionary-entry
/// mapping (`dict_value`): in a v4 section, a dictionary entry beginning
/// with the marker prefix is a typed payload — a `Value::Bytes` hex body
/// or an escaped string — never a raw user string. The column byte
/// layout is identical to v3.
const FORMAT_VERSION: u8 = 4;

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
}

impl CompactStoreSection {
    /// Creates a new section wrapping an existing store.
    #[must_use]
    pub fn new(store: Arc<CompactStore>) -> Self {
        Self {
            store: RwLock::new(Some(store)),
            dirty: AtomicBool::new(false),
        }
    }

    /// Creates an empty section (for deserialization).
    #[must_use]
    pub fn empty() -> Self {
        Self {
            store: RwLock::new(None),
            dirty: AtomicBool::new(false),
        }
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
    /// Same error semantics as [`Section::deserialize`].
    pub fn deserialize_from_bytes(
        &mut self,
        data: bytes::Bytes,
    ) -> grafeo_common::utils::error::Result<()> {
        let store = deserialize_compact_store(&data).map_err(|e| {
            grafeo_common::utils::error::Error::Internal(format!(
                "CompactStore deserialization failed: {e}"
            ))
        })?;
        *self.store.write() = Some(Arc::new(store));
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
            .map_or(0, |store| store.memory_bytes());
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

        // Starts small and grows to the largest table; `drain_chunk`
        // clears without releasing capacity, so later tables reuse it.
        let mut buf: Vec<u8> = Vec::with_capacity(CHUNK_TARGET_BYTES);
        let mut crc = crc32fast::Hasher::new();

        // Header.
        buf.extend_from_slice(&MAGIC);
        buf.push(version);
        let flags: u8 = u8::from(store.preserves_ids());
        buf.push(flags);

        // Node tables.
        write_len(&mut buf, store.node_tables_by_id.len());
        for nt in &store.node_tables_by_id {
            write_str(&mut buf, nt.label());
            write_len(&mut buf, nt.len());
            let columns = nt.columns();
            let zone_maps = nt.zone_maps();
            write_len(&mut buf, columns.len());
            for (key, codec) in columns {
                write_str(&mut buf, key.as_str());
                // Zone map for this column.
                if let Some(zm) = zone_maps.get(key) {
                    buf.push(1);
                    write_zone_map(&mut buf, zm);
                } else {
                    buf.push(0);
                }
                write_codec(
                    codec,
                    &mut buf,
                    version,
                    nt.block_zone_maps().get(key).map(Vec::as_slice),
                );
                // Column granularity: a single wide column is the
                // smallest unit this encoder can emit without rewriting
                // the sibling-module writers.
                drain_chunk(&mut buf, sink, &mut crc, false)?;
            }
        }

        // Relationship tables.
        write_len(&mut buf, store.rel_tables_by_id.len());
        for rt in &store.rel_tables_by_id {
            write_str(&mut buf, rt.edge_type().as_str());
            write_u16(&mut buf, rt.src_table_id());
            write_u16(&mut buf, rt.dst_table_id());
            rt.fwd().write_to(&mut buf);
            if let Some(bwd) = rt.bwd() {
                buf.push(1);
                bwd.write_to(&mut buf);
            } else {
                buf.push(0);
            }
            drain_chunk(&mut buf, sink, &mut crc, false)?;
            let properties = rt.properties();
            write_len(&mut buf, properties.len());
            for (key, codec) in properties {
                write_str(&mut buf, key.as_str());
                // Edge property columns don't track per-block zone maps
                // yet; v3 will compute them inline during write.
                write_codec(codec, &mut buf, version, None);
                drain_chunk(&mut buf, sink, &mut crc, false)?;
            }
        }

        self.write_id_maps(&mut buf, sink, &mut crc, store)?;

        // Flush whatever is left, then the CRC over everything written.
        drain_chunk(&mut buf, sink, &mut crc, true)?;
        sink.write_all(&crc.finalize().to_le_bytes())?;
        Ok(())
    }

    /// Writes the ID maps (when the store preserves ids), draining to
    /// the sink as the scratch buffer fills.
    fn write_id_maps(
        &self,
        buf: &mut Vec<u8>,
        sink: &mut dyn std::io::Write,
        crc: &mut crc32fast::Hasher,
        store: &CompactStore,
    ) -> grafeo_common::utils::error::Result<()> {
        if !store.preserves_ids() {
            return Ok(());
        }
        if let Some(ref node_map) = store.node_id_map {
            write_len(buf, node_map.len());
            for (&nid, &(tid, off)) in node_map {
                write_u64(buf, nid.as_u64());
                write_u16(buf, tid);
                write_u64(buf, off);
                drain_chunk(buf, sink, crc, false)?;
            }
        }
        if let Some(ref edge_map) = store.edge_id_map {
            write_len(buf, edge_map.len());
            for (&eid, &(rtid, pos)) in edge_map {
                write_u64(buf, eid.as_u64());
                write_u16(buf, rtid);
                write_u64(buf, pos);
                drain_chunk(buf, sink, crc, false)?;
            }
        }
        Ok(())
    }
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
        self.deserialize_from_bytes(owned)
    }

    fn is_dirty(&self) -> bool {
        self.dirty.load(Ordering::Acquire)
    }

    fn mark_clean(&self) {
        self.dirty.store(false, Ordering::Release);
    }

    fn memory_usage(&self) -> usize {
        self.store.read().as_ref().map_or(0, |s| s.memory_bytes())
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
        FORMAT_VERSION_V3 | FORMAT_VERSION => ColumnCodec::read_from_v3(data, pos)
            .map(|(c, stats)| (c, Some(stats)))
            .map_err(|e| e.to_string())?,
        _ => return Err(format!("unsupported CompactStore version {version}")),
    };
    if version < DICT_MARKERS_SINCE_VERSION {
        codec.escape_legacy_dict_markers();
    }
    Ok((codec, stats))
}

fn deserialize_compact_store(data_bytes: &bytes::Bytes) -> Result<CompactStore, String> {
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
        FORMAT_VERSION | FORMAT_VERSION_V3 | FORMAT_VERSION_V2 | FORMAT_VERSION_V1
    ) {
        return Err(format!(
            "unsupported CompactStore section version {version} (supported: {FORMAT_VERSION_V1}, {FORMAT_VERSION_V2}, {FORMAT_VERSION_V3}, {FORMAT_VERSION})"
        ));
    }
    let flags = data[pos];
    pos += 1;
    let preserves_ids = flags & 0x01 != 0;

    // Node tables.
    let num_node_tables = read_u32(data, &mut pos)? as usize;
    let mut node_tables = Vec::with_capacity(num_node_tables);
    let mut label_to_table_id: FxHashMap<arcstr::ArcStr, u16> = FxHashMap::default();
    let mut table_id_to_label: Vec<arcstr::ArcStr> = Vec::with_capacity(num_node_tables);

    for table_idx in 0..num_node_tables {
        let table_id = u16::try_from(table_idx).unwrap_or(0);
        let label = read_string(data, &mut pos)?;
        let label = arcstr::ArcStr::from(label.as_str());
        let row_count = read_u32(data, &mut pos)? as usize;
        let num_cols = read_u32(data, &mut pos)? as usize;

        let mut columns: FxHashMap<PropertyKey, ColumnCodec> = FxHashMap::default();
        let mut zone_maps: FxHashMap<PropertyKey, ZoneMap> = FxHashMap::default();
        let mut block_zone_maps: FxHashMap<PropertyKey, Vec<ZoneMap>> = FxHashMap::default();
        let mut col_defs = Vec::with_capacity(num_cols);

        for _ in 0..num_cols {
            let key_str = read_string(data, &mut pos)?;
            let key = PropertyKey::new(&key_str);

            let has_zm = *data.get(pos).ok_or("truncated zone map flag")?;
            pos += 1;
            if has_zm == 1 {
                let zm = read_zone_map(data, &mut pos)?;
                zone_maps.insert(key.clone(), zm);
            }

            let (codec, maybe_block_stats) =
                read_codec(data_bytes, &mut pos, version).map_err(|e| format!("codec: {e}"))?;
            if let Some(stats) = maybe_block_stats {
                block_zone_maps.insert(key.clone(), stats);
            }
            let col_type = infer_column_type_from_codec(&codec);
            col_defs.push(ColumnDef::new(&key_str, col_type));
            columns.insert(key, codec);
        }

        let schema = TableSchema::new(label.as_str(), table_id, col_defs);
        let table = NodeTable::from_columns_with_block_stats(
            schema,
            columns,
            zone_maps,
            block_zone_maps,
            row_count,
        );
        node_tables.push(table);
        label_to_table_id.insert(label.clone(), table_id);
        table_id_to_label.push(label);
    }

    // Relationship tables.
    let num_rel_tables = read_u32(data, &mut pos)? as usize;
    let mut rel_tables = Vec::with_capacity(num_rel_tables);
    let mut edge_type_to_rel_id: FxHashMap<arcstr::ArcStr, Vec<u16>> = FxHashMap::default();
    let mut rel_table_id_to_type: Vec<arcstr::ArcStr> = Vec::with_capacity(num_rel_tables);

    for rel_idx in 0..num_rel_tables {
        let rel_table_id = u16::try_from(rel_idx).unwrap_or(0);
        let edge_type = read_string(data, &mut pos)?;
        let edge_type = arcstr::ArcStr::from(edge_type.as_str());
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
        let mut properties: FxHashMap<PropertyKey, ColumnCodec> = FxHashMap::default();
        let mut prop_defs = Vec::with_capacity(num_props);
        for _ in 0..num_props {
            let key_str = read_string(data, &mut pos)?;
            let key = PropertyKey::new(&key_str);
            let (codec, _block_stats) = read_codec(data_bytes, &mut pos, version)
                .map_err(|e| format!("edge codec: {e}"))?;
            let col_type = infer_column_type_from_codec(&codec);
            prop_defs.push(ColumnDef::new(&key_str, col_type));
            properties.insert(key, codec);
        }

        let src_label = table_id_to_label
            .get(src_tid as usize)
            .cloned()
            .unwrap_or_default();
        let dst_label = table_id_to_label
            .get(dst_tid as usize)
            .cloned()
            .unwrap_or_default();

        let schema = EdgeSchema::new(
            edge_type.as_str(),
            rel_table_id,
            src_label.as_str(),
            dst_label.as_str(),
            prop_defs,
        );

        let table = RelTable::new(schema, fwd, bwd, properties, src_tid, dst_tid);
        edge_type_to_rel_id
            .entry(edge_type.clone())
            .or_default()
            .push(rel_table_id);
        rel_table_id_to_type.push(edge_type);
        rel_tables.push(table);
    }

    // Compute statistics.
    let mut stats = Statistics::new();
    let mut total_nodes = 0u64;
    let mut total_edges = 0u64;
    for (idx, nt) in node_tables.iter().enumerate() {
        let c = nt.len() as u64;
        total_nodes += c;
        stats.update_label(table_id_to_label[idx].as_str(), LabelStatistics::new(c));
    }
    let mut edge_counts: FxHashMap<&str, u64> = FxHashMap::default();
    for (idx, rt) in rel_tables.iter().enumerate() {
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
        node_tables,
        label_to_table_id,
        rel_tables,
        edge_type_to_rel_id,
        table_id_to_label,
        rel_table_id_to_type,
        stats,
    );

    // ID maps.
    if preserves_ids {
        let node_map_len = read_u32(data, &mut pos)? as usize;
        let mut node_id_map = FxHashMap::with_capacity_and_hasher(node_map_len, Default::default());
        let num_tables = store.node_tables_by_id.len();
        let mut node_offset_to_id: Vec<Vec<NodeId>> = vec![Vec::new(); num_tables];
        for _ in 0..node_map_len {
            let nid = NodeId::new(read_u64(data, &mut pos)?);
            let tid = read_u16(data, &mut pos)?;
            let off = read_u64(data, &mut pos)?;
            node_id_map.insert(nid, (tid, off));
            let off_idx = usize::try_from(off).unwrap_or(usize::MAX);
            if let Some(rev) = node_offset_to_id.get_mut(tid as usize) {
                while rev.len() <= off_idx {
                    rev.push(NodeId::INVALID);
                }
                rev[off_idx] = nid;
            }
        }

        let edge_map_len = read_u32(data, &mut pos)? as usize;
        let mut edge_id_map = FxHashMap::with_capacity_and_hasher(edge_map_len, Default::default());
        let num_rel = store.rel_tables_by_id.len();
        let mut edge_offset_to_id: Vec<Vec<EdgeId>> = vec![Vec::new(); num_rel];
        for _ in 0..edge_map_len {
            let eid = EdgeId::new(read_u64(data, &mut pos)?);
            let rtid = read_u16(data, &mut pos)?;
            let csr_pos = read_u64(data, &mut pos)?;
            edge_id_map.insert(eid, (rtid, csr_pos));
            let pos_idx = usize::try_from(csr_pos).unwrap_or(usize::MAX);
            if let Some(rev) = edge_offset_to_id.get_mut(rtid as usize) {
                while rev.len() <= pos_idx {
                    rev.push(EdgeId::INVALID);
                }
                rev[pos_idx] = eid;
            }
        }

        store.set_id_maps(
            node_id_map,
            edge_id_map,
            node_offset_to_id,
            edge_offset_to_id,
        );
    }

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

fn write_str(buf: &mut Vec<u8>, s: &str) {
    let bytes = s.as_bytes();
    let slen = u16::try_from(bytes.len()).expect("string exceeds u16::MAX in compact section");
    write_u16(buf, slen);
    buf.extend_from_slice(bytes);
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
        Some(grafeo_common::types::Value::String(s)) => {
            buf.push(3);
            write_str(buf, s.as_str());
        }
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
