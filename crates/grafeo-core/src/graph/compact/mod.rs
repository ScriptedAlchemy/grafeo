//! CompactStore: a read-only columnar store for memory-constrained environments.
//!
//! Implements [`GraphStore`](crate::graph::traits::GraphStore) using per-label
//! columnar tables and double-indexed CSR adjacency. Designed for static
//! snapshot data in WASM, edge workers, and embedded devices.
//! Fully behind `#[cfg(feature = "compact-store")]`.

/// Builder API for constructing a [`CompactStore`] from raw data.
pub mod builder;
/// Columnar codecs for node and edge properties.
pub mod column;
/// Compressed Sparse Row (CSR) adjacency representation.
pub mod csr;
/// Container section serialization for the layered overlay deletion log.
#[cfg(feature = "lpg")]
pub mod deletions_section;
pub(crate) mod dict_value;
mod graph_store_impl;
mod heap;
/// Node/edge ID encoding and decoding helpers.
pub mod id;
mod id_map;
/// Two-layer store: columnar base + mutable LPG overlay.
#[cfg(feature = "lpg")]
pub mod layered;
/// Per-label node tables with columnar property storage.
pub mod node_table;
/// Per-type relationship tables backed by forward/backward CSR.
pub mod rel_table;
/// Schema definitions for node tables and edge schemas.
pub mod schema;
/// Container section serialization for CompactStore.
pub mod section;
#[cfg(test)]
mod tests;
mod value_order;
/// Zone maps for skip-pruning predicate evaluation.
pub mod zone_map;

pub use builder::{
    CompactStoreBuilder, IncrementalCompactStoreBuilder, from_graph_store,
    from_graph_store_preserving_ids,
};

use std::sync::Arc;

use arcstr::ArcStr;
use grafeo_common::types::{EdgeId, NodeId, PropertyKey};
use grafeo_common::utils::hash::FxHashMap;

use grafeo_common::memory::heap::{arc_slice_bytes, arcstr_bytes, vec_bytes};

use self::heap::{key_bytes, statistics_bytes};
use self::id_map::IdMap;
use self::node_table::NodeTable;
use self::rel_table::RelTable;
use self::value_order::{RowOrder, build_row_order};
use crate::codec::SectionSpan;
use crate::codec::pages::SectionPages;
use crate::graph::Direction;
use crate::statistics::Statistics;

/// A read-only columnar graph store.
///
/// Node data is stored in per-label [`NodeTable`]s and edge data in per-type
/// [`RelTable`]s. The store is immutable after construction: use
/// [`CompactStoreBuilder`] to populate it from raw data.
pub struct CompactStore {
    /// Node tables indexed by table_id for O(1) lookup from NodeId.
    node_tables_by_id: Vec<NodeTable>,
    /// table_id lookup from label string (for nodes_by_label).
    label_to_table_id: FxHashMap<ArcStr, u16>,
    /// Relationship tables indexed by rel_table_id for O(1) lookup from EdgeId.
    rel_tables_by_id: Vec<RelTable>,
    /// rel_table_id lookup from edge type string (one edge type may span
    /// multiple src/dst label combinations, so the value is a Vec).
    edge_type_to_rel_id: FxHashMap<ArcStr, Vec<u16>>,
    /// Lookup: table ID -> label.
    table_id_to_label: Vec<ArcStr>,
    /// Lookup: rel table ID -> edge type.
    rel_table_id_to_type: Vec<ArcStr>,
    /// Pre-computed: for each node table_id, the rel_table_ids where it is the source.
    src_rel_table_ids: Vec<Vec<u16>>,
    /// Pre-computed: for each node table_id, the rel_table_ids where it is the destination.
    dst_rel_table_ids: Vec<Vec<u16>>,
    /// Cached statistics.
    statistics: Arc<Statistics>,

    // ── ID-preserving maps (for layered store integration) ──────────
    /// Original `NodeId`s and their (table_id, row_offset). Present when
    /// the store preserves the ids of the store it was built from.
    node_ids: Option<IdMap>,
    /// Original `EdgeId`s and their (rel_table_id, csr_position).
    edge_ids: Option<IdMap>,

    /// Indexed node properties: for each, every table carrying the
    /// property with its rows in value order.
    ///
    /// Purely an accelerator for
    /// [`find_nodes_by_property`](crate::graph::traits::GraphStoreSearch::find_nodes_by_property),
    /// which otherwise zone-map-prunes and then scans the surviving
    /// columns — linear in the table, and orders of magnitude slower than
    /// the `LpgStore` property index it replaces after a compaction. A
    /// section stores each order beside its column, so a reopened store
    /// serves them in place.
    property_value_indexes: parking_lot::RwLock<FxHashMap<PropertyKey, Arc<PropertyValueIndex>>>,

    /// The section buffer a deserialized store's column bodies are views
    /// into; empty for a store built in memory.
    section: SectionSpan,
    /// Page checksums of the mapped section the store reads in place,
    /// verified as reads first touch each page.
    pages: Option<Arc<SectionPages>>,
}

/// One property's index: `(table_id, rows in value order)` for every node
/// table carrying the property. See
/// [`CompactStore::enable_property_indexes`].
type PropertyValueIndex = Vec<(u16, RowOrder)>;

impl std::fmt::Debug for CompactStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompactStore")
            .field("node_tables_by_id", &self.node_tables_by_id)
            .field("rel_tables_by_id", &self.rel_tables_by_id)
            .field("table_id_to_label", &self.table_id_to_label)
            .field("rel_table_id_to_type", &self.rel_table_id_to_type)
            .finish_non_exhaustive()
    }
}

impl CompactStore {
    /// Creates a new `CompactStore` from pre-built components.
    ///
    /// Prefer using [`CompactStoreBuilder`] which validates schemas and
    /// computes statistics automatically. This constructor is `pub(crate)`
    /// because it assumes all invariants are already satisfied.
    #[must_use]
    pub(crate) fn new(
        node_tables_by_id: Vec<NodeTable>,
        label_to_table_id: FxHashMap<ArcStr, u16>,
        rel_tables_by_id: Vec<RelTable>,
        edge_type_to_rel_id: FxHashMap<ArcStr, Vec<u16>>,
        table_id_to_label: Vec<ArcStr>,
        rel_table_id_to_type: Vec<ArcStr>,
        statistics: Statistics,
    ) -> Self {
        // Pre-compute src/dst rel_table_id mappings per node table_id.
        let node_table_count = node_tables_by_id.len();
        let mut src_rel_table_ids = vec![Vec::new(); node_table_count];
        let mut dst_rel_table_ids = vec![Vec::new(); node_table_count];

        debug_assert!(
            rel_tables_by_id.len() <= usize::from(id::MAX_TABLE_ID) + 1,
            "rel table count {} exceeds 15-bit limit; caller must validate",
            rel_tables_by_id.len()
        );
        for (rel_idx, rt) in rel_tables_by_id.iter().enumerate() {
            // Caller (CompactStoreBuilder::build) validates table count fits u16.
            let rel_id = u16::try_from(rel_idx).expect("caller validated table count");

            let src_tid = rt.src_table_id() as usize;
            let dst_tid = rt.dst_table_id() as usize;
            if src_tid < node_table_count {
                src_rel_table_ids[src_tid].push(rel_id);
            }
            if dst_tid < node_table_count {
                dst_rel_table_ids[dst_tid].push(rel_id);
            }
        }

        Self {
            node_tables_by_id,
            label_to_table_id,
            rel_tables_by_id,
            edge_type_to_rel_id,
            table_id_to_label,
            rel_table_id_to_type,
            src_rel_table_ids,
            dst_rel_table_ids,
            statistics: Arc::new(statistics),
            node_ids: None,
            edge_ids: None,
            property_value_indexes: parking_lot::RwLock::new(FxHashMap::default()),
            section: SectionSpan::default(),
            pages: None,
        }
    }

    /// Resolves a table_id to its [`NodeTable`].
    #[inline]
    fn resolve_node_table(&self, table_id: u16) -> Option<&NodeTable> {
        self.node_tables_by_id.get(table_id as usize)
    }

    /// Resolves a rel_table_id to its [`RelTable`].
    #[inline]
    fn resolve_rel_table(&self, rel_table_id: u16) -> Option<&RelTable> {
        self.rel_tables_by_id.get(rel_table_id as usize)
    }

    /// Returns a reference to the node table for the given label, if any.
    #[must_use]
    pub fn node_table(&self, label: &str) -> Option<&NodeTable> {
        let &tid = self.label_to_table_id.get(label)?;
        self.node_tables_by_id.get(tid as usize)
    }

    /// Returns a reference to the first relationship table for the given edge type.
    ///
    /// When an edge type spans multiple label pairs, use [`Self::rel_tables_for_type`]
    /// to get all matching tables.
    #[must_use]
    pub fn rel_table(&self, edge_type: &str) -> Option<&RelTable> {
        let rids = self.edge_type_to_rel_id.get(edge_type)?;
        let &rid = rids.first()?;
        self.rel_tables_by_id.get(rid as usize)
    }

    /// Returns all relationship tables for the given edge type.
    #[must_use]
    pub fn rel_tables_for_type(&self, edge_type: &str) -> Vec<&RelTable> {
        self.edge_type_to_rel_id
            .get(edge_type)
            .map(|rids| {
                rids.iter()
                    .filter_map(|&rid| self.rel_tables_by_id.get(rid as usize))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Returns the label for a given table ID, if valid.
    #[must_use]
    pub fn label_for_table_id(&self, table_id: u16) -> Option<&ArcStr> {
        self.table_id_to_label.get(table_id as usize)
    }

    /// Returns the edge type for a given rel table ID, if valid.
    #[must_use]
    pub fn edge_type_for_rel_table_id(&self, rel_table_id: u16) -> Option<&ArcStr> {
        self.rel_table_id_to_type.get(rel_table_id as usize)
    }

    /// Collects edges from snapshot RelTables for a given node in a direction.
    ///
    /// When ID-preserving, the returned `NodeId`/`EdgeId` values are translated
    /// back to the original IDs from the source store.
    fn collect_edges(
        &self,
        node_table_id: u16,
        node_offset: u32,
        direction: Direction,
    ) -> Vec<(NodeId, EdgeId)> {
        let tid = node_table_id as usize;
        let mut results = Vec::new();

        if matches!(direction, Direction::Outgoing | Direction::Both)
            && let Some(rel_ids) = self.src_rel_table_ids.get(tid)
        {
            for &rel_id in rel_ids {
                let rt = &self.rel_tables_by_id[rel_id as usize];
                results.extend(rt.edges_from_source(node_offset));
            }
        }

        if matches!(direction, Direction::Incoming | Direction::Both)
            && let Some(rel_ids) = self.dst_rel_table_ids.get(tid)
        {
            for &rel_id in rel_ids {
                let rt = &self.rel_tables_by_id[rel_id as usize];
                if let Some(edges) = rt.edges_to_target(node_offset) {
                    results.extend(edges);
                }
            }
        }

        // Translate compact-encoded IDs to original IDs when preserving.
        if self.preserves_ids() {
            for (target_id, edge_id) in &mut results {
                *target_id = self.to_original_node_id(*target_id);
                *edge_id = self.to_original_edge_id(*edge_id);
            }
        }

        results
    }

    /// Heap bytes the store owns: every table, lookup map, and id map, the
    /// statistics, and the property value indexes.
    ///
    /// A store read from a section keeps its column bodies as views into
    /// that section's buffer; those bytes are the section's, reported by
    /// [`section_bytes`](Self::section_bytes), and are not counted here.
    #[must_use]
    pub fn heap_bytes(&self) -> usize {
        let section = &self.section;
        let vecs_bytes =
            |vecs: &Vec<Vec<u16>>| vec_bytes(vecs) + vecs.iter().map(vec_bytes).sum::<usize>();
        let labels_bytes = |labels: &Vec<ArcStr>| {
            vec_bytes(labels) + labels.iter().map(arcstr_bytes).sum::<usize>()
        };
        vec_bytes(&self.node_tables_by_id)
            + self
                .node_tables_by_id
                .iter()
                .map(|table| table.heap_bytes(section))
                .sum::<usize>()
            + vec_bytes(&self.rel_tables_by_id)
            + self
                .rel_tables_by_id
                .iter()
                .map(|table| table.heap_bytes(section))
                .sum::<usize>()
            // The label and edge-type keys share their allocations with
            // the id-to-name vectors.
            + self.label_to_table_id.allocation_size()
            + self.edge_type_to_rel_id.allocation_size()
            + self.edge_type_to_rel_id.values().map(vec_bytes).sum::<usize>()
            + labels_bytes(&self.table_id_to_label)
            + labels_bytes(&self.rel_table_id_to_type)
            + vecs_bytes(&self.src_rel_table_ids)
            + vecs_bytes(&self.dst_rel_table_ids)
            + statistics_bytes(&self.statistics)
            + self.id_map_heap_bytes()
            + self.property_index_heap_bytes()
    }

    /// Length of the section buffer the column bodies are views into, or 0
    /// for a store built in memory. The buffer is mapped from the file on
    /// the zero-copy open and a heap copy otherwise; its holder knows which.
    #[must_use]
    pub fn section_bytes(&self) -> usize {
        self.section.len()
    }

    // ── ID-preserving accessors ────────────────────────────────────

    /// Returns `true` if original IDs are preserved (built via
    /// [`from_graph_store_preserving_ids`]).
    #[must_use]
    pub fn preserves_ids(&self) -> bool {
        self.node_ids.is_some()
    }

    /// Attaches ID maps from each table's position-ordered original ids.
    /// `NodeId::INVALID` / `EdgeId::INVALID` positions are padding.
    pub(crate) fn set_id_maps(
        &mut self,
        node_positions: Vec<Vec<NodeId>>,
        edge_positions: Vec<Vec<EdgeId>>,
    ) {
        let node_positions = node_positions
            .into_iter()
            .map(|ids| ids.iter().map(NodeId::as_u64).collect())
            .collect();
        let edge_positions = edge_positions
            .into_iter()
            .map(|ids| ids.iter().map(EdgeId::as_u64).collect())
            .collect();
        self.attach_id_maps(
            IdMap::from_positions(node_positions, NodeId::INVALID.as_u64()),
            IdMap::from_positions(edge_positions, EdgeId::INVALID.as_u64()),
        );
    }

    pub(crate) fn attach_id_maps(&mut self, node_ids: IdMap, edge_ids: IdMap) {
        self.node_ids = Some(node_ids);
        self.edge_ids = Some(edge_ids);
    }

    pub(crate) fn id_maps(&self) -> Option<(&IdMap, &IdMap)> {
        Some((self.node_ids.as_ref()?, self.edge_ids.as_ref()?))
    }

    /// The first page checksum failure a read of this store's mapped
    /// section found, if any.
    ///
    /// A store read in place from a sealed section verifies each page the
    /// first time a read touches it. A read that lands on a corrupt page
    /// returns nothing for that page; this is where its holder learns the
    /// read was refused rather than empty.
    #[must_use]
    pub fn integrity_fault(&self) -> Option<String> {
        self.pages.as_ref()?.fault().map(str::to_owned)
    }

    /// Page checksum progress of the mapped section the store reads in
    /// place: `(verified pages, total pages)`, or `None` for a store that
    /// is not read in place from a paged section.
    #[must_use]
    pub fn verified_pages(&self) -> Option<(usize, usize)> {
        let pages = self.pages.as_ref()?;
        Some((pages.verified_pages(), pages.page_count()))
    }

    /// Returns the highest `NodeId` the base owns, or `None` when it holds
    /// no nodes.
    ///
    /// The layered overlay seeds its ID allocator from this so IDs minted
    /// after a reopen cannot collide with — and thereby shadow or
    /// tombstone — a base row.
    #[must_use]
    pub fn max_node_id(&self) -> Option<NodeId> {
        if let Some(ids) = &self.node_ids {
            return ids.max_id().map(NodeId::new);
        }
        // Synthetic IDs: the largest one a table owns is its last row.
        self.node_tables_by_id
            .iter()
            .enumerate()
            .filter(|(_, nt)| !nt.is_empty())
            .filter_map(|(tid, nt)| {
                let table_id = u16::try_from(tid).ok()?;
                Some(id::encode_node_id(table_id, nt.len() as u64 - 1))
            })
            .max()
    }

    /// Declares which node properties should be served from an index,
    /// building each one not already indexed now.
    ///
    /// Mirrors [`LpgStore::create_property_index`](crate::graph::lpg::LpgStore::create_property_index):
    /// the engine calls this with the source store's indexed properties
    /// after a compaction, and again after a reload, so an indexed lookup
    /// stays a binary search once the rows move into the columnar base
    /// instead of falling back to a column scan.
    ///
    /// The columns are immutable, so an index already present (built here
    /// or read with the section) is kept as is. Properties no table carries
    /// in an orderable column are ignored.
    pub fn enable_property_indexes<I>(&self, keys: I)
    where
        I: IntoIterator<Item = PropertyKey>,
    {
        for key in keys {
            if self.property_value_indexes.read().contains_key(&key) {
                continue;
            }
            let Some(index) = self.build_property_value_index(&key) else {
                continue;
            };
            self.property_value_indexes
                .write()
                .insert(key, Arc::new(index));
        }
    }

    /// Installs indexes a section stored beside their columns.
    pub(crate) fn attach_property_indexes(
        &mut self,
        indexes: FxHashMap<PropertyKey, PropertyValueIndex>,
    ) {
        *self.property_value_indexes.get_mut() = indexes
            .into_iter()
            .map(|(key, index)| (key, Arc::new(index)))
            .collect();
    }

    /// Drops the hash index for `key`, returning whether one existed.
    ///
    /// The inverse of [`Self::enable_property_indexes`]: the engine calls
    /// this when an index is dropped after a compaction, so
    /// `has_property_index` stops reporting the property as indexed and
    /// lookups fall back to the column scan.
    pub fn disable_property_index(&self, key: &PropertyKey) -> bool {
        self.property_value_indexes.write().remove(key).is_some()
    }

    /// Returns the property names currently served from a hash index.
    #[must_use]
    pub fn indexed_property_keys(&self) -> Vec<PropertyKey> {
        self.property_value_indexes.read().keys().cloned().collect()
    }

    /// Returns the hash index for `key`, if one was declared.
    pub(crate) fn property_value_index(
        &self,
        key: &PropertyKey,
    ) -> Option<Arc<PropertyValueIndex>> {
        self.property_value_indexes.read().get(key).cloned()
    }

    /// Orders the rows of every table carrying `key` by value.
    ///
    /// Returns `None` when no table has an orderable column for the
    /// property, so a stale index name costs nothing.
    fn build_property_value_index(&self, key: &PropertyKey) -> Option<PropertyValueIndex> {
        let index: PropertyValueIndex = self
            .node_tables_by_id
            .iter()
            .filter_map(|nt| {
                let order = build_row_order(nt.column(key)?, nt.null_mask(key))?;
                Some((nt.table_id(), order))
            })
            .collect();
        (!index.is_empty()).then_some(index)
    }

    /// Returns the highest `EdgeId` the base owns, or `None` when it holds
    /// no edges.
    ///
    /// See [`max_node_id`](Self::max_node_id).
    #[must_use]
    pub fn max_edge_id(&self) -> Option<EdgeId> {
        if let Some(ids) = &self.edge_ids {
            return ids.max_id().map(EdgeId::new);
        }
        self.rel_tables_by_id
            .iter()
            .enumerate()
            .filter(|(_, rt)| rt.num_edges() > 0)
            .filter_map(|(rid, rt)| {
                let rel_table_id = u16::try_from(rid).ok()?;
                Some(id::encode_edge_id(rel_table_id, rt.num_edges() as u64 - 1))
            })
            .max()
    }

    /// Resolves an input `NodeId` to (table_id, offset).
    ///
    /// When ID-preserving, looks up the original ID in the map.
    /// Otherwise, decodes the compact-encoded bits.
    #[inline]
    pub(crate) fn resolve_node(&self, id: NodeId) -> Option<(u16, u64)> {
        if let Some(ref map) = self.node_ids {
            map.resolve(id.as_u64())
        } else {
            Some(id::decode_node_id(id))
        }
    }

    /// Resolves an input `EdgeId` to (rel_table_id, csr_position).
    #[inline]
    pub(crate) fn resolve_edge(&self, id: EdgeId) -> Option<(u16, u64)> {
        if let Some(ref map) = self.edge_ids {
            map.resolve(id.as_u64())
        } else {
            Some(id::decode_edge_id(id))
        }
    }

    /// Translates a compact-encoded `NodeId` (from internal CSR/table lookups)
    /// back to the original preserved ID. No-op when not ID-preserving.
    #[inline]
    pub(crate) fn to_original_node_id(&self, compact_id: NodeId) -> NodeId {
        if let Some(ref map) = self.node_ids {
            let (table_id, offset) = id::decode_node_id(compact_id);
            map.original(table_id, offset)
                .map_or(compact_id, NodeId::new)
        } else {
            compact_id
        }
    }

    /// Translates a compact-encoded `EdgeId` back to the original preserved ID.
    #[inline]
    pub(crate) fn to_original_edge_id(&self, compact_id: EdgeId) -> EdgeId {
        if let Some(ref map) = self.edge_ids {
            let (rel_table_id, csr_pos) = id::decode_edge_id(compact_id);
            map.original(rel_table_id, csr_pos)
                .map_or(compact_id, EdgeId::new)
        } else {
            compact_id
        }
    }

    /// Heap bytes of the property indexes: the index map, each index's
    /// shared allocation and table list, and every in-memory row order.
    fn property_index_heap_bytes(&self) -> usize {
        let indexes = self.property_value_indexes.read();
        indexes.allocation_size()
            + indexes
                .iter()
                .map(|(key, index)| {
                    key_bytes(key)
                        + arc_slice_bytes::<PropertyValueIndex>(1)
                        + vec_bytes(index.as_ref())
                        + index
                            .iter()
                            .map(|(_, order)| order.heap_bytes())
                            .sum::<usize>()
                })
                .sum::<usize>()
    }

    fn id_map_heap_bytes(&self) -> usize {
        self.node_ids.as_ref().map_or(0, IdMap::heap_bytes)
            + self.edge_ids.as_ref().map_or(0, IdMap::heap_bytes)
    }
}
