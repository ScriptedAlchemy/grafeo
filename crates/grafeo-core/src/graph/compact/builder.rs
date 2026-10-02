//! Builder for constructing a [`CompactStore`] from raw data.
//!
//! The builder provides a fluent API for defining node tables, relationship
//! tables, and their columns. Data is loaded in bulk at construction time,
//! producing an immutable, read-only store.

use arcstr::ArcStr;
use grafeo_common::types::{PropertyKey, Value};
use grafeo_common::utils::hash::{FxHashMap, FxHashSet};
use thiserror::Error;

use super::CompactStore;
use super::column::ColumnCodec;
use super::csr::CsrAdjacency;
use super::id::MAX_TABLE_ID;
use super::node_table::NodeTable;
use super::rel_table::RelTable;
use super::schema::{ColumnDef, ColumnType, EdgeSchema, TableSchema};
use super::zone_map::ZoneMap;
use crate::codec::{BitPackedInts, BitVector, DictionaryBuilder};
use crate::statistics::{EdgeTypeStatistics, LabelStatistics, Statistics};

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur while building a [`CompactStore`].
#[derive(Debug, Clone, Error)]
#[non_exhaustive]
pub enum CompactStoreError {
    /// A relationship table references a node label that was not defined.
    #[error("node label not found: {0:?}")]
    LabelNotFound(String),
    /// A column was added with a length that does not match the table.
    #[error("column length mismatch: expected {expected} rows, got {got}")]
    ColumnLengthMismatch {
        /// Expected number of rows (inferred from the first column added).
        expected: usize,
        /// Actual number of rows in the column.
        got: usize,
    },
    /// Two node tables were defined with the same label.
    #[error("duplicate node label: {0:?}")]
    DuplicateLabel(String),
    /// Two relationship tables were defined with the same (edge type, src, dst) triple.
    #[error("duplicate edge type: {0:?}")]
    DuplicateEdgeType(String),
    /// A backward edge has no corresponding forward edge (data inconsistency).
    #[error("inconsistent edge data: {0}")]
    InconsistentEdgeData(String),
    /// A bit-packed column contains a value that exceeds `i64::MAX`.
    #[error("value overflow in column {column:?}: {value} exceeds i64::MAX ({max})")]
    ValueOverflow {
        /// Column name.
        column: String,
        /// The offending value.
        value: u64,
        /// Maximum allowed value.
        max: u64,
    },
    /// The number of tables exceeds the compact ID encoding limit (15-bit table ID).
    #[error("table count {count} exceeds compact ID limit of {max} ({kind} tables)")]
    TableCountOverflow {
        /// Kind of table ("node" or "relationship").
        kind: &'static str,
        /// Actual table count.
        count: usize,
        /// Maximum allowed table count.
        max: u16,
    },
    /// A node table received more rows than a `u32` offset can address.
    #[error("node table {table:?} exceeds the u32 row limit")]
    TableRowOverflow {
        /// Label key of the overfull table.
        table: String,
    },
    /// A node was pushed without any label; compact tables are keyed by label.
    #[error("node {0} has no label")]
    UnlabeledNode(u64),
    /// The same node id was pushed twice.
    #[error("duplicate node id {0}")]
    DuplicateNodeId(u64),
    /// The same edge id was pushed twice.
    #[error("duplicate edge id {0}")]
    DuplicateEdgeId(u64),
    /// An edge names an endpoint that was never pushed as a node.
    #[error("edge {edge} references unknown node {node}")]
    UnknownEndpoint {
        /// The edge id.
        edge: u64,
        /// The missing endpoint's node id.
        node: u64,
    },
    /// The incremental builder's column spool could not record or return a
    /// value.
    #[error("column spool: {0}")]
    Spool(String),
}

// ---------------------------------------------------------------------------
// NodeTableBuilder
// ---------------------------------------------------------------------------

/// Builder for node table columns. Obtained through [`CompactStoreBuilder::node_table`].
pub struct NodeTableBuilder {
    label: ArcStr,
    columns: Vec<(PropertyKey, ColumnCodec)>,
    zone_maps: Vec<(PropertyKey, ZoneMap)>,
    null_masks: Vec<(PropertyKey, BitVector)>,
    len: Option<usize>,
    length_mismatch: Option<(usize, usize)>,
    value_overflow: Option<(String, u64)>,
}

impl NodeTableBuilder {
    fn new(label: impl Into<ArcStr>) -> Self {
        Self {
            label: label.into(),
            columns: Vec::new(),
            zone_maps: Vec::new(),
            null_masks: Vec::new(),
            len: None,
            length_mismatch: None,
            value_overflow: None,
        }
    }

    /// Adds a bit-packed integer column.
    ///
    /// `bits` is the number of bits per value. Values are packed using
    /// [`BitPackedInts::pack_with_bits`]. All values must fit in `i64`
    /// (i.e., be at most `i64::MAX`); overflow is recorded and reported
    /// as [`CompactStoreError::ValueOverflow`] at build time.
    pub fn column_bitpacked(&mut self, name: &str, values: &[u64], bits: u8) -> &mut Self {
        self.record_len(values.len());

        // Validate that all values fit in i64.
        if let Some(&bad) = values.iter().find(|&&v| v > i64::MAX as u64) {
            self.value_overflow = Some((name.to_string(), bad));
        }

        let bp = BitPackedInts::pack_with_bits(values, bits);

        // Compute zone map from raw values.
        let zone_map = compute_zone_map_u64(values);
        self.zone_maps.push((PropertyKey::new(name), zone_map));

        self.columns
            .push((PropertyKey::new(name), ColumnCodec::BitPacked(bp)));
        self
    }

    /// Adds a dictionary-encoded string column.
    pub fn column_dict(&mut self, name: &str, values: &[&str]) -> &mut Self {
        self.record_len(values.len());

        let mut builder = DictionaryBuilder::new();
        for &v in values {
            builder.add(v);
        }
        let dict = builder.build();

        // Compute zone map for strings.
        let zone_map = compute_zone_map_strings(values);
        self.zone_maps.push((PropertyKey::new(name), zone_map));

        self.columns
            .push((PropertyKey::new(name), ColumnCodec::Dict(dict)));
        self
    }

    /// Adds an int8 quantised vector column (for embeddings).
    ///
    /// # Panics
    ///
    /// Panics if `data.len()` is not a multiple of `dimensions`.
    pub fn column_int8_vector(&mut self, name: &str, data: Vec<i8>, dimensions: u16) -> &mut Self {
        let dims = dimensions as usize;
        let row_count = if dims == 0 {
            0
        } else {
            assert!(
                data.len().is_multiple_of(dims),
                "Int8Vector data length {} is not a multiple of dimensions {dimensions}",
                data.len(),
            );
            data.len() / dims
        };
        self.record_len(row_count);

        // No meaningful zone map for vector columns.
        self.columns.push((
            PropertyKey::new(name),
            ColumnCodec::int8_vector(data, dimensions),
        ));
        self
    }

    /// Adds a boolean bitmap column.
    pub fn column_bitmap(&mut self, name: &str, values: &[bool]) -> &mut Self {
        self.record_len(values.len());

        let bv = BitVector::from_bools(values);

        // Zone map for booleans.
        let zone_map = compute_zone_map_bool(values);
        self.zone_maps.push((PropertyKey::new(name), zone_map));

        self.columns
            .push((PropertyKey::new(name), ColumnCodec::Bitmap(bv)));
        self
    }

    /// Adds a pre-built column codec (for advanced use).
    pub fn column(&mut self, name: &str, codec: ColumnCodec) -> &mut Self {
        self.record_len(codec.len());
        self.columns.push((PropertyKey::new(name), codec));
        self
    }

    /// Adds a column whose codec is inferred from its row values (the
    /// `from_graph_store` type mapping), with the zone map that codec
    /// supports and the mask of rows that hold no value.
    fn inferred_column(&mut self, key: &PropertyKey, values: &[Value]) -> &mut Self {
        let (codec, zone_map, nulls) = encode_inferred_column(values);
        self.record_len(values.len());
        if let Some(zone_map) = zone_map {
            self.zone_maps.push((key.clone(), zone_map));
        }
        if let Some(nulls) = nulls {
            self.null_masks.push((key.clone(), nulls));
        }
        self.columns.push((key.clone(), codec));
        self
    }

    /// Records the row count from the first column and validates subsequent ones.
    fn record_len(&mut self, col_len: usize) {
        match self.len {
            None => self.len = Some(col_len),
            Some(expected) => {
                if expected != col_len {
                    self.length_mismatch = Some((expected, col_len));
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// RelTableBuilder
// ---------------------------------------------------------------------------

/// Builder for relationship table edges and properties. Obtained through [`CompactStoreBuilder::rel_table`].
pub struct RelTableBuilder {
    edge_type: ArcStr,
    src_label: ArcStr,
    dst_label: ArcStr,
    edges: Vec<(u32, u32)>,
    backward: bool,
    properties: Vec<(PropertyKey, ColumnCodec)>,
    null_masks: Vec<(PropertyKey, BitVector)>,
}

impl RelTableBuilder {
    fn new(
        edge_type: impl Into<ArcStr>,
        src_label: impl Into<ArcStr>,
        dst_label: impl Into<ArcStr>,
    ) -> Self {
        Self {
            edge_type: edge_type.into(),
            src_label: src_label.into(),
            dst_label: dst_label.into(),
            edges: Vec::new(),
            backward: false,
            properties: Vec::new(),
            null_masks: Vec::new(),
        }
    }

    /// Sets the `(src_offset, dst_offset)` edge pairs.
    pub fn edges(&mut self, pairs: impl Into<Vec<(u32, u32)>>) -> &mut Self {
        self.edges = pairs.into();
        self
    }

    /// Enables or disables backward CSR construction.
    pub fn backward(&mut self, enabled: bool) -> &mut Self {
        self.backward = enabled;
        self
    }

    /// Adds a bit-packed property column on edges.
    pub fn column_bitpacked(&mut self, name: &str, values: &[u64], bits: u8) -> &mut Self {
        let bp = BitPackedInts::pack_with_bits(values, bits);
        self.properties
            .push((PropertyKey::new(name), ColumnCodec::BitPacked(bp)));
        self
    }

    /// Adds an edge property column whose codec is inferred from its row
    /// values (the `from_graph_store` type mapping), with the mask of rows
    /// that hold no value. Edge columns carry no zone maps.
    fn inferred_column(&mut self, key: &PropertyKey, values: &[Value]) -> &mut Self {
        let (codec, _zone_map, nulls) = encode_inferred_column(values);
        if let Some(nulls) = nulls {
            self.null_masks.push((key.clone(), nulls));
        }
        self.properties.push((key.clone(), codec));
        self
    }
}

// ---------------------------------------------------------------------------
// CompactStoreBuilder
// ---------------------------------------------------------------------------

/// Fluent builder for constructing a [`CompactStore`] from raw data.
///
/// # Example
///
/// ```ignore
/// let store = CompactStoreBuilder::new()
///     .node_table("Person", |t| {
///         t.column_bitpacked("age", &[25, 30, 35], 6)
///          .column_dict("name", &["Alix", "Gus", "Vincent"])
///     })
///     .build()
///     .unwrap();
/// ```
#[derive(Default)]
pub struct CompactStoreBuilder {
    node_table_builders: Vec<NodeTableBuilder>,
    rel_table_builders: Vec<RelTableBuilder>,
}

impl CompactStoreBuilder {
    /// Creates a new empty builder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Defines a node table with the given label.
    ///
    /// The closure receives a [`NodeTableBuilder`] that can be used to add
    /// columns.
    pub fn node_table(
        mut self,
        label: &str,
        f: impl FnOnce(&mut NodeTableBuilder) -> &mut NodeTableBuilder,
    ) -> Self {
        let mut builder = NodeTableBuilder::new(label);
        f(&mut builder);
        self.node_table_builders.push(builder);
        self
    }

    /// Defines a relationship table connecting two node labels.
    ///
    /// The closure receives a [`RelTableBuilder`] that can be used to set
    /// edges, backward CSR, and properties.
    pub fn rel_table(
        mut self,
        edge_type: &str,
        src_label: &str,
        dst_label: &str,
        f: impl FnOnce(&mut RelTableBuilder) -> &mut RelTableBuilder,
    ) -> Self {
        let mut builder = RelTableBuilder::new(edge_type, src_label, dst_label);
        f(&mut builder);
        self.rel_table_builders.push(builder);
        self
    }

    /// Consumes the builder and constructs a [`CompactStore`].
    ///
    /// # Errors
    ///
    /// Returns [`CompactStoreError::LabelNotFound`] if a relationship table
    /// references a node label that was not defined.
    pub fn build(self) -> Result<CompactStore, CompactStoreError> {
        // Step 1: Validate column length mismatches and value overflows.
        for ntb in &self.node_table_builders {
            if let Some((expected, got)) = ntb.length_mismatch {
                return Err(CompactStoreError::ColumnLengthMismatch { expected, got });
            }
            if let Some((ref column, value)) = ntb.value_overflow {
                return Err(CompactStoreError::ValueOverflow {
                    column: column.clone(),
                    max: i64::MAX as u64,
                    value,
                });
            }
        }

        // Step 2: Validate no duplicate labels.
        {
            let mut seen_labels = FxHashSet::default();
            for ntb in &self.node_table_builders {
                if !seen_labels.insert(&ntb.label) {
                    return Err(CompactStoreError::DuplicateLabel(ntb.label.to_string()));
                }
            }
        }

        // Step 2b: Validate no duplicate (edge_type, src_label, dst_label) triples.
        {
            let mut seen_triples = FxHashSet::default();
            for rtb in &self.rel_table_builders {
                if !seen_triples.insert((&rtb.edge_type, &rtb.src_label, &rtb.dst_label)) {
                    return Err(CompactStoreError::DuplicateEdgeType(format!(
                        "{} ({} -> {})",
                        rtb.edge_type, rtb.src_label, rtb.dst_label
                    )));
                }
            }
        }

        // Step 2c: Validate table counts fit within the 15-bit compact ID encoding.
        let max_tables = usize::from(MAX_TABLE_ID) + 1; // 32768
        if self.node_table_builders.len() > max_tables {
            return Err(CompactStoreError::TableCountOverflow {
                kind: "node",
                count: self.node_table_builders.len(),
                max: MAX_TABLE_ID,
            });
        }
        if self.rel_table_builders.len() > max_tables {
            return Err(CompactStoreError::TableCountOverflow {
                kind: "relationship",
                count: self.rel_table_builders.len(),
                max: MAX_TABLE_ID,
            });
        }

        // Step 3: Assign sequential table IDs.
        let mut label_to_table_id: FxHashMap<ArcStr, u16> = FxHashMap::default();
        let mut table_id_to_label: Vec<ArcStr> = Vec::new();

        for (idx, ntb) in self.node_table_builders.iter().enumerate() {
            // Validated in Step 2c: count <= MAX_TABLE_ID + 1, so idx fits u16.
            let table_id =
                u16::try_from(idx).map_err(|_| CompactStoreError::TableCountOverflow {
                    kind: "node",
                    count: idx,
                    max: MAX_TABLE_ID,
                })?;
            label_to_table_id.insert(ntb.label.clone(), table_id);
            table_id_to_label.push(ntb.label.clone());
        }

        // Step 4: Build each NodeTable.
        let mut node_tables_by_id: Vec<NodeTable> =
            Vec::with_capacity(self.node_table_builders.len());

        for (idx, ntb) in self.node_table_builders.into_iter().enumerate() {
            // Validated in Step 2c: count <= MAX_TABLE_ID + 1, so idx fits u16.
            let table_id =
                u16::try_from(idx).map_err(|_| CompactStoreError::TableCountOverflow {
                    kind: "node",
                    count: idx,
                    max: MAX_TABLE_ID,
                })?;
            let row_count = ntb.len.unwrap_or(0);

            // Build column definitions for the schema.
            let col_defs: Vec<ColumnDef> = ntb
                .columns
                .iter()
                .map(|(key, codec)| {
                    let col_type = infer_column_type(codec);
                    ColumnDef::new(key.as_str(), col_type)
                })
                .collect();

            let schema = TableSchema::new(ntb.label.as_str(), table_id, col_defs);

            let columns: FxHashMap<PropertyKey, ColumnCodec> = ntb.columns.into_iter().collect();

            let zone_maps: FxHashMap<PropertyKey, ZoneMap> = ntb.zone_maps.into_iter().collect();

            // Phase 2c: compute per-block zone maps so range scans (Phase 4)
            // can skip entire blocks whose stats prove no match. Empty
            // when a column is empty; otherwise one entry per block.
            let block_zone_maps: FxHashMap<PropertyKey, Vec<ZoneMap>> = columns
                .iter()
                .map(|(key, codec)| (key.clone(), super::zone_map::compute_block_zone_maps(codec)))
                .collect();

            let table = NodeTable::from_columns_with_block_stats(
                schema,
                columns,
                zone_maps,
                block_zone_maps,
                row_count,
            )
            .with_null_masks(ntb.null_masks.into_iter().collect());
            node_tables_by_id.push(table);
        }

        // Step 5: Build each RelTable.
        let mut rel_tables_by_id: Vec<RelTable> = Vec::with_capacity(self.rel_table_builders.len());
        let mut edge_type_to_rel_id: FxHashMap<ArcStr, Vec<u16>> = FxHashMap::default();
        let mut rel_table_id_to_type: Vec<ArcStr> = Vec::new();

        for (idx, rtb) in self.rel_table_builders.into_iter().enumerate() {
            // Validated in Step 2c: count <= MAX_TABLE_ID + 1, so idx fits u16.
            let rel_table_id =
                u16::try_from(idx).map_err(|_| CompactStoreError::TableCountOverflow {
                    kind: "relationship",
                    count: idx,
                    max: MAX_TABLE_ID,
                })?;
            rel_table_id_to_type.push(rtb.edge_type.clone());

            // Resolve labels to table IDs.
            let src_table_id = *label_to_table_id
                .get(&rtb.src_label)
                .ok_or_else(|| CompactStoreError::LabelNotFound(rtb.src_label.to_string()))?;
            let dst_table_id = *label_to_table_id
                .get(&rtb.dst_label)
                .ok_or_else(|| CompactStoreError::LabelNotFound(rtb.dst_label.to_string()))?;

            // Get source and destination node counts for CSR sizing.
            let src_node_count = node_tables_by_id
                .get(src_table_id as usize)
                .map_or(0, |t| t.len());
            let dst_node_count = node_tables_by_id
                .get(dst_table_id as usize)
                .map_or(0, |t| t.len());

            let (fwd, bwd) =
                rel_adjacency(&rtb.edges, src_node_count, dst_node_count, rtb.backward)?;

            // Build edge property columns.
            let property_col_defs: Vec<ColumnDef> = rtb
                .properties
                .iter()
                .map(|(key, codec)| {
                    let col_type = infer_column_type(codec);
                    ColumnDef::new(key.as_str(), col_type)
                })
                .collect();

            let schema = EdgeSchema::new(
                rtb.edge_type.as_str(),
                rel_table_id,
                rtb.src_label.as_str(),
                rtb.dst_label.as_str(),
                property_col_defs,
            );

            let properties: FxHashMap<PropertyKey, ColumnCodec> =
                rtb.properties.into_iter().collect();

            let table = RelTable::new(schema, fwd, bwd, properties, src_table_id, dst_table_id)
                .with_null_masks(rtb.null_masks.into_iter().collect());
            edge_type_to_rel_id
                .entry(rtb.edge_type.clone())
                .or_default()
                .push(rel_table_id);
            rel_tables_by_id.push(table);
        }

        // Step 6: Compute initial Statistics.
        let mut stats = Statistics::new();
        let mut total_nodes: u64 = 0;
        let mut total_edges: u64 = 0;

        for (idx, nt) in node_tables_by_id.iter().enumerate() {
            let count = nt.len() as u64;
            total_nodes += count;
            let label = &table_id_to_label[idx];
            stats.update_label(label.as_str(), LabelStatistics::new(count));
        }

        let mut edge_type_counts: FxHashMap<&str, u64> = FxHashMap::default();
        for (idx, rt) in rel_tables_by_id.iter().enumerate() {
            let count = rt.num_edges() as u64;
            total_edges += count;
            let edge_type = &rel_table_id_to_type[idx];
            *edge_type_counts.entry(edge_type.as_str()).or_default() += count;
        }
        for (edge_type, count) in edge_type_counts {
            stats.update_edge_type(edge_type, EdgeTypeStatistics::new(count, 0.0, 0.0));
        }

        stats.total_nodes = total_nodes;
        stats.total_edges = total_edges;

        // Step 7: Construct the CompactStore.
        Ok(CompactStore::new(
            node_tables_by_id,
            label_to_table_id,
            rel_tables_by_id,
            edge_type_to_rel_id,
            table_id_to_label,
            rel_table_id_to_type,
            stats,
        ))
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Builds a relationship table's forward CSR and, when `backward` is set,
/// its backward CSR with the backward-to-forward position mapping that
/// spares `edges_to_target` an O(degree) scan at query time.
fn rel_adjacency(
    edges: &[(u32, u32)],
    src_node_count: usize,
    dst_node_count: usize,
    backward: bool,
) -> Result<(CsrAdjacency, Option<CsrAdjacency>), CompactStoreError> {
    let mut fwd_edges = edges.to_vec();
    fwd_edges.sort_by_key(|&(src, _dst)| src);
    let fwd = CsrAdjacency::from_sorted_edges(src_node_count, &fwd_edges);
    drop(fwd_edges);
    if !backward {
        return Ok((fwd, None));
    }
    let mut bwd_edges: Vec<(u32, u32)> = edges.iter().map(|&(src, dst)| (dst, src)).collect();
    bwd_edges.sort_by_key(|&(dst, _src)| dst);
    let mut bwd_csr = CsrAdjacency::from_sorted_edges(dst_node_count, &bwd_edges);
    let mut mapping = Vec::with_capacity(bwd_edges.len());
    for &(dst, src) in &bwd_edges {
        let fwd_neighbors = fwd.neighbors(src);
        let fwd_start = fwd.offset_of(src);
        let local_idx = fwd_neighbors
            .iter()
            .position(|&t| t == dst)
            .ok_or_else(|| {
                CompactStoreError::InconsistentEdgeData(format!(
                    "backward edge ({dst}->{src}) has no corresponding forward edge"
                ))
            })?;
        // reason: local index within CSR neighbors fits u32
        #[allow(clippy::cast_possible_truncation)]
        mapping.push(fwd_start + local_idx as u32);
    }
    bwd_csr.set_edge_data(mapping);
    Ok((fwd, Some(bwd_csr)))
}

/// Infers a [`ColumnType`] from a [`ColumnCodec`] variant.
fn infer_column_type(codec: &ColumnCodec) -> ColumnType {
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

/// Computes a zone map from u64 values (bit-packed column).
///
/// If the maximum value exceeds `i64::MAX`, the zone map is returned without
/// min/max bounds (conservative, won't prune). This avoids incorrect ordering
/// comparisons caused by the `u64 as i64` sign-bit wrap.
fn compute_zone_map_u64(values: &[u64]) -> ZoneMap {
    let Some(&min) = values.iter().min() else {
        return ZoneMap::new();
    };
    let max = *values.iter().max().expect("non-empty after min check");
    if max > i64::MAX as u64 {
        // Values exceed i64 range: zone map would compare with wrong ordering.
        // Return conservative (no bounds) zone map.
        return ZoneMap {
            row_count: values.len(),
            ..ZoneMap::default()
        };
    }
    // reason: max <= i64::MAX checked above, min <= max
    #[allow(clippy::cast_possible_wrap)]
    ZoneMap {
        min: Some(Value::Int64(min as i64)),
        max: Some(Value::Int64(max as i64)),
        null_count: 0,
        row_count: values.len(),
    }
}

/// Computes a zone map from signed i64 values (RawI64 column).
///
/// Produces `Value::Int64` min/max, which flows naturally into `compare_values`
/// and yields correct signed ordering in predicate pushdown.
fn compute_zone_map_i64(values: &[i64]) -> ZoneMap {
    let Some(&min) = values.iter().min() else {
        return ZoneMap::new();
    };
    let max = *values.iter().max().expect("non-empty after min check");
    ZoneMap {
        min: Some(Value::Int64(min)),
        max: Some(Value::Int64(max)),
        null_count: 0,
        row_count: values.len(),
    }
}

/// Computes a zone map from string values (dict column).
fn compute_zone_map_strings(values: &[&str]) -> ZoneMap {
    let Some(&min) = values.iter().min() else {
        return ZoneMap::new();
    };
    let max = *values.iter().max().expect("non-empty after min check");
    ZoneMap {
        min: Some(Value::from(min)),
        max: Some(Value::from(max)),
        null_count: 0,
        row_count: values.len(),
    }
}

/// Computes a zone map from boolean values.
fn compute_zone_map_bool(values: &[bool]) -> ZoneMap {
    if values.is_empty() {
        return ZoneMap::new();
    }
    let has_false = values.iter().any(|&v| !v);
    let has_true = values.iter().any(|&v| v);
    let min = !has_false; // false if has_false, true if all true
    let max = has_true; // true if has_true, false if all false
    ZoneMap {
        min: Some(Value::Bool(min)),
        max: Some(Value::Bool(max)),
        null_count: 0,
        row_count: values.len(),
    }
}

// ---------------------------------------------------------------------------
// Conversion from GraphStore
// ---------------------------------------------------------------------------

/// Which columnar encoding to use for a property key, inferred from values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InferredType {
    /// All non-null values are `Value::Int64` with value >= 0.
    BitPacked,
    /// All non-null values are `Value::Int64`, with at least one negative.
    /// Uses the `ColumnCodec::RawI64` encoding so signed ordering works in
    /// `find_eq`, `find_in_range`, and zone-map comparisons, and the
    /// `Int64` type is preserved on decode.
    RawI64,
    /// All non-null values are `Value::Float64`, or mixed `Int64`+`Float64`.
    Float64,
    /// All non-null values are `Value::Bool`.
    Bitmap,
    /// All non-null values are `Value::Vector` with consistent dimensions.
    Float32Vector { dimensions: u16 },
    /// All non-null values are `Value::String`, or mixed/unsupported types.
    Dict,
}

/// Converts any [`GraphStore`](crate::graph::GraphStore) into a [`CompactStore`].
///
/// Reads all nodes grouped by label, infers column types from property values,
/// reads all edges grouped by type, and builds a `CompactStore` with backward
/// CSR enabled for every relationship table.
///
/// # Type mapping
///
/// | Source type | Codec | Notes |
/// |-------------|-------|-------|
/// | `Int64` (>= 0) | `BitPacked` | Auto bit-width via `BitPackedInts::pack` |
/// | `Bool` | `Bitmap` | |
/// | `String` | `Dict` | |
/// | All others | `Dict` | Serialized via `Display` |
///
/// Nodes with multiple labels use a canonical combined key (labels sorted,
/// joined with `|`). `Null` values are stored as zero/false/empty-string
/// depending on the inferred codec.
///
/// # Errors
///
/// Propagates any [`CompactStoreError`] from the underlying builder (e.g.
/// if there are more than 32,767 distinct labels or edge types).
pub fn from_graph_store(
    store: &dyn crate::graph::traits::GraphStore,
) -> Result<CompactStore, CompactStoreError> {
    // Step 1: Collect all nodes grouped by label, build ID mapping.
    let labels = store.all_labels();
    if labels.is_empty() {
        return CompactStoreBuilder::new().build();
    }

    // old_node_id -> (label_key, offset_within_label)
    let mut id_map: FxHashMap<grafeo_common::types::NodeId, (ArcStr, u32)> = FxHashMap::default();

    // label_key -> (ordered node IDs, property_key -> Vec<Value>)
    // We use Vec<Value> to collect per-column values in row order.
    let mut label_data: Vec<(
        ArcStr,
        Vec<grafeo_common::types::NodeId>,
        FxHashMap<PropertyKey, Vec<Value>>,
    )> = Vec::new();

    // Collect all node IDs per label. Nodes with multiple labels use a
    // compound key (sorted labels joined with "|").
    let mut seen_node_ids: FxHashSet<grafeo_common::types::NodeId> = FxHashSet::default();
    let mut label_key_index: FxHashMap<ArcStr, usize> = FxHashMap::default();

    for label in &labels {
        let node_ids = store.nodes_by_label(label);
        for &nid in &node_ids {
            if !seen_node_ids.insert(nid) {
                continue; // already assigned via an earlier label
            }

            // Get the node to check its full label set.
            let Some(node) = store.get_node(nid) else {
                continue;
            };

            let label_key: ArcStr = if node.labels.len() <= 1 {
                ArcStr::from(label.as_str())
            } else {
                let mut sorted: Vec<&str> = node.labels.iter().map(|l| l.as_str()).collect();
                sorted.sort_unstable();
                ArcStr::from(sorted.join("|"))
            };

            // Find or create the label_data entry.
            let entry_idx = if let Some(&idx) = label_key_index.get(&label_key) {
                idx
            } else {
                let idx = label_data.len();
                label_key_index.insert(label_key.clone(), idx);
                label_data.push((label_key.clone(), Vec::new(), FxHashMap::default()));
                idx
            };

            let (_, ref mut node_ids_vec, ref mut props_map) = label_data[entry_idx];
            // reason: node offset within a table fits u32
            #[allow(clippy::cast_possible_truncation)]
            let offset = node_ids_vec.len() as u32;
            node_ids_vec.push(nid);
            id_map.insert(nid, (label_key, offset));

            // Collect properties.
            for (key, value) in node.properties.iter() {
                let col = props_map
                    .entry(key.clone())
                    .or_insert_with(|| vec![Value::Null; offset as usize]);
                // Pad with nulls if this key appeared for the first time.
                while col.len() < offset as usize {
                    col.push(Value::Null);
                }
                col.push(value.clone());
            }

            // Pad all existing columns that this node didn't have.
            let expected_len = offset as usize + 1;
            for col in props_map.values_mut() {
                while col.len() < expected_len {
                    col.push(Value::Null);
                }
            }
        }
    }

    // Step 2: Infer column types and build CompactStoreBuilder.
    let mut builder = CompactStoreBuilder::new();

    for (label_key, node_ids_for_label, props_map) in &label_data {
        let node_count = node_ids_for_label.len();
        builder = builder.node_table(label_key.as_str(), |t| {
            // Ensure row count is set even when there are no properties.
            t.record_len(node_count);
            for (key, values) in props_map {
                t.inferred_column(key, values);
            }
            t
        });
    }

    // Step 3: Collect all edges in a single pass, grouped by (edge_type, src_label, dst_label).
    // Key: (edge_type, src_label_key, dst_label_key) -> Vec<(src_offset, dst_offset)>
    type EdgeGroupKey = (ArcStr, ArcStr, ArcStr);
    let mut edge_groups: FxHashMap<EdgeGroupKey, Vec<(u32, u32)>> = FxHashMap::default();
    let mut edge_props_groups: FxHashMap<EdgeGroupKey, FxHashMap<PropertyKey, Vec<Value>>> =
        FxHashMap::default();

    // Iterate all nodes and their outgoing edges.
    for (_label_key, node_ids, _) in &label_data {
        for &nid in node_ids {
            let outgoing = store.edges_from(nid, crate::graph::Direction::Outgoing);
            for (_target_nid, edge_id) in outgoing {
                let Some(edge) = store.get_edge(edge_id) else {
                    continue;
                };

                let Some((src_label, src_offset)) = id_map.get(&edge.src) else {
                    continue;
                };
                let Some((dst_label, dst_offset)) = id_map.get(&edge.dst) else {
                    continue;
                };

                let group_key: EdgeGroupKey =
                    (edge.edge_type.clone(), src_label.clone(), dst_label.clone());

                let edges_vec = edge_groups.entry(group_key.clone()).or_default();
                let edge_idx = edges_vec.len();
                edges_vec.push((*src_offset, *dst_offset));

                // Collect edge properties.
                if !edge.properties.is_empty() {
                    let props = edge_props_groups.entry(group_key).or_default();
                    for (key, value) in edge.properties.iter() {
                        let col = props
                            .entry(key.clone())
                            .or_insert_with(|| vec![Value::Null; edge_idx]);
                        while col.len() < edge_idx {
                            col.push(Value::Null);
                        }
                        col.push(value.clone());
                    }
                    let expected_len = edge_idx + 1;
                    for col in props.values_mut() {
                        while col.len() < expected_len {
                            col.push(Value::Null);
                        }
                    }
                }
            }
        }
    }

    // Step 4: Add relationship tables to the builder.
    for ((edge_type, src_label, dst_label), edges) in &edge_groups {
        let edge_props =
            edge_props_groups.get(&(edge_type.clone(), src_label.clone(), dst_label.clone()));

        builder = builder.rel_table(
            edge_type.as_str(),
            src_label.as_str(),
            dst_label.as_str(),
            |r| {
                r.edges(edges.clone()).backward(true);

                // Add edge property columns.
                if let Some(props) = edge_props {
                    for (key, values) in props {
                        r.inferred_column(key, values);
                    }
                }

                r
            },
        );
    }

    builder.build()
}

/// Builds a [`CompactStore`] from any [`GraphStore`](crate::graph::GraphStore) with original ID preservation.
///
/// Same columnar conversion as [`from_graph_store`], but the resulting store
/// keeps a bidirectional mapping between the original `NodeId`/`EdgeId` values
/// and the internal compact positions. This enables layered storage where an
/// overlay store shares the same ID namespace.
///
/// # Errors
///
/// Same as [`from_graph_store`].
pub fn from_graph_store_preserving_ids(
    store: &dyn crate::graph::traits::GraphStore,
) -> Result<CompactStore, CompactStoreError> {
    let mut compact = from_graph_store(store)?;

    // ── Build node ID maps (replicate the label grouping logic) ────

    let labels = store.all_labels();
    if labels.is_empty() {
        compact.set_id_maps(Vec::new(), Vec::new());
        return Ok(compact);
    }

    let mut node_id_map: FxHashMap<grafeo_common::types::NodeId, (u16, u64)> = FxHashMap::default();
    let num_tables = compact.node_tables_by_id.len();
    let mut node_offset_to_id: Vec<Vec<grafeo_common::types::NodeId>> =
        vec![Vec::new(); num_tables];

    // Track per-label-key offset counters (same order as from_graph_store step 1).
    let mut seen: FxHashSet<grafeo_common::types::NodeId> = FxHashSet::default();
    let mut label_key_offsets: FxHashMap<ArcStr, u32> = FxHashMap::default();

    for label in &labels {
        let node_ids = store.nodes_by_label(label);
        for &nid in &node_ids {
            if !seen.insert(nid) {
                continue;
            }
            let Some(node) = store.get_node(nid) else {
                continue;
            };

            let label_key: ArcStr = if node.labels.len() <= 1 {
                ArcStr::from(label.as_str())
            } else {
                let mut sorted: Vec<&str> = node.labels.iter().map(|l| l.as_str()).collect();
                sorted.sort_unstable();
                ArcStr::from(sorted.join("|"))
            };

            let offset = label_key_offsets.entry(label_key.clone()).or_insert(0);
            let current_offset = *offset;
            *offset += 1;

            if let Some(&table_id) = compact.label_to_table_id.get(&label_key) {
                node_id_map.insert(nid, (table_id, u64::from(current_offset)));
                if let Some(rev) = node_offset_to_id.get_mut(table_id as usize) {
                    // Extend if needed (offsets should be sequential).
                    while rev.len() <= current_offset as usize {
                        rev.push(grafeo_common::types::NodeId::INVALID);
                    }
                    rev[current_offset as usize] = nid;
                }
            }
        }
    }

    // ── Build edge ID maps ─────────────────────────────────────────

    // Build (edge_type, src_table_id, dst_table_id) -> rel_table_id lookup.
    type RelKey = (ArcStr, u16, u16);
    let mut rel_key_to_id: FxHashMap<RelKey, u16> = FxHashMap::default();
    for (idx, rt) in compact.rel_tables_by_id.iter().enumerate() {
        let key = (rt.edge_type().clone(), rt.src_table_id(), rt.dst_table_id());
        let Ok(rel_id) = u16::try_from(idx) else {
            continue;
        };
        rel_key_to_id.insert(key, rel_id);
    }

    // Collect all edges grouped by (edge_type, src_table, dst_table), tracking
    // original EdgeId and (src_offset, dst_offset) for each.
    type EdgeGroupEntry = (grafeo_common::types::EdgeId, u32, u32); // (original_eid, src_off, dst_off)
    let mut edge_groups: FxHashMap<RelKey, Vec<EdgeGroupEntry>> = FxHashMap::default();

    let mut seen_edges: FxHashSet<grafeo_common::types::EdgeId> = FxHashSet::default();
    for &nid in node_id_map.keys() {
        let outgoing = store.edges_from(nid, crate::graph::Direction::Outgoing);
        for (_target_nid, edge_id) in outgoing {
            if !seen_edges.insert(edge_id) {
                continue;
            }
            let Some(edge) = store.get_edge(edge_id) else {
                continue;
            };
            let Some(&(src_tid, src_off)) = node_id_map.get(&edge.src) else {
                continue;
            };
            let Some(&(dst_tid, dst_off)) = node_id_map.get(&edge.dst) else {
                continue;
            };

            let key: RelKey = (edge.edge_type.clone(), src_tid, dst_tid);
            edge_groups.entry(key).or_default().push((
                edge_id,
                u32::try_from(src_off).unwrap_or(0),
                u32::try_from(dst_off).unwrap_or(0),
            ));
        }
    }

    // Match CompactStoreBuilder's stable source-only sort exactly. Sorting by
    // destination here would reorder equal-source edges relative to the CSR,
    // attaching each preserved EdgeId to another edge's native endpoints.
    let num_rel_tables = compact.rel_tables_by_id.len();
    let mut edge_offset_to_id: Vec<Vec<grafeo_common::types::EdgeId>> =
        vec![Vec::new(); num_rel_tables];

    for (key, mut entries) in edge_groups {
        let Some(&rel_table_id) = rel_key_to_id.get(&key) else {
            continue;
        };
        entries.sort_by_key(|&(_, src, _dst)| src);

        let rev = &mut edge_offset_to_id[rel_table_id as usize];
        for (csr_pos, (original_eid, _src, _dst)) in entries.iter().enumerate() {
            while rev.len() <= csr_pos {
                rev.push(grafeo_common::types::EdgeId::INVALID);
            }
            rev[csr_pos] = *original_eid;
        }
    }

    compact.set_id_maps(node_offset_to_id, edge_offset_to_id);
    Ok(compact)
}

/// Rows of `values` holding `Value::Null`, as a mask the table consults
/// before decoding a row; `None` when the column is dense.
///
/// A column body has no null representation — a padded row decodes as `0`,
/// `""`, or `false` — so without the mask a property a node never had would
/// read back as a value it never had.
fn null_mask(values: &[Value]) -> Option<BitVector> {
    values
        .iter()
        .any(|v| matches!(v, Value::Null))
        .then(|| values.iter().map(|v| matches!(v, Value::Null)).collect())
}

/// Encodes one property column from its row values with the codec
/// [`infer_type_from_values`] selects, plus the zone map that codec supports
/// (`None` for float and vector columns, which carry no zone statistics)
/// and the [`null_mask`] of the rows that hold no value.
///
/// This is the single definition of the `from_graph_store` type mapping;
/// the whole-store conversion and the incremental builder both encode
/// through it, so a store built either way holds byte-identical columns.
fn encode_inferred_column(values: &[Value]) -> (ColumnCodec, Option<ZoneMap>, Option<BitVector>) {
    let nulls = null_mask(values);
    let (codec, zone_map) = encode_inferred_codec(values);
    (codec, zone_map, nulls)
}

fn encode_inferred_codec(values: &[Value]) -> (ColumnCodec, Option<ZoneMap>) {
    match infer_type_from_values(values) {
        InferredType::BitPacked => {
            let u64_values: Vec<u64> = values
                .iter()
                .map(|v| match v {
                    // reason: ID encoding: i64 <-> u64 for bit-packed storage
                    #[allow(clippy::cast_sign_loss)]
                    Value::Int64(n) => *n as u64,
                    _ => 0,
                })
                .collect();
            let zone_map = compute_zone_map_u64(&u64_values);
            (
                ColumnCodec::BitPacked(BitPackedInts::pack(&u64_values)),
                Some(zone_map),
            )
        }
        InferredType::RawI64 => {
            let i64_values: Vec<i64> = values
                .iter()
                .map(|v| match v {
                    Value::Int64(n) => *n,
                    _ => 0,
                })
                .collect();
            let zone_map = compute_zone_map_i64(&i64_values);
            (ColumnCodec::raw_i64(i64_values), Some(zone_map))
        }
        InferredType::Float64 => {
            let f64_values: Vec<f64> = values
                .iter()
                .map(|v| match v {
                    Value::Float64(f) => *f,
                    Value::Int64(n) => *n as f64,
                    _ => 0.0,
                })
                .collect();
            (ColumnCodec::float64(f64_values), None)
        }
        InferredType::Float32Vector { dimensions } => {
            let mut flat: Vec<f32> = Vec::with_capacity(values.len() * dimensions as usize);
            for v in values {
                match v {
                    Value::Vector(vec) => flat.extend_from_slice(vec),
                    _ => flat.extend(std::iter::repeat_n(0.0f32, usize::from(dimensions))),
                }
            }
            (ColumnCodec::float32_vector(flat, dimensions), None)
        }
        InferredType::Bitmap => {
            let bool_values: Vec<bool> = values
                .iter()
                .map(|v| matches!(v, Value::Bool(true)))
                .collect();
            let zone_map = compute_zone_map_bool(&bool_values);
            (
                ColumnCodec::Bitmap(BitVector::from_bools(&bool_values)),
                Some(zone_map),
            )
        }
        InferredType::Dict => {
            let str_values: Vec<String> = values
                .iter()
                .map(super::dict_value::encode_dict_entry)
                .collect();
            let str_refs: Vec<&str> = str_values.iter().map(String::as_str).collect();
            let mut dict_builder = DictionaryBuilder::new();
            for s in &str_refs {
                dict_builder.add(s);
            }
            let dict = dict_builder.build();
            // A marked entry (Bytes payload or escaped string) stores an
            // encoded form; min/max bounds over encoded forms must not be
            // compared against raw query values, so those columns carry no
            // zone-map statistics.
            let has_marked = str_refs
                .iter()
                .any(|s| s.starts_with(super::dict_value::DICT_MARKER_PREFIX));
            let zone_map = if has_marked {
                ZoneMap::new()
            } else {
                compute_zone_map_strings(&str_refs)
            };
            (ColumnCodec::Dict(dict), Some(zone_map))
        }
    }
}

// ---------------------------------------------------------------------------
// Incremental construction from a row stream
// ---------------------------------------------------------------------------

/// Builds a [`CompactStore`] with preserved ids from a stream of nodes and
/// edges, without first materializing them in a live [`LpgStore`].
///
/// The result is what [`from_graph_store_preserving_ids`] would build from a
/// store holding the same rows: the same label grouping (multi-label nodes
/// under the sorted, `|`-joined label key), the same inferred column codecs
/// and zone maps, the same source-sorted CSR order, and the same
/// `NodeId`/`EdgeId` maps. Node tables and relationship tables are numbered
/// in order of first appearance in the stream, so identical input in
/// identical order builds an identical store, and the caller — not a hash
/// map walk — fixes that order.
///
/// Every node must be pushed before an edge refers to it; a relationship
/// table exists only once its first edge arrives.
///
/// Pushed property values are not held as [`Value`]s. Each column appends
/// its present values to a byte spool — in memory, or in the file given to
/// [`spooling_to`](Self::spooling_to) once a column's buffer fills — and
/// encoding decodes one column at a time. With a spool file, the resident
/// footprint while pushing is the row topology plus one small buffer per
/// column, and [`write_section`](Self::write_section) adds one column's
/// values and encoding on top of that, never the whole store.
///
/// [`LpgStore`]: crate::graph::lpg::LpgStore
#[derive(Default)]
pub struct IncrementalCompactStoreBuilder {
    node_tables: Vec<PendingNodeTable>,
    node_table_index: FxHashMap<ArcStr, usize>,
    /// Pushed node id -> (node table index, offset within the table).
    node_positions: FxHashMap<grafeo_common::types::NodeId, (usize, u32)>,
    rel_tables: Vec<PendingRelTable>,
    rel_table_index: FxHashMap<(ArcStr, usize, usize), usize>,
    edge_ids: FxHashSet<grafeo_common::types::EdgeId>,
    spool: ColumnSpool,
}

/// Row values for one node table, collected before encoding.
struct PendingNodeTable {
    label_key: ArcStr,
    node_ids: Vec<grafeo_common::types::NodeId>,
    columns: FxHashMap<PropertyKey, SpooledColumn>,
}

/// Row values for one relationship table, collected before encoding.
struct PendingRelTable {
    edge_type: ArcStr,
    src_table: usize,
    dst_table: usize,
    /// (edge id, source offset, destination offset) in push order.
    edges: Vec<(grafeo_common::types::EdgeId, u32, u32)>,
    columns: FxHashMap<PropertyKey, SpooledColumn>,
}

/// A column buffer is moved to the spool once it reaches this size.
const SPOOL_CHUNK_BYTES: usize = 32 * 1024;

/// The present values of one column in push order, as bincode
/// `(rows skipped since the previous value, value)` records. Rows without a
/// value are never written; decoding pads them with [`Value::Null`], which
/// is what the null-padded column form held for them.
#[derive(Default)]
struct SpooledColumn {
    chunks: Vec<SpoolChunk>,
    tail: Vec<u8>,
    /// Row of the last recorded value.
    last_row: Option<u32>,
}

/// A sealed run of whole records.
enum SpoolChunk {
    Resident(Vec<u8>),
    Spilled { offset: u64, len: usize },
}

/// Where full column buffers go. Without a file they stay resident as
/// encoded bytes.
#[derive(Default)]
struct ColumnSpool {
    file: Option<std::fs::File>,
    end: u64,
    /// A failed spool write leaves a column short, so the builder refuses
    /// everything after it instead of encoding a store missing values.
    failure: Option<String>,
}

fn spool_config() -> bincode::config::Configuration {
    bincode::config::standard()
}

fn spool_error(what: &str, error: impl std::fmt::Display) -> CompactStoreError {
    CompactStoreError::Spool(format!("{what}: {error}"))
}

impl ColumnSpool {
    fn check(&self) -> Result<(), CompactStoreError> {
        match &self.failure {
            Some(failure) => Err(CompactStoreError::Spool(failure.clone())),
            None => Ok(()),
        }
    }

    /// Records `value` at `row`. The first value pushed for a row wins, as
    /// it did in the null-padded column form.
    fn append(
        &mut self,
        column: &mut SpooledColumn,
        row: u32,
        value: &Value,
    ) -> Result<(), CompactStoreError> {
        let gap = match column.last_row {
            Some(last) if last >= row => return Ok(()),
            Some(last) => row - last - 1,
            None => row,
        };
        bincode::serde::encode_into_std_write((gap, value), &mut column.tail, spool_config())
            .map_err(|error| spool_error("encode a column value", error))?;
        column.last_row = Some(row);
        if column.tail.len() >= SPOOL_CHUNK_BYTES {
            self.seal(column)?;
        }
        Ok(())
    }

    fn seal(&mut self, column: &mut SpooledColumn) -> Result<(), CompactStoreError> {
        use std::io::Write;
        let Some(file) = self.file.as_mut() else {
            column
                .chunks
                .push(SpoolChunk::Resident(std::mem::take(&mut column.tail)));
            return Ok(());
        };
        if let Err(error) = file.write_all(&column.tail) {
            let error = spool_error("write the column spool", error);
            self.failure = Some(error.to_string());
            return Err(error);
        }
        let len = column.tail.len();
        column.chunks.push(SpoolChunk::Spilled {
            offset: self.end,
            len,
        });
        self.end += len as u64;
        column.tail.clear();
        Ok(())
    }

    /// Decodes a column into one value per row, `Value::Null` where the row
    /// has none.
    fn read(
        &mut self,
        column: SpooledColumn,
        rows: usize,
    ) -> Result<Vec<Value>, CompactStoreError> {
        use std::io::{Read, Seek, SeekFrom};
        let mut values = vec![Value::Null; rows];
        let mut next_row = 0usize;
        let mut spilled = Vec::new();
        let mut decode = |bytes: &[u8]| -> Result<(), CompactStoreError> {
            let mut position = 0;
            while position < bytes.len() {
                let ((gap, value), used): ((u32, Value), usize) =
                    bincode::serde::decode_from_slice(&bytes[position..], spool_config())
                        .map_err(|error| spool_error("decode a column value", error))?;
                position += used;
                let row = next_row + gap as usize;
                let slot = values.get_mut(row).ok_or_else(|| {
                    CompactStoreError::Spool(format!(
                        "a spooled value names row {row} of a {rows}-row table"
                    ))
                })?;
                *slot = value;
                next_row = row + 1;
            }
            Ok(())
        };
        for chunk in column.chunks {
            match chunk {
                SpoolChunk::Resident(bytes) => decode(&bytes)?,
                SpoolChunk::Spilled { offset, len } => {
                    let file = self.file.as_mut().ok_or_else(|| {
                        CompactStoreError::Spool("a spilled column has no spool file".into())
                    })?;
                    spilled.resize(len, 0);
                    file.seek(SeekFrom::Start(offset))
                        .and_then(|_| file.read_exact(&mut spilled))
                        .map_err(|error| spool_error("read the column spool", error))?;
                    decode(&spilled)?;
                }
            }
        }
        decode(&column.tail)?;
        Ok(values)
    }
}

/// Appends one row's properties to the columns of a table holding `row`
/// rows before it.
fn push_row_properties<'a>(
    spool: &mut ColumnSpool,
    columns: &mut FxHashMap<PropertyKey, SpooledColumn>,
    row: u32,
    properties: impl IntoIterator<Item = (&'a PropertyKey, &'a Value)>,
) -> Result<(), CompactStoreError> {
    for (key, value) in properties {
        let column = columns.entry(key.clone()).or_default();
        spool.append(column, row, value)?;
    }
    Ok(())
}

/// A table's columns in ascending key order: the section's column order.
fn columns_by_key(
    columns: FxHashMap<PropertyKey, SpooledColumn>,
) -> Vec<(PropertyKey, SpooledColumn)> {
    let mut sorted: Vec<_> = columns.into_iter().collect();
    sorted.sort_unstable_by(|(a, _), (b, _)| a.as_str().cmp(b.as_str()));
    sorted
}

/// A relationship table's edges in forward CSR order (source-sorted,
/// stable): the push rows in that order, and the `(src, dst)` pairs.
fn csr_order(edges: &[(grafeo_common::types::EdgeId, u32, u32)]) -> (Vec<usize>, Vec<(u32, u32)>) {
    let mut order: Vec<usize> = (0..edges.len()).collect();
    order.sort_by_key(|&row| edges[row].1);
    let pairs = order
        .iter()
        .map(|&row| (edges[row].1, edges[row].2))
        .collect();
    (order, pairs)
}

/// Moves push-order `values` into CSR `order`.
fn permute(mut values: Vec<Value>, order: &[usize]) -> Vec<Value> {
    order
        .iter()
        .map(|&row| std::mem::replace(&mut values[row], Value::Null))
        .collect()
}

fn table_id(index: usize, kind: &'static str) -> Result<u16, CompactStoreError> {
    u16::try_from(index)
        .ok()
        .filter(|&id| id <= MAX_TABLE_ID)
        .ok_or(CompactStoreError::TableCountOverflow {
            kind,
            count: index + 1,
            max: MAX_TABLE_ID,
        })
}

impl IncrementalCompactStoreBuilder {
    /// Creates an empty builder whose column spool stays in memory.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates an empty builder that moves full column buffers into
    /// `file`, which must be empty, readable, and writable. The caller owns
    /// the file's lifetime; an anonymous temporary file removes itself.
    #[must_use]
    pub fn spooling_to(file: std::fs::File) -> Self {
        Self {
            spool: ColumnSpool {
                file: Some(file),
                ..ColumnSpool::default()
            },
            ..Self::default()
        }
    }

    /// Nodes pushed so far.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.node_positions.len()
    }

    /// Edges pushed so far.
    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.edge_ids.len()
    }

    /// Adds a node with its labels and properties.
    ///
    /// # Errors
    ///
    /// - [`CompactStoreError::UnlabeledNode`] if `labels` is empty: a
    ///   compact table is keyed by label, so an unlabeled node has no table.
    /// - [`CompactStoreError::DuplicateNodeId`] if `id` was already pushed.
    /// - [`CompactStoreError::Spool`] if the column spool failed, now or on
    ///   an earlier push.
    pub fn push_node<'a>(
        &mut self,
        id: grafeo_common::types::NodeId,
        labels: impl IntoIterator<Item = &'a str>,
        properties: impl IntoIterator<Item = (&'a PropertyKey, &'a Value)>,
    ) -> Result<(), CompactStoreError> {
        self.spool.check()?;
        let mut sorted: Vec<&str> = labels.into_iter().collect();
        let label_key: ArcStr = match sorted.as_slice() {
            [] => return Err(CompactStoreError::UnlabeledNode(id.0)),
            [single] => ArcStr::from(*single),
            _ => {
                sorted.sort_unstable();
                sorted.dedup();
                ArcStr::from(sorted.join("|"))
            }
        };
        if self.node_positions.contains_key(&id) {
            return Err(CompactStoreError::DuplicateNodeId(id.0));
        }

        let table_index = match self.node_table_index.get(&label_key) {
            Some(&index) => index,
            None => {
                let index = self.node_tables.len();
                self.node_table_index.insert(label_key.clone(), index);
                self.node_tables.push(PendingNodeTable {
                    label_key,
                    node_ids: Vec::new(),
                    columns: FxHashMap::default(),
                });
                index
            }
        };
        let table = &mut self.node_tables[table_index];
        let row = table.node_ids.len();
        let offset = u32::try_from(row).map_err(|_| CompactStoreError::TableRowOverflow {
            table: table.label_key.to_string(),
        })?;
        table.node_ids.push(id);
        self.node_positions.insert(id, (table_index, offset));
        push_row_properties(&mut self.spool, &mut table.columns, offset, properties)
    }

    /// Adds an edge between two nodes pushed earlier.
    ///
    /// # Errors
    ///
    /// - [`CompactStoreError::UnknownEndpoint`] if `src` or `dst` was not
    ///   pushed as a node.
    /// - [`CompactStoreError::DuplicateEdgeId`] if `id` was already pushed.
    /// - [`CompactStoreError::Spool`] if the column spool failed, now or on
    ///   an earlier push.
    pub fn push_edge<'a>(
        &mut self,
        id: grafeo_common::types::EdgeId,
        edge_type: &str,
        src: grafeo_common::types::NodeId,
        dst: grafeo_common::types::NodeId,
        properties: impl IntoIterator<Item = (&'a PropertyKey, &'a Value)>,
    ) -> Result<(), CompactStoreError> {
        self.spool.check()?;
        let &(src_table, src_offset) =
            self.node_positions
                .get(&src)
                .ok_or(CompactStoreError::UnknownEndpoint {
                    edge: id.0,
                    node: src.0,
                })?;
        let &(dst_table, dst_offset) =
            self.node_positions
                .get(&dst)
                .ok_or(CompactStoreError::UnknownEndpoint {
                    edge: id.0,
                    node: dst.0,
                })?;
        if self.edge_ids.contains(&id) {
            return Err(CompactStoreError::DuplicateEdgeId(id.0));
        }

        let key = (ArcStr::from(edge_type), src_table, dst_table);
        let table_index = match self.rel_table_index.get(&key) {
            Some(&index) => index,
            None => {
                let index = self.rel_tables.len();
                self.rel_tables.push(PendingRelTable {
                    edge_type: key.0.clone(),
                    src_table,
                    dst_table,
                    edges: Vec::new(),
                    columns: FxHashMap::default(),
                });
                self.rel_table_index.insert(key, index);
                index
            }
        };
        let table = &mut self.rel_tables[table_index];
        let row =
            u32::try_from(table.edges.len()).map_err(|_| CompactStoreError::TableRowOverflow {
                table: table.edge_type.to_string(),
            })?;
        self.edge_ids.insert(id);
        table.edges.push((id, src_offset, dst_offset));
        push_row_properties(&mut self.spool, &mut table.columns, row, properties)
    }

    /// Encodes the collected rows into an immutable [`CompactStore`] whose
    /// id maps name every pushed node and edge.
    ///
    /// Columns are decoded and encoded one at a time, so the transient on
    /// top of the finished store is one column's values.
    ///
    /// # Errors
    ///
    /// Propagates [`CompactStoreBuilder::build`] errors (e.g. more than
    /// 32,768 node tables or relationship tables) and spool failures.
    pub fn finish(mut self) -> Result<CompactStore, CompactStoreError> {
        self.spool.check()?;
        let mut builder = CompactStoreBuilder::new();
        let node_tables = std::mem::take(&mut self.node_tables);
        let mut labels = Vec::with_capacity(node_tables.len());
        let mut node_ids = Vec::with_capacity(node_tables.len());
        for table in node_tables {
            let rows = table.node_ids.len();
            let mut t = NodeTableBuilder::new(table.label_key.clone());
            t.record_len(rows);
            for (key, column) in columns_by_key(table.columns) {
                let values = self.spool.read(column, rows)?;
                t.inferred_column(&key, &values);
            }
            builder.node_table_builders.push(t);
            labels.push(table.label_key);
            node_ids.push(table.node_ids);
        }

        // Relationship tables take the builder's sequential ids in push
        // order; the source-sorted (stable) order is the forward CSR order
        // `CompactStoreBuilder::build` produces, so each edge id maps to the
        // position its endpoints occupy.
        let rel_tables = std::mem::take(&mut self.rel_tables);
        let mut edge_offset_to_id: Vec<Vec<grafeo_common::types::EdgeId>> =
            Vec::with_capacity(rel_tables.len());
        for (index, table) in rel_tables.into_iter().enumerate() {
            table_id(index, "relationship")?;
            let (order, pairs) = csr_order(&table.edges);
            let ids: Vec<grafeo_common::types::EdgeId> =
                order.iter().map(|&row| table.edges[row].0).collect();
            edge_offset_to_id.push(ids);
            let mut r = RelTableBuilder::new(
                table.edge_type,
                labels[table.src_table].clone(),
                labels[table.dst_table].clone(),
            );
            r.edges(pairs).backward(true);
            for (key, column) in columns_by_key(table.columns) {
                let values = permute(self.spool.read(column, table.edges.len())?, &order);
                r.inferred_column(&key, &values);
            }
            builder.rel_table_builders.push(r);
        }

        let mut compact = builder.build()?;
        compact.set_id_maps(node_ids, edge_offset_to_id);
        Ok(compact)
    }

    /// Streams the rows as a CompactStore section into `sink`: byte for
    /// byte what serializing [`finish`](Self::finish)'s store writes once
    /// `indexed_properties` are enabled on it, at the current section
    /// version, without building that store.
    ///
    /// Each column is decoded, encoded, written, and dropped before the
    /// next, and each relationship table's adjacency is built only while
    /// that table is written, so the resident peak on top of the pushed
    /// topology is the largest single column or adjacency.
    ///
    /// # Errors
    ///
    /// `Error::Internal` for the [`finish`](Self::finish) refusals and
    /// spool failures, `Error::Io` when `sink` rejects a write.
    pub fn write_section(
        mut self,
        sink: &mut dyn std::io::Write,
        indexed_properties: &[PropertyKey],
    ) -> grafeo_common::utils::error::Result<()> {
        let internal = |error: CompactStoreError| {
            grafeo_common::utils::error::Error::Internal(format!(
                "incremental compact section: {error}"
            ))
        };
        self.spool.check().map_err(internal)?;
        table_id(self.node_tables.len().saturating_sub(1), "node").map_err(internal)?;
        table_id(self.rel_tables.len().saturating_sub(1), "relationship").map_err(internal)?;

        let mut writer = super::section::SectionWriter::begin(sink, true)?;
        let node_tables = std::mem::take(&mut self.node_tables);
        let mut node_ids = Vec::with_capacity(node_tables.len());
        writer.count(node_tables.len());
        for table in node_tables {
            let rows = table.node_ids.len();
            let columns = columns_by_key(table.columns);
            writer.node_table(table.label_key.as_str(), rows, columns.len())?;
            for (key, column) in columns {
                let values = self.spool.read(column, rows).map_err(internal)?;
                let (codec, zone_map, nulls) = encode_inferred_column(&values);
                drop(values);
                let block_zone_maps = super::zone_map::compute_block_zone_maps(&codec);
                let order = indexed_properties
                    .contains(&key)
                    .then(|| super::value_order::build_row_order(&codec, nulls.as_ref()))
                    .flatten();
                writer.node_column(
                    &key,
                    zone_map.as_ref(),
                    nulls.as_ref(),
                    &codec,
                    Some(&block_zone_maps),
                    order.as_ref(),
                )?;
            }
            node_ids.push(table.node_ids);
        }

        let rel_tables = std::mem::take(&mut self.rel_tables);
        let mut edge_entries: Vec<(u64, u16, u64)> = Vec::with_capacity(self.edge_ids.len());
        let mut edge_positions: Vec<Vec<u64>> = Vec::with_capacity(rel_tables.len());
        writer.count(rel_tables.len());
        for (index, table) in rel_tables.into_iter().enumerate() {
            let rel_table_id = table_id(index, "relationship").map_err(internal)?;
            let (order, pairs) = csr_order(&table.edges);
            let mut positions = Vec::with_capacity(order.len());
            for (position, &row) in order.iter().enumerate() {
                let id = table.edges[row].0.as_u64();
                edge_entries.push((id, rel_table_id, position as u64));
                positions.push(id);
            }
            edge_positions.push(positions);
            let (fwd, bwd) = rel_adjacency(
                &pairs,
                node_ids[table.src_table].len(),
                node_ids[table.dst_table].len(),
                true,
            )
            .map_err(internal)?;
            drop(pairs);
            let columns = columns_by_key(table.columns);
            writer.rel_table(
                table.edge_type.as_str(),
                table_id(table.src_table, "node").map_err(internal)?,
                table_id(table.dst_table, "node").map_err(internal)?,
                &fwd,
                bwd.as_ref(),
                columns.len(),
            )?;
            drop((fwd, bwd));
            for (key, column) in columns {
                let values = permute(
                    self.spool
                        .read(column, table.edges.len())
                        .map_err(internal)?,
                    &order,
                );
                let (codec, _zone_map, nulls) = encode_inferred_column(&values);
                drop(values);
                writer.rel_column(&key, nulls.as_ref(), &codec)?;
            }
        }

        let mut node_entries: Vec<(u64, u16, u64)> = Vec::with_capacity(self.node_positions.len());
        for (index, ids) in node_ids.iter().enumerate() {
            let table_id = table_id(index, "node").map_err(internal)?;
            node_entries.extend(
                ids.iter()
                    .enumerate()
                    .map(|(offset, id)| (id.as_u64(), table_id, offset as u64)),
            );
        }
        node_entries.sort_unstable_by_key(|&(id, _, _)| id);
        writer.id_records(&node_entries)?;
        drop(node_entries);
        writer.count(node_ids.len());
        for ids in &node_ids {
            let ids: Vec<u64> = ids.iter().map(|id| id.as_u64()).collect();
            writer.reverse_ids(&ids)?;
        }
        drop(node_ids);
        edge_entries.sort_unstable_by_key(|&(id, _, _)| id);
        writer.id_records(&edge_entries)?;
        drop(edge_entries);
        writer.count(edge_positions.len());
        for ids in &edge_positions {
            writer.reverse_ids(ids)?;
        }
        writer.finish()
    }
}

/// Infers the columnar encoding type from a slice of [`Value`]s.
///
/// Rules:
/// - If all non-null values are `Int64` with value >= 0, returns `BitPacked`.
/// - If all non-null values are `Bool`, returns `Bitmap`.
/// - Otherwise returns `Dict` (string fallback).
fn infer_type_from_values(values: &[Value]) -> InferredType {
    let mut saw_unsigned_int = false; // Value::Int64 with n >= 0
    let mut saw_signed_int = false; // Value::Int64 with n < 0
    let mut saw_float = false;
    let mut saw_bool = false;
    let mut saw_vector = false;
    let mut saw_other = false;
    let mut vector_dims: Option<u16> = None;

    for v in values {
        match v {
            Value::Null => {} // skip nulls
            Value::Int64(n) if *n >= 0 => saw_unsigned_int = true,
            Value::Int64(_) => saw_signed_int = true,
            Value::Float64(_) => saw_float = true,
            Value::Bool(_) => saw_bool = true,
            Value::Vector(vec) => {
                saw_vector = true;
                let Ok(dims) = u16::try_from(vec.len()) else {
                    saw_other = true; // too many dimensions for columnar storage
                    continue;
                };
                if let Some(prev) = vector_dims {
                    if prev != dims {
                        saw_other = true; // mixed dimensions → fallback
                    }
                } else {
                    vector_dims = Some(dims);
                }
            }
            _ => saw_other = true,
        }
    }

    let saw_any_int = saw_unsigned_int || saw_signed_int;

    // Vectors are exclusive; mixed with other types falls back to Dict.
    // Zero-dimension vectors cannot be round-tripped through the Float32Vector
    // codec (stride=0 means no row can be decoded), so those fall back to Dict
    // as well.
    if saw_vector
        && !saw_other
        && !saw_any_int
        && !saw_float
        && !saw_bool
        && let Some(dims) = vector_dims
        && dims > 0
    {
        return InferredType::Float32Vector { dimensions: dims };
    }

    // Mixed Int64+Float64 coalesces to Float64.
    // Vectors mixed with any other type fall back to Dict.
    if saw_other || saw_vector || (saw_any_int && saw_bool) || (saw_float && saw_bool) {
        InferredType::Dict
    } else if saw_float {
        InferredType::Float64
    } else if saw_signed_int {
        // Any negative value routes the whole column to RawI64, which uses
        // native i64 ordering. Non-negative-only columns still use the
        // more compact BitPacked encoding.
        InferredType::RawI64
    } else if saw_unsigned_int {
        InferredType::BitPacked
    } else if saw_bool {
        InferredType::Bitmap
    } else {
        // All nulls: default to Dict.
        InferredType::Dict
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::traits::GraphStore;

    #[test]
    fn test_builder_basic() {
        let store = CompactStoreBuilder::new()
            .node_table("Person", |t| {
                t.column_bitpacked("age", &[25, 30, 35, 40, 45], 6)
                    .column_dict("name", &["Alix", "Gus", "Vincent", "Jules", "Mia"])
            })
            .build()
            .unwrap();

        // Verify we can query it.
        let ids = store.nodes_by_label("Person");
        assert_eq!(ids.len(), 5);
    }

    #[test]
    fn test_builder_with_edges() {
        let store = CompactStoreBuilder::new()
            .node_table("A", |t| t.column_bitpacked("val", &[1, 2, 3], 4))
            .node_table("B", |t| t.column_bitpacked("val", &[10, 20], 8))
            .rel_table("LINKS", "A", "B", |r| {
                r.edges([(0, 0), (0, 1), (1, 0), (2, 1)]).backward(true)
            })
            .build()
            .unwrap();

        let a_ids = store.nodes_by_label("A");
        assert_eq!(a_ids.len(), 3);
        let b_ids = store.nodes_by_label("B");
        assert_eq!(b_ids.len(), 2);
    }

    #[test]
    fn test_builder_label_not_found() {
        let result = CompactStoreBuilder::new()
            .node_table("A", |t| t.column_bitpacked("val", &[1], 4))
            .rel_table("LINKS", "A", "B", |r| {
                // "B" doesn't exist
                r.edges([(0, 0)])
            })
            .build();

        assert!(result.is_err());
    }

    #[test]
    fn test_from_graph_store_round_trip() {
        // Build a CompactStore via the builder, then convert it back via
        // from_graph_store and verify the data survives the round-trip.
        let original = CompactStoreBuilder::new()
            .node_table("Person", |t| {
                t.column_bitpacked("age", &[25, 30, 35], 6)
                    .column_dict("name", &["Alix", "Gus", "Vincent"])
                    .column_bitmap("active", &[true, false, true])
            })
            .node_table("City", |t| t.column_dict("name", &["Amsterdam", "Berlin"]))
            .rel_table("LIVES_IN", "Person", "City", |r| {
                r.edges([(0, 0), (1, 1), (2, 0)]).backward(true)
            })
            .build()
            .unwrap();

        // Round-trip through from_graph_store.
        let converted = from_graph_store(&original).unwrap();

        // Verify node counts.
        assert_eq!(converted.nodes_by_label("Person").len(), 3);
        assert_eq!(converted.nodes_by_label("City").len(), 2);

        // Verify properties survived.
        let person_ids = converted.nodes_by_label("Person");
        let mut ages: Vec<i64> = person_ids
            .iter()
            .filter_map(|&id| {
                converted
                    .get_node_property(id, &PropertyKey::new("age"))
                    .and_then(|v| v.as_int64())
            })
            .collect();
        ages.sort_unstable();
        assert_eq!(ages, vec![25, 30, 35]);

        // Verify edges survived.
        let city_ids = converted.nodes_by_label("City");
        let mut total_edges = 0;
        for &pid in &person_ids {
            let edges = converted.edges_from(pid, crate::graph::Direction::Outgoing);
            total_edges += edges.len();
        }
        assert_eq!(total_edges, 3);

        // Verify backward edges (incoming to cities).
        for &cid in &city_ids {
            let incoming = converted.edges_from(cid, crate::graph::Direction::Incoming);
            assert!(!incoming.is_empty());
        }
    }

    #[test]
    fn test_from_graph_store_empty() {
        let empty = CompactStoreBuilder::new().build().unwrap();
        let converted = from_graph_store(&empty).unwrap();
        assert_eq!(converted.nodes_by_label("Anything").len(), 0);
    }

    #[test]
    fn test_from_graph_store_with_lpg_store() {
        use crate::graph::lpg::LpgStore;

        let store = LpgStore::new().unwrap();

        // Insert nodes.
        let alix_id = store.create_node(&["Person"]);
        store.set_node_property(alix_id, "name", Value::from("Alix"));
        store.set_node_property(alix_id, "age", Value::Int64(30));

        let gus_id = store.create_node(&["Person"]);
        store.set_node_property(gus_id, "name", Value::from("Gus"));
        store.set_node_property(gus_id, "age", Value::Int64(25));

        let amsterdam_id = store.create_node(&["City"]);
        store.set_node_property(amsterdam_id, "name", Value::from("Amsterdam"));

        // Insert edges.
        store.create_edge(alix_id, amsterdam_id, "LIVES_IN");
        store.create_edge(gus_id, amsterdam_id, "LIVES_IN");

        // Convert.
        let compact = from_graph_store(&store).unwrap();

        // Verify.
        assert_eq!(compact.nodes_by_label("Person").len(), 2);
        assert_eq!(compact.nodes_by_label("City").len(), 1);

        // Check that properties are readable.
        let person_ids = compact.nodes_by_label("Person");
        let mut names: Vec<String> = person_ids
            .iter()
            .filter_map(|&id| {
                compact
                    .get_node_property(id, &PropertyKey::new("name"))
                    .and_then(|v| v.as_str().map(|s| s.to_string()))
            })
            .collect();
        names.sort();
        assert_eq!(names, vec!["Alix", "Gus"]);

        // Check edges: both persons should have outgoing edges.
        let mut total_outgoing = 0;
        for &pid in &person_ids {
            let edges = compact.edges_from(pid, crate::graph::Direction::Outgoing);
            total_outgoing += edges.len();
        }
        assert_eq!(total_outgoing, 2);

        // Check incoming edges on the city.
        let city_ids = compact.nodes_by_label("City");
        assert_eq!(city_ids.len(), 1);
        let incoming = compact.edges_from(city_ids[0], crate::graph::Direction::Incoming);
        assert_eq!(incoming.len(), 2);
    }

    #[test]
    fn test_from_graph_store_edge_properties() {
        use crate::graph::lpg::LpgStore;

        let store = LpgStore::new().unwrap();

        let alix = store.create_node(&["Person"]);
        store.set_node_property(alix, "name", Value::from("Alix"));

        let gus = store.create_node(&["Person"]);
        store.set_node_property(gus, "name", Value::from("Gus"));

        // Edge with int property (BitPacked path).
        let e1 = store.create_edge(alix, gus, "KNOWS");
        store.set_edge_property(e1, "since", Value::Int64(2020));

        // Edge with string property (Dict path).
        let e2 = store.create_edge(gus, alix, "KNOWS");
        store.set_edge_property(e2, "since", Value::Int64(2021));

        let compact = from_graph_store(&store).unwrap();

        // Verify edge count.
        let person_ids = compact.nodes_by_label("Person");
        let mut total_edges = 0;
        for &pid in &person_ids {
            total_edges += compact
                .edges_from(pid, crate::graph::Direction::Outgoing)
                .len();
        }
        assert_eq!(total_edges, 2);

        // Verify edge properties survived.
        for &pid in &person_ids {
            let edges = compact.edges_from(pid, crate::graph::Direction::Outgoing);
            for (_target, eid) in &edges {
                let edge = compact.get_edge(*eid).unwrap();
                let since = edge.properties.get(&PropertyKey::new("since")).unwrap();
                match since {
                    Value::Int64(v) => assert!(*v == 2020 || *v == 2021),
                    _ => panic!("expected Int64 for 'since', got {since:?}"),
                }
            }
        }
    }

    #[test]
    fn test_from_graph_store_edge_bool_properties() {
        use crate::graph::lpg::LpgStore;

        let store = LpgStore::new().unwrap();

        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);

        let e = store.create_edge(a, b, "LINK");
        store.set_edge_property(e, "active", Value::Bool(true));

        let compact = from_graph_store(&store).unwrap();

        let ids = compact.nodes_by_label("Node");
        let edges = compact.edges_from(ids[0], crate::graph::Direction::Outgoing);
        assert_eq!(edges.len(), 1);

        let edge = compact.get_edge(edges[0].1).unwrap();
        assert_eq!(
            edge.properties.get(&PropertyKey::new("active")),
            Some(&Value::Bool(true))
        );
    }

    #[test]
    fn test_from_graph_store_edge_string_properties() {
        use crate::graph::lpg::LpgStore;

        let store = LpgStore::new().unwrap();

        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);

        let e = store.create_edge(a, b, "LINK");
        store.set_edge_property(e, "label", Value::from("primary"));

        let compact = from_graph_store(&store).unwrap();

        let ids = compact.nodes_by_label("Node");
        let edges = compact.edges_from(ids[0], crate::graph::Direction::Outgoing);
        let edge = compact.get_edge(edges[0].1).unwrap();
        assert_eq!(
            edge.properties.get(&PropertyKey::new("label")),
            Some(&Value::String(ArcStr::from("primary")))
        );
    }

    #[test]
    fn test_from_graph_store_negative_int_preserves_int64_type() {
        use crate::graph::lpg::LpgStore;

        let store = LpgStore::new().unwrap();

        let a = store.create_node(&["Item"]);
        store.set_node_property(a, "temp", Value::Int64(-10));

        let b = store.create_node(&["Item"]);
        store.set_node_property(b, "temp", Value::Int64(5));

        let compact = from_graph_store(&store).unwrap();

        // Signed Int64 columns round-trip as Value::Int64 via the RawI64 codec.
        // Prior behaviour (<=0.5.40) stringified them into a Dict column,
        // breaking WHERE matches and promoting sum() to Float64.
        let ids = compact.nodes_by_label("Item");
        assert_eq!(ids.len(), 2);
        let mut temps: Vec<i64> = ids
            .iter()
            .filter_map(
                |&id| match compact.get_node_property(id, &PropertyKey::new("temp")) {
                    Some(Value::Int64(n)) => Some(n),
                    _ => None,
                },
            )
            .collect();
        temps.sort_unstable();
        assert_eq!(temps, vec![-10, 5]);
    }

    #[test]
    fn test_from_graph_store_float64_column() {
        use crate::graph::lpg::LpgStore;

        let store = LpgStore::new().unwrap();

        let a = store.create_node(&["Sensor"]);
        store.set_node_property(a, "reading", Value::Float64(98.6));

        let compact = from_graph_store(&store).unwrap();

        let ids = compact.nodes_by_label("Sensor");
        assert_eq!(ids.len(), 1);

        // Float64 values are stored natively.
        let val = compact
            .get_node_property(ids[0], &PropertyKey::new("reading"))
            .unwrap();
        assert_eq!(val, Value::Float64(98.6));
    }

    #[test]
    fn test_from_graph_store_mixed_types_fall_back_to_dict() {
        use crate::graph::lpg::LpgStore;

        let store = LpgStore::new().unwrap();

        // Same property key with different types across nodes.
        let a = store.create_node(&["Thing"]);
        store.set_node_property(a, "value", Value::Int64(42));

        let b = store.create_node(&["Thing"]);
        store.set_node_property(b, "value", Value::Bool(true));

        let compact = from_graph_store(&store).unwrap();

        // Mixed Int64 + Bool should fall back to Dict.
        let ids = compact.nodes_by_label("Thing");
        assert_eq!(ids.len(), 2);

        for &id in &ids {
            let val = compact
                .get_node_property(id, &PropertyKey::new("value"))
                .unwrap();
            // All values should be strings (Dict encoding).
            assert!(
                matches!(val, Value::String(_)),
                "expected String (Dict fallback), got {val:?}"
            );
        }
    }

    #[test]
    fn test_from_graph_store_sparse_properties() {
        use crate::graph::lpg::LpgStore;

        let store = LpgStore::new().unwrap();

        // Node A has both properties.
        let a = store.create_node(&["Item"]);
        store.set_node_property(a, "name", Value::from("alpha"));
        store.set_node_property(a, "score", Value::Int64(10));

        // Node B has only 'name', no 'score'.
        let b = store.create_node(&["Item"]);
        store.set_node_property(b, "name", Value::from("beta"));

        // Node C has only 'score', no 'name'.
        let c = store.create_node(&["Item"]);
        store.set_node_property(c, "score", Value::Int64(20));

        let compact = from_graph_store(&store).unwrap();

        let ids = compact.nodes_by_label("Item");
        assert_eq!(ids.len(), 3);

        // Every node has exactly the properties it was given: a property a
        // node never had is absent, not the column's padding value.
        let name = PropertyKey::new("name");
        let score = PropertyKey::new("score");
        let mut names = Vec::new();
        let mut scores = Vec::new();
        for &id in &ids {
            let node = compact.get_node(id).unwrap();
            names.push(node.properties.get(&name).cloned());
            scores.push(node.properties.get(&score).cloned());
            assert_eq!(compact.get_node_property(id, &name), names[names.len() - 1]);
            assert_eq!(
                compact.get_node_property(id, &score),
                scores[scores.len() - 1]
            );
        }
        names.sort_by_key(|v| v.is_none());
        scores.sort_by_key(|v| v.is_none());
        assert!(names.contains(&Some(Value::from("alpha"))));
        assert!(names.contains(&Some(Value::from("beta"))));
        assert_eq!(names[2], None, "node c never had a name");
        assert!(scores.contains(&Some(Value::Int64(10))));
        assert!(scores.contains(&Some(Value::Int64(20))));
        assert_eq!(scores[2], None, "node b never had a score");
        // Neither padding value is findable as a real value.
        assert!(
            compact
                .find_nodes_by_property("name", &Value::from(""))
                .is_empty()
        );
        assert!(
            compact
                .find_nodes_by_property("score", &Value::Int64(0))
                .is_empty()
        );
    }

    #[test]
    fn test_from_graph_store_multi_label_nodes() {
        use crate::graph::lpg::LpgStore;

        let store = LpgStore::new().unwrap();

        let a = store.create_node(&["Person", "Actor"]);
        store.set_node_property(a, "name", Value::from("Vincent"));

        let b = store.create_node(&["Person"]);
        store.set_node_property(b, "name", Value::from("Jules"));

        let compact = from_graph_store(&store).unwrap();

        // Single-label node goes to "Person" table.
        let person_ids = compact.nodes_by_label("Person");
        assert_eq!(person_ids.len(), 1);

        // Multi-label node goes to "Actor|Person" compound table.
        let compound_ids = compact.nodes_by_label("Actor|Person");
        assert_eq!(compound_ids.len(), 1);

        // Verify the multi-label node's property survived.
        let val = compact
            .get_node_property(compound_ids[0], &PropertyKey::new("name"))
            .unwrap();
        assert_eq!(val, Value::String(ArcStr::from("Vincent")));
    }

    #[test]
    fn test_from_graph_store_all_null_column() {
        use crate::graph::lpg::LpgStore;

        let store = LpgStore::new().unwrap();

        // Two nodes with different property keys, creating gaps.
        let a = store.create_node(&["Item"]);
        store.set_node_property(a, "x", Value::Int64(1));

        let b = store.create_node(&["Item"]);
        store.set_node_property(b, "y", Value::Int64(2));

        let compact = from_graph_store(&store).unwrap();

        let ids = compact.nodes_by_label("Item");
        assert_eq!(ids.len(), 2);

        // Node a has 'x' and no 'y'; node b has 'y' and no 'x'. Each reads
        // back with exactly one property.
        let x = PropertyKey::new("x");
        let y = PropertyKey::new("y");
        let mut seen = Vec::new();
        for &id in &ids {
            let node = compact.get_node(id).unwrap();
            assert_eq!(node.properties.len(), 1, "{:?}", node.properties);
            seen.push((
                compact.get_node_property(id, &x),
                compact.get_node_property(id, &y),
            ));
        }
        seen.sort_by_key(|(x, _)| x.is_none());
        assert_eq!(
            seen,
            vec![(Some(Value::Int64(1)), None), (None, Some(Value::Int64(2))),]
        );
    }

    #[test]
    fn test_infer_type_all_nulls() {
        assert_eq!(
            infer_type_from_values(&[Value::Null, Value::Null]),
            InferredType::Dict
        );
    }

    #[test]
    fn test_infer_type_int_only() {
        assert_eq!(
            infer_type_from_values(&[Value::Int64(5), Value::Int64(10)]),
            InferredType::BitPacked
        );
    }

    #[test]
    fn test_infer_type_bool_only() {
        assert_eq!(
            infer_type_from_values(&[Value::Bool(true), Value::Bool(false)]),
            InferredType::Bitmap
        );
    }

    #[test]
    fn test_infer_type_mixed_int_bool() {
        assert_eq!(
            infer_type_from_values(&[Value::Int64(1), Value::Bool(true)]),
            InferredType::Dict
        );
    }

    #[test]
    fn test_infer_type_negative_int() {
        // Any negative Int64 routes the whole column to RawI64 (prior
        // behaviour fell back to Dict, which broke type round-trip and
        // ordered operations).
        assert_eq!(
            infer_type_from_values(&[Value::Int64(-5), Value::Int64(10)]),
            InferredType::RawI64
        );
        assert_eq!(
            infer_type_from_values(&[Value::Int64(-5)]),
            InferredType::RawI64
        );
        // Non-negative-only columns still use BitPacked for compression.
        assert_eq!(
            infer_type_from_values(&[Value::Int64(5), Value::Int64(10)]),
            InferredType::BitPacked
        );
    }

    #[test]
    fn test_infer_type_float() {
        assert_eq!(
            infer_type_from_values(&[Value::Float64(1.5)]),
            InferredType::Float64
        );
    }

    #[test]
    fn test_infer_type_mixed_int_float_coalesces_to_float() {
        assert_eq!(
            infer_type_from_values(&[Value::Int64(1), Value::Float64(2.5)]),
            InferredType::Float64
        );
    }

    #[test]
    fn test_infer_type_int_with_nulls() {
        assert_eq!(
            infer_type_from_values(&[Value::Int64(5), Value::Null, Value::Int64(10)]),
            InferredType::BitPacked
        );
    }

    /// Same edge type spanning multiple label pairs — normal in LPGs.
    /// Regression test for <https://github.com/GrafeoDB/grafeo/issues/221>.
    #[test]
    fn test_from_graph_store_multi_label_edge_type() {
        use crate::graph::lpg::LpgStore;

        let store = LpgStore::new().unwrap();

        // Three node types
        let m1 = store.create_node(&["Method"]);
        store.set_node_property(m1, "name", Value::from("foo"));
        let m2 = store.create_node(&["Method"]);
        store.set_node_property(m2, "name", Value::from("bar"));
        let c1 = store.create_node(&["Class"]);
        store.set_node_property(c1, "name", Value::from("MyClass"));
        let i1 = store.create_node(&["Interface"]);
        store.set_node_property(i1, "name", Value::from("MyInterface"));

        // CALLS edges between different label pairs
        store.create_edge(m1, m2, "CALLS"); // Method -> Method
        store.create_edge(c1, m1, "CALLS"); // Class -> Method
        // USES_TYPE edges between different label pairs
        store.create_edge(m1, c1, "USES_TYPE"); // Method -> Class
        store.create_edge(m1, i1, "USES_TYPE"); // Method -> Interface

        // This should not panic — same edge type across multiple label pairs is valid.
        let compact = from_graph_store(&store).unwrap();

        // Verify all nodes survived
        assert_eq!(compact.nodes_by_label("Method").len(), 2);
        assert_eq!(compact.nodes_by_label("Class").len(), 1);
        assert_eq!(compact.nodes_by_label("Interface").len(), 1);

        // Verify edges survived — check via rel_tables_for_type
        let calls_tables = compact.rel_tables_for_type("CALLS");
        let uses_tables = compact.rel_tables_for_type("USES_TYPE");

        // CALLS spans 2 label pairs: Method→Method, Class→Method
        assert_eq!(
            calls_tables.len(),
            2,
            "CALLS should have 2 rel tables (different label pairs)"
        );
        // USES_TYPE spans 2 label pairs: Method→Class, Method→Interface
        assert_eq!(
            uses_tables.len(),
            2,
            "USES_TYPE should have 2 rel tables (different label pairs)"
        );

        // Total edges across all CALLS tables
        let total_calls: usize = calls_tables.iter().map(|rt| rt.num_edges()).sum();
        assert_eq!(total_calls, 2, "Should have 2 CALLS edges total");
        // Total edges across all USES_TYPE tables
        let total_uses: usize = uses_tables.iter().map(|rt| rt.num_edges()).sum();
        assert_eq!(total_uses, 2, "Should have 2 USES_TYPE edges total");

        // Verify all_edge_types returns deduplicated type names
        use crate::graph::traits::GraphStore;
        let mut edge_types = compact.all_edge_types();
        edge_types.sort();
        assert_eq!(
            edge_types,
            vec!["CALLS", "USES_TYPE"],
            "all_edge_types should return each type once, not per rel table"
        );

        // Verify estimate_avg_degree deduplicates shared source labels.
        // USES_TYPE has Method→Class and Method→Interface — Method appears as source in both
        // rel tables but should only be counted once in the denominator.
        // 2 edges / 2 source nodes (Method) = 1.0 (not 2 edges / 4 = 0.5 if double-counted)
        let avg_out = compact.estimate_avg_degree("USES_TYPE", true);
        assert!(avg_out > 0.0, "USES_TYPE outgoing degree should be > 0");
        assert!(
            (avg_out - 1.0).abs() < f64::EPSILON,
            "USES_TYPE avg outgoing degree should be 1.0 (2 edges / 2 Method nodes), got {avg_out}"
        );

        // Verify unknown edge type returns 0
        let unknown = compact.estimate_avg_degree("NONEXISTENT", true);
        assert!(
            (unknown - 0.0).abs() < f64::EPSILON,
            "Unknown edge type should return 0.0 avg degree"
        );
    }

    // -------------------------------------------------------------------
    // Zone map helper tests
    // -------------------------------------------------------------------

    #[test]
    fn test_zone_map_u64_values_exceeding_i64_max() {
        // Values that exceed i64::MAX should produce a zone map without
        // min/max bounds (conservative, no pruning).
        let values = vec![0u64, i64::MAX as u64 + 1, u64::MAX];
        let zm = compute_zone_map_u64(&values);
        assert!(
            zm.min.is_none(),
            "min should be None when values overflow i64"
        );
        assert!(
            zm.max.is_none(),
            "max should be None when values overflow i64"
        );
        assert_eq!(zm.row_count, 3);
    }

    #[test]
    fn test_zone_map_u64_within_i64_range() {
        let values = vec![10u64, 20, 30];
        let zm = compute_zone_map_u64(&values);
        assert_eq!(zm.min, Some(Value::Int64(10)));
        assert_eq!(zm.max, Some(Value::Int64(30)));
        assert_eq!(zm.null_count, 0);
        assert_eq!(zm.row_count, 3);
    }

    #[test]
    fn test_zone_map_u64_empty_slice() {
        let zm = compute_zone_map_u64(&[]);
        assert!(zm.min.is_none());
        assert!(zm.max.is_none());
        assert_eq!(zm.row_count, 0);
    }

    #[test]
    fn test_zone_map_strings_empty_slice() {
        let zm = compute_zone_map_strings(&[]);
        assert!(zm.min.is_none());
        assert!(zm.max.is_none());
        assert_eq!(zm.row_count, 0);
    }

    #[test]
    fn test_zone_map_strings_sorted() {
        let values = &["Paris", "Amsterdam", "Berlin"];
        let zm = compute_zone_map_strings(values);
        assert_eq!(zm.min, Some(Value::from("Amsterdam")));
        assert_eq!(zm.max, Some(Value::from("Paris")));
        assert_eq!(zm.row_count, 3);
    }

    #[test]
    fn test_zone_map_bool_all_true() {
        let values = &[true, true, true];
        let zm = compute_zone_map_bool(values);
        // All true: min = true, max = true.
        assert_eq!(zm.min, Some(Value::Bool(true)));
        assert_eq!(zm.max, Some(Value::Bool(true)));
        assert_eq!(zm.row_count, 3);
    }

    #[test]
    fn test_zone_map_bool_all_false() {
        let values = &[false, false];
        let zm = compute_zone_map_bool(values);
        // All false: min = false, max = false.
        assert_eq!(zm.min, Some(Value::Bool(false)));
        assert_eq!(zm.max, Some(Value::Bool(false)));
        assert_eq!(zm.row_count, 2);
    }

    #[test]
    fn test_zone_map_bool_mixed() {
        let values = &[false, true, false];
        let zm = compute_zone_map_bool(values);
        assert_eq!(zm.min, Some(Value::Bool(false)));
        assert_eq!(zm.max, Some(Value::Bool(true)));
        assert_eq!(zm.row_count, 3);
    }

    #[test]
    fn test_zone_map_bool_empty() {
        let zm = compute_zone_map_bool(&[]);
        assert!(zm.min.is_none());
        assert!(zm.max.is_none());
        assert_eq!(zm.row_count, 0);
    }

    // -------------------------------------------------------------------
    // Type inference edge cases
    // -------------------------------------------------------------------

    #[test]
    fn test_infer_type_string_values() {
        assert_eq!(
            infer_type_from_values(&[Value::from("Alix"), Value::from("Gus")]),
            InferredType::Dict
        );
    }

    #[test]
    fn test_infer_type_int_and_null() {
        // Nulls are skipped, so pure Int64 with nulls remains BitPacked.
        assert_eq!(
            infer_type_from_values(&[Value::Int64(0), Value::Null, Value::Int64(5)]),
            InferredType::BitPacked
        );
    }

    #[test]
    fn test_infer_type_bool_and_null() {
        // Nulls are skipped, so pure Bool with nulls remains Bitmap.
        assert_eq!(
            infer_type_from_values(&[Value::Bool(true), Value::Null]),
            InferredType::Bitmap
        );
    }

    #[test]
    fn test_infer_type_empty_values() {
        // Empty slice: no non-null values seen, defaults to Dict.
        assert_eq!(infer_type_from_values(&[]), InferredType::Dict);
    }

    // -------------------------------------------------------------------
    // from_graph_store: null properties and multi-label nodes
    // -------------------------------------------------------------------

    #[test]
    fn test_from_graph_store_nodes_with_no_properties() {
        use crate::graph::lpg::LpgStore;

        let store = LpgStore::new().unwrap();

        // Nodes with no properties at all.
        store.create_node(&["Marker"]);
        store.create_node(&["Marker"]);

        let compact = from_graph_store(&store).unwrap();
        let ids = compact.nodes_by_label("Marker");
        assert_eq!(ids.len(), 2);
    }

    // -------------------------------------------------------------------
    // from_graph_store: vector-valued properties fall back to Dict
    // -------------------------------------------------------------------

    /// Nodes with `Value::Vector` properties of different dimensions produce
    /// a Dict column (vectors are not an inferred type). Covers the Dict
    /// fallback for "other" values in `infer_type_from_values` and the
    /// per-row Display serialization inside the Dict build path.
    #[test]
    fn test_from_graph_store_mixed_vector_dims() {
        use crate::graph::lpg::LpgStore;
        use std::sync::Arc;

        let store = LpgStore::new().unwrap();

        // Two nodes with differently-dimensioned embeddings.
        let alix = store.create_node(&["Doc"]);
        let short: Arc<[f32]> = Arc::from([0.1f32, 0.2, 0.3].as_slice());
        store.set_node_property(alix, "embedding", Value::Vector(short));

        let gus = store.create_node(&["Doc"]);
        let long: Arc<[f32]> = Arc::from([0.4f32, 0.5, 0.6, 0.7, 0.8].as_slice());
        store.set_node_property(gus, "embedding", Value::Vector(long));

        let compact = from_graph_store(&store).unwrap();
        let ids = compact.nodes_by_label("Doc");
        assert_eq!(ids.len(), 2);

        // Both embeddings should be readable as strings (Dict fallback).
        let mut seen_vec_strings = 0usize;
        for &id in &ids {
            let val = compact
                .get_node_property(id, &PropertyKey::new("embedding"))
                .expect("embedding property missing");
            match val {
                Value::String(s) => {
                    // Display format is lowercase "vector([...])"; compare
                    // case-insensitively to be resilient to formatting tweaks.
                    let lower = s.to_lowercase();
                    assert!(
                        lower.contains("vector"),
                        "dict-encoded vector should include the vector tag: {s}"
                    );
                    seen_vec_strings += 1;
                }
                other => panic!("expected Dict fallback (String), got {other:?}"),
            }
        }
        assert_eq!(seen_vec_strings, 2);
    }

    /// A `Value::Vector` with consistent dimensions across nodes round-trips
    /// through the Float32Vector column codec and comes back as `Value::Vector`
    /// with the original dimensions and values.
    #[test]
    fn test_from_graph_store_float32_vector() {
        use crate::graph::lpg::LpgStore;
        use std::sync::Arc;

        let store = LpgStore::new().unwrap();
        let expected: [f32; 4] = [0.1, 0.2, 0.3, 0.4];
        for name in ["Alix", "Gus", "Vincent"] {
            let id = store.create_node(&["Doc"]);
            store.set_node_property(id, "name", Value::from(name));
            let emb: Arc<[f32]> = Arc::from(expected.as_slice());
            store.set_node_property(id, "embedding", Value::Vector(emb));
        }

        let compact = from_graph_store(&store).unwrap();
        let ids = compact.nodes_by_label("Doc");
        assert_eq!(ids.len(), 3);

        for &id in &ids {
            let v = compact
                .get_node_property(id, &PropertyKey::new("embedding"))
                .expect("embedding missing");
            match v {
                Value::Vector(data) => {
                    assert_eq!(&*data, &expected, "unexpected vector contents");
                }
                other => panic!("expected Value::Vector, got {other:?}"),
            }
        }
    }

    /// Nodes where only ~10% of them carry a given property exercise the
    /// null-padding path in `from_graph_store`. Covers the `push(Value::Null)`
    /// fill loops that keep column lengths aligned to row count.
    #[test]
    fn test_from_graph_store_all_null() {
        use crate::graph::lpg::LpgStore;

        let store = LpgStore::new().unwrap();
        let mut ids = Vec::new();
        for i in 0..20 {
            let nid = store.create_node(&["Item"]);
            // Only 2 out of 20 nodes (10%) have "flag" set.
            if i == 3 || i == 17 {
                store.set_node_property(nid, "flag", Value::Int64(i64::from(i)));
            }
            ids.push(nid);
        }

        let compact = from_graph_store(&store).unwrap();
        let rows = compact.nodes_by_label("Item");
        assert_eq!(rows.len(), 20);

        // Count rows whose "flag" decoded value is nonzero (BitPacked null
        // padding decodes as 0). We expect exactly 2.
        let mut nonzero = 0usize;
        for &id in &rows {
            if let Some(Value::Int64(v)) = compact.get_node_property(id, &PropertyKey::new("flag"))
                && v > 0
            {
                nonzero += 1;
            }
        }
        assert_eq!(
            nonzero, 2,
            "only 2 nodes had a real 'flag' value; the rest should be zero-padded"
        );
    }

    // -------------------------------------------------------------------
    // Zone map boundary cases
    // -------------------------------------------------------------------

    /// Zone map for a u64 column whose maximum equals `i64::MAX` exactly
    /// should keep both bounds (boundary-inclusive), while a max one above
    /// `i64::MAX` drops them. Guards against off-by-one errors in the
    /// overflow check.
    #[test]
    fn test_compute_zone_map_i64_boundary() {
        // Exactly at the boundary: keep bounds.
        let at_boundary = vec![0u64, 100, i64::MAX as u64];
        let zm = compute_zone_map_u64(&at_boundary);
        assert_eq!(zm.min, Some(Value::Int64(0)));
        assert_eq!(zm.max, Some(Value::Int64(i64::MAX)));
        assert_eq!(zm.row_count, 3);

        // One above the boundary: drop bounds, preserve row_count.
        let above_boundary = vec![0u64, i64::MAX as u64 + 1];
        let zm = compute_zone_map_u64(&above_boundary);
        assert!(zm.min.is_none());
        assert!(zm.max.is_none());
        assert_eq!(zm.row_count, 2);
    }

    /// Zone map for a mixed true/false bool column: min is false, max is
    /// true. Distinct from the existing `test_zone_map_bool_mixed` in that
    /// the ratio of true/false is skewed to sanity-check the any()-based
    /// implementation.
    #[test]
    fn test_compute_zone_map_bool() {
        // Heavily skewed: one true, many false.
        let mostly_false = vec![false; 9]
            .into_iter()
            .chain(std::iter::once(true))
            .collect::<Vec<_>>();
        let zm = compute_zone_map_bool(&mostly_false);
        assert_eq!(zm.min, Some(Value::Bool(false)));
        assert_eq!(zm.max, Some(Value::Bool(true)));
        assert_eq!(zm.row_count, 10);

        // Heavily skewed the other way: one false, many true.
        let mostly_true = std::iter::once(false)
            .chain(std::iter::repeat_n(true, 9))
            .collect::<Vec<_>>();
        let zm = compute_zone_map_bool(&mostly_true);
        assert_eq!(zm.min, Some(Value::Bool(false)));
        assert_eq!(zm.max, Some(Value::Bool(true)));
        assert_eq!(zm.row_count, 10);
    }

    #[test]
    fn test_from_graph_store_multi_label_sorted_key() {
        use crate::graph::lpg::LpgStore;

        let store = LpgStore::new().unwrap();

        // Labels "Zebra" and "Alpha" should be sorted to "Alpha|Zebra".
        let a = store.create_node(&["Zebra", "Alpha"]);
        store.set_node_property(a, "name", Value::from("Butch"));

        let compact = from_graph_store(&store).unwrap();
        let ids = compact.nodes_by_label("Alpha|Zebra");
        assert_eq!(ids.len(), 1);

        let val = compact
            .get_node_property(ids[0], &PropertyKey::new("name"))
            .unwrap();
        assert_eq!(val, Value::String(ArcStr::from("Butch")));
    }

    // -------------------------------------------------------------------
    // infer_type_from_values: all Value::* fall-through branches
    // -------------------------------------------------------------------

    #[test]
    fn test_infer_type_timestamp_falls_back_to_dict() {
        use grafeo_common::types::Timestamp;
        let ts = Value::Timestamp(Timestamp::from_millis(1_700_000_000_000));
        assert_eq!(infer_type_from_values(&[ts]), InferredType::Dict);
    }

    #[test]
    fn test_infer_type_date_falls_back_to_dict() {
        use grafeo_common::types::Date;
        let d = Value::Date(Date::from_days(19000));
        assert_eq!(infer_type_from_values(&[d]), InferredType::Dict);
    }

    #[test]
    fn test_infer_type_vector_consistent_dim_picks_float32_vector() {
        // Consistent-dimension Float32 vectors get a dedicated column type.
        let v = Value::Vector(std::sync::Arc::from([0.1f32, 0.2, 0.3].as_slice()));
        assert_eq!(
            infer_type_from_values(&[v]),
            InferredType::Float32Vector { dimensions: 3 }
        );
    }

    #[test]
    fn test_infer_type_mixed_dim_vectors_fall_back_to_dict() {
        // Inconsistent dimensions force the Dict fallback.
        let v3 = Value::Vector(std::sync::Arc::from([0.1f32, 0.2, 0.3].as_slice()));
        let v5 = Value::Vector(std::sync::Arc::from(
            [0.1f32, 0.2, 0.3, 0.4, 0.5].as_slice(),
        ));
        assert_eq!(infer_type_from_values(&[v3, v5]), InferredType::Dict);
    }

    #[test]
    fn test_infer_type_bytes_falls_back_to_dict() {
        let b = Value::Bytes(std::sync::Arc::from(b"payload".as_slice()));
        assert_eq!(infer_type_from_values(&[b]), InferredType::Dict);
    }

    #[test]
    fn test_infer_type_null_with_bool_yields_bitmap() {
        // Nulls skipped; a pure Bool column stays Bitmap.
        assert_eq!(
            infer_type_from_values(&[Value::Null, Value::Bool(false), Value::Null]),
            InferredType::Bitmap
        );
    }

    #[test]
    fn test_infer_type_saw_int_and_other_yields_dict() {
        // Int + String (saw_other) should force Dict.
        assert_eq!(
            infer_type_from_values(&[Value::Int64(1), Value::from("hello")]),
            InferredType::Dict
        );
    }

    // -------------------------------------------------------------------
    // Builder error variants: DuplicateLabel, DuplicateEdgeType
    // -------------------------------------------------------------------

    #[test]
    fn test_builder_duplicate_label_error() {
        let result = CompactStoreBuilder::new()
            .node_table("Person", |t| t.column_bitpacked("age", &[30], 6))
            .node_table("Person", |t| t.column_bitpacked("age", &[40], 6))
            .build();
        assert!(matches!(
            result,
            Err(CompactStoreError::DuplicateLabel(ref s)) if s == "Person"
        ));
    }

    #[test]
    fn test_builder_duplicate_edge_type_error() {
        let result = CompactStoreBuilder::new()
            .node_table("A", |t| t.column_bitpacked("v", &[1], 4))
            .node_table("B", |t| t.column_bitpacked("v", &[1], 4))
            .rel_table("LINKS", "A", "B", |r| r.edges([(0, 0)]))
            .rel_table("LINKS", "A", "B", |r| r.edges([(0, 0)]))
            .build();
        assert!(matches!(
            result,
            Err(CompactStoreError::DuplicateEdgeType(_))
        ));
    }

    #[test]
    fn test_builder_column_length_mismatch_error() {
        let result = CompactStoreBuilder::new()
            .node_table("Person", |t| {
                t.column_bitpacked("age", &[25, 30, 35], 6)
                    .column_dict("name", &["Alix", "Gus"]) // length 2, mismatch
            })
            .build();
        assert!(matches!(
            result,
            Err(CompactStoreError::ColumnLengthMismatch {
                expected: 3,
                got: 2
            })
        ));
    }

    #[test]
    fn test_builder_value_overflow_error() {
        // Values exceeding i64::MAX must be flagged.
        let bad_value = (i64::MAX as u64) + 10;
        let result = CompactStoreBuilder::new()
            .node_table("Person", |t| {
                t.column_bitpacked("x", &[1u64, bad_value], 64)
            })
            .build();
        assert!(matches!(
            result,
            Err(CompactStoreError::ValueOverflow { ref column, value, .. })
                if column == "x" && value == bad_value
        ));
    }

    // -------------------------------------------------------------------
    // Pre-built codec passthrough: NodeTableBuilder::column
    // -------------------------------------------------------------------

    #[test]
    fn test_node_table_builder_prebuilt_column() {
        use crate::codec::BitPackedInts;

        let bp = BitPackedInts::pack(&[100u64, 200, 300]);
        let codec = ColumnCodec::BitPacked(bp);

        let store = CompactStoreBuilder::new()
            .node_table("Item", |t| t.column("value", codec))
            .build()
            .unwrap();

        let ids = store.nodes_by_label("Item");
        assert_eq!(ids.len(), 3);

        // Check the pre-built column values are readable.
        let mut values: Vec<i64> = ids
            .iter()
            .filter_map(|&id| {
                store
                    .get_node_property(id, &PropertyKey::new("value"))
                    .and_then(|v| v.as_int64())
            })
            .collect();
        values.sort_unstable();
        assert_eq!(values, vec![100, 200, 300]);
    }

    // -------------------------------------------------------------------
    // column_int8_vector: zero dimensions case (row_count=0)
    // -------------------------------------------------------------------

    #[test]
    fn test_node_table_builder_int8_vector_zero_dimensions() {
        // Zero dimensions yields 0 rows, no panic.
        let store = CompactStoreBuilder::new()
            .node_table("Item", |t| t.column_int8_vector("embed", Vec::new(), 0))
            .build()
            .unwrap();

        let ids = store.nodes_by_label("Item");
        assert_eq!(ids.len(), 0);
    }

    #[test]
    fn test_node_table_builder_int8_vector_multi_row() {
        // Two 3-dim vectors packed in one flat array.
        let store = CompactStoreBuilder::new()
            .node_table("Doc", |t| {
                t.column_int8_vector("embed", vec![1i8, 2, 3, 4, 5, 6], 3)
            })
            .build()
            .unwrap();

        let ids = store.nodes_by_label("Doc");
        assert_eq!(ids.len(), 2);
    }

    // -------------------------------------------------------------------
    // from_graph_store / from_graph_store_preserving_ids edge cases
    // -------------------------------------------------------------------

    #[test]
    fn test_from_graph_store_preserving_ids_empty() {
        use crate::graph::lpg::LpgStore;

        let store = LpgStore::new().unwrap();
        // No nodes, no edges.
        let compact = crate::graph::compact::from_graph_store_preserving_ids(&store).unwrap();
        assert_eq!(compact.node_count(), 0);
        assert_eq!(compact.edge_count(), 0);
    }

    #[test]
    fn test_from_graph_store_single_node_no_properties() {
        use crate::graph::lpg::LpgStore;

        let store = LpgStore::new().unwrap();
        store.create_node(&["Loner"]);

        let compact = from_graph_store(&store).unwrap();
        let ids = compact.nodes_by_label("Loner");
        assert_eq!(ids.len(), 1);
    }

    #[test]
    fn test_from_graph_store_preserving_ids_with_data() {
        use crate::graph::lpg::LpgStore;

        let store = LpgStore::new().unwrap();
        let alix = store.create_node(&["Person"]);
        store.set_node_property(alix, "name", Value::from("Alix"));
        let gus = store.create_node(&["Person"]);
        store.set_node_property(gus, "name", Value::from("Gus"));
        let edge_id = store.create_edge(alix, gus, "KNOWS");
        store.set_edge_property(edge_id, "since", Value::Int64(2020));

        let compact = crate::graph::compact::from_graph_store_preserving_ids(&store).unwrap();

        // Original IDs should still resolve to nodes.
        let alix_resolved = compact.get_node(alix);
        assert!(
            alix_resolved.is_some(),
            "original NodeId should remain resolvable after preserve_ids"
        );
        assert_eq!(
            alix_resolved
                .unwrap()
                .properties
                .get(&PropertyKey::new("name")),
            Some(&Value::String(ArcStr::from("Alix")))
        );

        let edge_resolved = compact.get_edge(edge_id);
        assert!(
            edge_resolved.is_some(),
            "original EdgeId should remain resolvable after preserve_ids"
        );
    }

    #[test]
    fn test_from_graph_store_skewed_properties() {
        // One label with 5 nodes, each has only one of three properties.
        // This stresses null-padding in sparse columns.
        use crate::graph::lpg::LpgStore;

        let store = LpgStore::new().unwrap();
        for (i, (name, key)) in [
            ("Alix", "a"),
            ("Gus", "b"),
            ("Vincent", "c"),
            ("Jules", "a"),
            ("Mia", "b"),
        ]
        .iter()
        .enumerate()
        {
            let nid = store.create_node(&["Person"]);
            store.set_node_property(nid, "name", Value::from(*name));
            // Property 'a', 'b', or 'c' stored as Int64.
            let score = i64::try_from(i).unwrap_or(0);
            store.set_node_property(nid, key, Value::Int64(score));
        }

        let compact = from_graph_store(&store).unwrap();
        assert_eq!(compact.nodes_by_label("Person").len(), 5);
    }

    // -------------------------------------------------------------------
    // Builder column methods: unique scenarios not covered by earlier tests.
    // `column_int8_vector` happy-path and zero-dim cases live at
    // `test_node_table_builder_int8_vector_multi_row` / `_zero_dimensions`;
    // `column` pre-built-codec passthrough lives at
    // `test_node_table_builder_prebuilt_column`. Keep those canonical.
    // -------------------------------------------------------------------

    #[test]
    #[should_panic(expected = "is not a multiple of dimensions")]
    fn test_node_table_column_int8_vector_not_multiple_panics() {
        // 5 bytes with 2 dimensions is not a multiple.
        let _ = CompactStoreBuilder::new().node_table("Bad", |t| {
            t.column_int8_vector("vec", vec![1, 2, 3, 4, 5], 2)
        });
    }

    #[test]
    fn test_node_table_column_bitmap() {
        let store = CompactStoreBuilder::new()
            .node_table("Flag", |t| {
                t.column_bitmap("active", &[true, false, true, false])
            })
            .build()
            .unwrap();

        let ids = store.nodes_by_label("Flag");
        assert_eq!(ids.len(), 4);

        let v0 = store
            .get_node_property(ids[0], &PropertyKey::new("active"))
            .unwrap();
        assert_eq!(v0, Value::Bool(true));
    }

    // Error-path tests (`column_length_mismatch`, `value_overflow`,
    // `duplicate_label`, `duplicate_edge_type`) live at the matching
    // `*_error` tests above; keep those canonical.

    #[test]
    fn test_builder_same_edge_type_different_labels_allowed() {
        // Same edge type across different label pairs should NOT trigger
        // DuplicateEdgeType (issue #221 regression coverage).
        let result = CompactStoreBuilder::new()
            .node_table("A", |t| t.column_bitpacked("v", &[1], 4))
            .node_table("B", |t| t.column_bitpacked("v", &[1], 4))
            .node_table("C", |t| t.column_bitpacked("v", &[1], 4))
            .rel_table("LINKS", "A", "B", |r| r.edges([(0, 0)]))
            .rel_table("LINKS", "A", "C", |r| r.edges([(0, 0)]))
            .build();
        assert!(result.is_ok());
    }

    #[test]
    fn test_builder_error_types_trait_impls() {
        // Exercise Debug/Display/Clone on CompactStoreError variants.
        let err = CompactStoreError::LabelNotFound("Paris".to_string());
        let cloned = err.clone();
        assert!(format!("{cloned}").contains("Paris"));
        assert!(format!("{cloned:?}").contains("LabelNotFound"));

        let mismatch = CompactStoreError::ColumnLengthMismatch {
            expected: 10,
            got: 5,
        };
        assert!(format!("{mismatch}").contains("10"));
        assert!(format!("{mismatch}").contains('5'));

        let dup_label = CompactStoreError::DuplicateLabel("Berlin".to_string());
        assert!(format!("{dup_label}").contains("Berlin"));

        let dup_edge = CompactStoreError::DuplicateEdgeType("KNOWS".to_string());
        assert!(format!("{dup_edge}").contains("KNOWS"));

        let inconsistent = CompactStoreError::InconsistentEdgeData("boom".to_string());
        assert!(format!("{inconsistent}").contains("boom"));

        let overflow = CompactStoreError::ValueOverflow {
            column: "age".to_string(),
            value: u64::MAX,
            max: i64::MAX as u64,
        };
        assert!(format!("{overflow}").contains("age"));

        let table_overflow = CompactStoreError::TableCountOverflow {
            kind: "node",
            count: 99_999,
            max: MAX_TABLE_ID,
        };
        assert!(format!("{table_overflow}").contains("node"));
        assert!(format!("{table_overflow}").contains("99999"));
    }

    // -------------------------------------------------------------------
    // RelTableBuilder: bit-packed edge properties
    // -------------------------------------------------------------------

    #[test]
    fn test_rel_table_column_bitpacked() {
        // Exercise RelTableBuilder::column_bitpacked (pre-built codec injection on edges).
        let store = CompactStoreBuilder::new()
            .node_table("A", |t| t.column_bitpacked("v", &[1, 2], 4))
            .node_table("B", |t| t.column_bitpacked("v", &[3, 4], 4))
            .rel_table("LINKS", "A", "B", |r| {
                r.edges([(0, 0), (1, 1)])
                    .backward(true)
                    .column_bitpacked("weight", &[100, 200], 8)
            })
            .build()
            .unwrap();

        let a_ids = store.nodes_by_label("A");
        assert_eq!(a_ids.len(), 2);

        // Verify edges exist.
        let mut total_edges = 0;
        for &id in &a_ids {
            total_edges += store
                .edges_from(id, crate::graph::Direction::Outgoing)
                .len();
        }
        assert_eq!(total_edges, 2);
    }

    // -------------------------------------------------------------------
    // from_graph_store_preserving_ids: specialized cases.
    // The basic happy-path is covered by
    // `test_from_graph_store_preserving_ids_with_data`; multi-label and
    // CSR-edge-ordering are unique to this section.
    // -------------------------------------------------------------------

    #[test]
    fn test_from_graph_store_preserving_ids_multi_label() {
        use crate::graph::lpg::LpgStore;

        let store = LpgStore::new().unwrap();
        let butch = store.create_node(&["Person", "Boxer"]);
        store.set_node_property(butch, "name", Value::from("Butch"));

        let compact = from_graph_store_preserving_ids(&store).unwrap();
        assert!(compact.preserves_ids());

        // Multi-label key is sorted as "Boxer|Person".
        let name = compact
            .get_node_property(butch, &PropertyKey::new("name"))
            .and_then(|v| v.as_str().map(str::to_string));
        assert_eq!(name.as_deref(), Some("Butch"));
    }

    #[test]
    fn test_from_graph_store_preserving_ids_edges_sorted_by_csr_order() {
        use crate::graph::lpg::LpgStore;

        let store = LpgStore::new().unwrap();

        // Create nodes with deliberate insertion order to exercise CSR sorting.
        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        let c = store.create_node(&["Node"]);

        // Insert edges in an order that differs from (src, dst) sort order.
        let e_c_a = store.create_edge(c, a, "LINK");
        let e_a_b = store.create_edge(a, b, "LINK");
        let e_b_c = store.create_edge(b, c, "LINK");

        let compact = from_graph_store_preserving_ids(&store).unwrap();

        // All three original edge IDs should resolve and round-trip.
        for eid in [e_c_a, e_a_b, e_b_c] {
            let rec = compact.get_edge(eid).unwrap();
            assert_eq!(rec.edge_type.as_str(), "LINK");
        }
    }

    // -------------------------------------------------------------------
    // IncrementalCompactStoreBuilder
    // -------------------------------------------------------------------

    use grafeo_common::types::{EdgeId, NodeId};

    /// One row of the mixed fixture: every codec the type mapping selects,
    /// sparse keys, multi-label nodes, and edges pushed out of source order.
    struct FixtureRow {
        id: u64,
        labels: &'static [&'static str],
        properties: Vec<(PropertyKey, Value)>,
    }

    fn fixture_nodes() -> Vec<FixtureRow> {
        let key = PropertyKey::new;
        vec![
            FixtureRow {
                id: 10,
                labels: &["Person"],
                properties: vec![
                    (key("name"), Value::from("Alix")),
                    (key("age"), Value::Int64(41)),
                    (key("score"), Value::Float64(1.5)),
                    (key("active"), Value::Bool(true)),
                ],
            },
            FixtureRow {
                id: 11,
                labels: &["Person"],
                properties: vec![
                    (key("name"), Value::from("Gus")),
                    (key("age"), Value::Int64(-7)),
                    (key("blob"), Value::Bytes(vec![0, 1, 2, 255].into())),
                ],
            },
            FixtureRow {
                id: 12,
                labels: &["City", "Place"],
                properties: vec![(key("name"), Value::from("Lyon"))],
            },
            FixtureRow {
                id: 13,
                labels: &["Person"],
                properties: vec![(key("active"), Value::Bool(false))],
            },
            FixtureRow {
                id: 14,
                labels: &["Place", "City"],
                properties: vec![
                    (key("name"), Value::from("Oslo")),
                    (key("population"), Value::Int64(700_000)),
                ],
            },
        ]
    }

    /// `(edge id, type, src, dst, properties)`, deliberately not sorted by
    /// source so the CSR order differs from push order. Every edge carries a
    /// property: the numeric codecs have no null representation, and
    /// `from_graph_store` leaves a table's columns short when its last edges
    /// carry none, so a trailing property-less edge reads back `{}` there and
    /// `{key: 0}` here — a codec limit, not a builder difference.
    fn fixture_edges() -> Vec<(u64, &'static str, u64, u64, Vec<(PropertyKey, Value)>)> {
        let key = PropertyKey::new;
        vec![
            (
                100,
                "LIVES_IN",
                13,
                12,
                vec![(key("weight"), Value::Float64(1.0))],
            ),
            (
                101,
                "KNOWS",
                10,
                11,
                vec![(key("since"), Value::Int64(2020))],
            ),
            (
                102,
                "LIVES_IN",
                10,
                14,
                vec![
                    (key("weight"), Value::Float64(0.5)),
                    (key("note"), Value::from("sparse")),
                ],
            ),
            (
                103,
                "KNOWS",
                11,
                10,
                vec![(key("since"), Value::Int64(2021))],
            ),
            (
                104,
                "LIVES_IN",
                11,
                12,
                vec![(key("weight"), Value::Float64(0.25))],
            ),
            (
                105,
                "KNOWS",
                13,
                13,
                vec![(key("since"), Value::Int64(2022))],
            ),
        ]
    }

    fn fixture_lpg() -> crate::graph::lpg::LpgStore {
        let store = crate::graph::lpg::LpgStore::new().unwrap();
        for row in fixture_nodes() {
            store
                .create_node_with_id(NodeId(row.id), row.labels)
                .unwrap();
            for (key, value) in row.properties {
                store.set_node_property(NodeId(row.id), key.as_str(), value);
            }
        }
        for (id, edge_type, src, dst, properties) in fixture_edges() {
            store
                .create_edge_with_id(EdgeId(id), NodeId(src), NodeId(dst), edge_type)
                .unwrap();
            for (key, value) in properties {
                store.set_edge_property(EdgeId(id), key.as_str(), value);
            }
        }
        store
    }

    fn fixture_incremental() -> CompactStore {
        let mut builder = IncrementalCompactStoreBuilder::new();
        for row in fixture_nodes() {
            builder
                .push_node(
                    NodeId(row.id),
                    row.labels.iter().copied(),
                    row.properties.iter().map(|(k, v)| (k, v)),
                )
                .unwrap();
        }
        for (id, edge_type, src, dst, properties) in fixture_edges() {
            builder
                .push_edge(
                    EdgeId(id),
                    edge_type,
                    NodeId(src),
                    NodeId(dst),
                    properties.iter().map(|(k, v)| (k, v)),
                )
                .unwrap();
        }
        assert_eq!(builder.node_count(), 5);
        assert_eq!(builder.edge_count(), 6);
        builder.finish().unwrap()
    }

    fn section_bytes(store: CompactStore) -> Vec<u8> {
        use grafeo_common::storage::Section;
        super::super::section::CompactStoreSection::new(std::sync::Arc::new(store))
            .serialize()
            .unwrap()
    }

    /// The incremental build decodes every node and edge exactly as the
    /// whole-store conversion of the same rows does, under the original ids.
    #[test]
    fn incremental_builder_matches_from_graph_store_preserving_ids() {
        let lpg = fixture_lpg();
        let reference = from_graph_store_preserving_ids(&lpg).unwrap();
        let incremental = fixture_incremental();

        assert_eq!(incremental.node_count(), reference.node_count());
        assert_eq!(incremental.edge_count(), reference.edge_count());
        let mut labels = incremental.all_labels();
        labels.sort();
        let mut reference_labels = reference.all_labels();
        reference_labels.sort();
        assert_eq!(labels, reference_labels);
        assert_eq!(incremental.max_node_id(), reference.max_node_id());
        assert_eq!(incremental.max_edge_id(), reference.max_edge_id());

        for row in fixture_nodes() {
            let id = NodeId(row.id);
            let got = incremental.get_node(id).expect("node resolves");
            let want = reference.get_node(id).expect("reference node resolves");
            let mut got_labels: Vec<_> = got.labels.iter().map(|l| l.to_string()).collect();
            got_labels.sort();
            let mut want_labels: Vec<_> = want.labels.iter().map(|l| l.to_string()).collect();
            want_labels.sort();
            assert_eq!(got_labels, want_labels, "labels of node {}", row.id);
            let got_props: std::collections::BTreeMap<_, _> = got
                .properties
                .iter()
                .map(|(k, v)| (k.to_string(), v.clone()))
                .collect();
            let want_props: std::collections::BTreeMap<_, _> = want
                .properties
                .iter()
                .map(|(k, v)| (k.to_string(), v.clone()))
                .collect();
            assert_eq!(got_props, want_props, "properties of node {}", row.id);
            // The reference decodes the original values through the same
            // codecs, so this also pins the type mapping against the input.
            for (key, value) in &row.properties {
                let decoded = got_props.get(key.as_str()).expect("pushed key present");
                match value {
                    Value::Bytes(bytes) => assert_eq!(decoded, &Value::Bytes(bytes.clone())),
                    Value::Bool(b) => assert_eq!(decoded, &Value::Bool(*b)),
                    Value::Int64(n) => assert_eq!(decoded, &Value::Int64(*n)),
                    Value::Float64(f) => assert_eq!(decoded, &Value::Float64(*f)),
                    Value::String(s) => assert_eq!(decoded, &Value::String(s.clone())),
                    other => panic!("fixture has no {other:?} rows"),
                }
            }
        }

        for (id, edge_type, src, dst, properties) in fixture_edges() {
            let got = incremental.get_edge(EdgeId(id)).expect("edge resolves");
            let want = reference
                .get_edge(EdgeId(id))
                .expect("reference edge resolves");
            assert_eq!(got.edge_type.as_str(), edge_type);
            assert_eq!((got.src, got.dst), (NodeId(src), NodeId(dst)));
            assert_eq!((got.src, got.dst), (want.src, want.dst));
            let got_props: std::collections::BTreeMap<_, _> = got
                .properties
                .iter()
                .map(|(k, v)| (k.to_string(), v.clone()))
                .collect();
            let want_props: std::collections::BTreeMap<_, _> = want
                .properties
                .iter()
                .map(|(k, v)| (k.to_string(), v.clone()))
                .collect();
            assert_eq!(got_props, want_props, "properties of edge {id}");
            for (key, value) in &properties {
                assert_eq!(got_props.get(key.as_str()), Some(value));
            }
            // Backward adjacency resolves the same edge from its target.
            let incoming: Vec<EdgeId> = incremental
                .edges_from(NodeId(dst), crate::graph::Direction::Incoming)
                .into_iter()
                .map(|(_, eid)| eid)
                .collect();
            assert!(
                incoming.contains(&EdgeId(id)),
                "edge {id} reachable backward"
            );
        }
    }

    /// Rows of one table with different property sets read back with exactly
    /// the properties they were pushed with — in memory, after a section
    /// round trip, and through the property index — for node and edge
    /// columns of every codec, including the marked `Bytes` dictionary.
    #[test]
    fn incremental_builder_keeps_sparse_properties_absent() {
        use grafeo_common::storage::Section;
        use std::sync::Arc;

        let full: Vec<(PropertyKey, Value)> = vec![
            (
                PropertyKey::new("record"),
                Value::Bytes(vec![7u8; 4096].into()),
            ),
            (PropertyKey::new("marker"), Value::from("m")),
            (PropertyKey::new("count"), Value::Int64(3)),
            (PropertyKey::new("neg"), Value::Int64(-3)),
            (PropertyKey::new("ratio"), Value::Float64(0.5)),
            (PropertyKey::new("flag"), Value::Bool(true)),
        ];
        let bare: Vec<(PropertyKey, Value)> = Vec::new();
        let rows: [(u64, &Vec<(PropertyKey, Value)>); 4] =
            [(10, &full), (11, &bare), (12, &full), (13, &bare)];

        let mut builder = IncrementalCompactStoreBuilder::new();
        for (id, properties) in rows {
            builder
                .push_node(
                    NodeId(id),
                    ["Entity"],
                    properties.iter().map(|(k, v)| (k, v)),
                )
                .unwrap();
        }
        // Edges 20 and 22 carry a weight, 21 does not.
        let weight = [(PropertyKey::new("weight"), Value::Int64(7))];
        builder
            .push_edge(
                EdgeId(20),
                "REL",
                NodeId(10),
                NodeId(11),
                weight.iter().map(|(k, v)| (k, v)),
            )
            .unwrap();
        builder
            .push_edge(
                EdgeId(21),
                "REL",
                NodeId(11),
                NodeId(12),
                std::iter::empty(),
            )
            .unwrap();
        builder
            .push_edge(
                EdgeId(22),
                "REL",
                NodeId(12),
                NodeId(13),
                weight.iter().map(|(k, v)| (k, v)),
            )
            .unwrap();
        let built = builder.finish().unwrap();

        let section = super::super::section::CompactStoreSection::new(Arc::new(built));
        let bytes = section.serialize().unwrap();
        let mut restored = super::super::section::CompactStoreSection::empty();
        restored.deserialize(&bytes).unwrap();
        let reopened = restored.store().unwrap();
        reopened.enable_property_indexes([PropertyKey::new("marker")]);

        for store in [section.store().unwrap(), reopened] {
            for (id, properties) in rows {
                let node = store.get_node(NodeId(id)).unwrap();
                let expected: FxHashMap<PropertyKey, Value> = properties.iter().cloned().collect();
                let actual: FxHashMap<PropertyKey, Value> = node
                    .properties
                    .iter()
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect();
                assert_eq!(actual, expected, "node {id}");
                for (key, _) in &full {
                    assert_eq!(
                        store.get_node_property(NodeId(id), key),
                        expected.get(key).cloned(),
                        "node {id} property {key}"
                    );
                }
            }
            assert_eq!(
                store.get_edge(EdgeId(21)).unwrap().properties.len(),
                0,
                "edge 21 never had a weight"
            );
            assert_eq!(
                store.get_edge_property(EdgeId(20), &PropertyKey::new("weight")),
                Some(Value::Int64(7))
            );
            assert_eq!(
                store.get_edge_property(EdgeId(21), &PropertyKey::new("weight")),
                None
            );
            // The padding values are not findable, indexed or scanned.
            assert!(
                store
                    .find_nodes_by_property("marker", &Value::from(""))
                    .is_empty()
            );
            assert!(
                store
                    .find_nodes_by_property("count", &Value::Int64(0))
                    .is_empty()
            );
            assert!(
                store
                    .find_nodes_by_property("flag", &Value::Bool(false))
                    .is_empty()
            );
            let mut with_marker = store.find_nodes_by_property("marker", &Value::from("m"));
            with_marker.sort();
            assert_eq!(with_marker, vec![NodeId(10), NodeId(12)]);
            let mut in_range = store.find_nodes_in_range(
                "count",
                Some(&Value::Int64(0)),
                Some(&Value::Int64(10)),
                true,
                true,
            );
            in_range.sort();
            assert_eq!(in_range, vec![NodeId(10), NodeId(12)]);
        }
    }

    /// Identical rows in identical order serialize to identical bytes, and a
    /// different push order does not lose or reorder any row's identity.
    #[test]
    fn incremental_builder_is_byte_deterministic() {
        let first = section_bytes(fixture_incremental());
        let second = section_bytes(fixture_incremental());
        assert!(!first.is_empty());
        assert_eq!(first, second);

        // Reversed edge order: same rows, same ids, same CSR (source-sorted,
        // stable), so every id still resolves to its own endpoints.
        let mut builder = IncrementalCompactStoreBuilder::new();
        for row in fixture_nodes() {
            builder
                .push_node(
                    NodeId(row.id),
                    row.labels.iter().copied(),
                    row.properties.iter().map(|(k, v)| (k, v)),
                )
                .unwrap();
        }
        for (id, edge_type, src, dst, properties) in fixture_edges().into_iter().rev() {
            builder
                .push_edge(
                    EdgeId(id),
                    edge_type,
                    NodeId(src),
                    NodeId(dst),
                    properties.iter().map(|(k, v)| (k, v)),
                )
                .unwrap();
        }
        let reversed = builder.finish().unwrap();
        for (id, _, src, dst, _) in fixture_edges() {
            let edge = reversed.get_edge(EdgeId(id)).unwrap();
            assert_eq!((edge.src, edge.dst), (NodeId(src), NodeId(dst)));
        }
    }

    #[test]
    fn incremental_builder_rejects_duplicate_node_ids() {
        let mut builder = IncrementalCompactStoreBuilder::new();
        builder
            .push_node(NodeId(1), ["A"], std::iter::empty())
            .unwrap();
        let err = builder
            .push_node(NodeId(1), ["B"], std::iter::empty())
            .unwrap_err();
        assert!(
            matches!(err, CompactStoreError::DuplicateNodeId(1)),
            "{err}"
        );
        // The refused push left no trace: one node, one table.
        assert_eq!(builder.node_count(), 1);
        let store = builder.finish().unwrap();
        assert_eq!(store.all_labels(), vec!["A".to_string()]);
    }

    #[test]
    fn incremental_builder_rejects_duplicate_edge_ids() {
        let mut builder = IncrementalCompactStoreBuilder::new();
        builder
            .push_node(NodeId(1), ["A"], std::iter::empty())
            .unwrap();
        builder
            .push_node(NodeId(2), ["A"], std::iter::empty())
            .unwrap();
        builder
            .push_edge(EdgeId(7), "R", NodeId(1), NodeId(2), std::iter::empty())
            .unwrap();
        let err = builder
            .push_edge(EdgeId(7), "R", NodeId(2), NodeId(1), std::iter::empty())
            .unwrap_err();
        assert!(
            matches!(err, CompactStoreError::DuplicateEdgeId(7)),
            "{err}"
        );
        let store = builder.finish().unwrap();
        assert_eq!(store.edge_count(), 1);
        let edge = store.get_edge(EdgeId(7)).unwrap();
        assert_eq!((edge.src, edge.dst), (NodeId(1), NodeId(2)));
    }

    #[test]
    fn incremental_builder_rejects_edges_to_unknown_nodes() {
        let mut builder = IncrementalCompactStoreBuilder::new();
        builder
            .push_node(NodeId(1), ["A"], std::iter::empty())
            .unwrap();
        let err = builder
            .push_edge(EdgeId(7), "R", NodeId(1), NodeId(9), std::iter::empty())
            .unwrap_err();
        assert!(
            matches!(err, CompactStoreError::UnknownEndpoint { edge: 7, node: 9 }),
            "{err}"
        );
        let err = builder
            .push_edge(EdgeId(8), "R", NodeId(9), NodeId(1), std::iter::empty())
            .unwrap_err();
        assert!(
            matches!(err, CompactStoreError::UnknownEndpoint { edge: 8, node: 9 }),
            "{err}"
        );
        // A refused edge does not reserve its id or create its table.
        assert_eq!(builder.edge_count(), 0);
        builder
            .push_node(NodeId(9), ["A"], std::iter::empty())
            .unwrap();
        builder
            .push_edge(EdgeId(7), "R", NodeId(1), NodeId(9), std::iter::empty())
            .unwrap();
        let store = builder.finish().unwrap();
        assert_eq!(store.edge_count(), 1);
        assert_eq!(store.rel_tables_for_type("R").len(), 1);
    }

    #[test]
    fn incremental_builder_rejects_unlabeled_nodes() {
        let mut builder = IncrementalCompactStoreBuilder::new();
        let err = builder
            .push_node(NodeId(1), std::iter::empty(), std::iter::empty())
            .unwrap_err();
        assert!(matches!(err, CompactStoreError::UnlabeledNode(1)), "{err}");
        assert_eq!(builder.node_count(), 0);
    }

    #[test]
    fn incremental_builder_empty_finishes_to_empty_store() {
        let store = IncrementalCompactStoreBuilder::new().finish().unwrap();
        assert_eq!(store.node_count(), 0);
        assert_eq!(store.edge_count(), 0);
        assert!(store.max_node_id().is_none());
    }

    fn fixture_builder(builder: &mut IncrementalCompactStoreBuilder) {
        for row in fixture_nodes() {
            builder
                .push_node(
                    NodeId(row.id),
                    row.labels.iter().copied(),
                    row.properties.iter().map(|(k, v)| (k, v)),
                )
                .unwrap();
        }
        for (id, edge_type, src, dst, properties) in fixture_edges() {
            builder
                .push_edge(
                    EdgeId(id),
                    edge_type,
                    NodeId(src),
                    NodeId(dst),
                    properties.iter().map(|(k, v)| (k, v)),
                )
                .unwrap();
        }
    }

    fn streamed_bytes(builder: IncrementalCompactStoreBuilder) -> Vec<u8> {
        let mut bytes = Vec::new();
        builder.write_section(&mut bytes, &[]).unwrap();
        bytes
    }

    /// Rows enough to fill several spool chunks per column: long unique
    /// strings, a sparse column, signed and unsigned integers, and edges
    /// pushed against source order.
    fn wide_builder(mut builder: IncrementalCompactStoreBuilder) -> IncrementalCompactStoreBuilder {
        const NODES: u64 = 3_000;
        for index in 0..NODES {
            let mut properties = vec![
                (
                    PropertyKey::new("identity"),
                    Value::from(format!("entity:{index:06}:{}", "x".repeat(40))),
                ),
                (
                    PropertyKey::new("rank"),
                    Value::Int64(i64::try_from(index).unwrap() - 1_000),
                ),
            ];
            if index % 7 == 0 {
                properties.push((
                    PropertyKey::new("sparse"),
                    Value::Int64(i64::try_from(index).unwrap()),
                ));
            }
            let labels: &[&str] = if index % 2 == 0 { &["Even"] } else { &["Odd"] };
            builder
                .push_node(
                    NodeId(index),
                    labels.iter().copied(),
                    properties.iter().map(|(k, v)| (k, v)),
                )
                .unwrap();
        }
        for index in 0..NODES {
            let src = NodeId(NODES - 1 - index);
            let dst = NodeId((index * 7) % NODES);
            let properties = [(
                PropertyKey::new("relation"),
                Value::from(format!("relation:{index:06}:{}", "y".repeat(40))),
            )];
            builder
                .push_edge(
                    EdgeId(index),
                    "LINKS",
                    src,
                    dst,
                    properties.iter().map(|(k, v)| (k, v)),
                )
                .unwrap();
        }
        builder
    }

    /// The streamed section is the finished store's section, byte for byte,
    /// for the mixed fixture, the empty builder, and a spooled build whose
    /// columns span many spool chunks on disk.
    #[test]
    fn streamed_section_is_the_finished_store_section() {
        let mut streamed = IncrementalCompactStoreBuilder::new();
        fixture_builder(&mut streamed);
        assert_eq!(
            streamed_bytes(streamed),
            section_bytes(fixture_incremental())
        );

        assert_eq!(
            streamed_bytes(IncrementalCompactStoreBuilder::new()),
            section_bytes(IncrementalCompactStoreBuilder::new().finish().unwrap())
        );

        let resident = section_bytes(
            wide_builder(IncrementalCompactStoreBuilder::new())
                .finish()
                .unwrap(),
        );
        let spool = tempfile::tempfile().unwrap();
        let probe = spool.try_clone().unwrap();
        let spooled = wide_builder(IncrementalCompactStoreBuilder::spooling_to(spool));
        let spooled_bytes = probe.metadata().unwrap().len();
        assert!(
            spooled_bytes > 8 * SPOOL_CHUNK_BYTES as u64,
            "the wide fixture spilled {spooled_bytes} bytes"
        );
        assert_eq!(streamed_bytes(spooled), resident);
        let spooled_store = wide_builder(IncrementalCompactStoreBuilder::spooling_to(
            tempfile::tempfile().unwrap(),
        ))
        .finish()
        .unwrap();
        assert_eq!(section_bytes(spooled_store), resident);
    }

    /// A section streamed with indexed properties is the finished store's
    /// section once those properties are enabled on it, and the store read
    /// back serves the lookups from the stored order without building one.
    #[test]
    fn streamed_indexed_properties_reopen_already_indexed() {
        use grafeo_common::storage::Section;

        let indexed = [PropertyKey::new("identity"), PropertyKey::new("sparse")];
        let mut streamed = Vec::new();
        wide_builder(IncrementalCompactStoreBuilder::new())
            .write_section(&mut streamed, &indexed)
            .unwrap();
        let resident = wide_builder(IncrementalCompactStoreBuilder::new())
            .finish()
            .unwrap();
        resident.enable_property_indexes(indexed.iter().cloned());
        assert_eq!(streamed, section_bytes(resident));

        let mut restored = super::super::section::CompactStoreSection::empty();
        restored.deserialize(&streamed).unwrap();
        let store = restored.store().unwrap();
        let mut keys = store.indexed_property_keys();
        keys.sort_by(|a, b| a.as_str().cmp(b.as_str()));
        assert_eq!(keys, indexed.to_vec());
        let identity = Value::from(format!("entity:{:06}:{}", 1_234, "x".repeat(40)));
        assert_eq!(
            store.find_nodes_by_property("identity", &identity),
            vec![NodeId(1_234)]
        );
        assert_eq!(
            store.find_nodes_by_property("sparse", &Value::Int64(2_100)),
            vec![NodeId(2_100)]
        );
        assert_eq!(
            store.find_nodes_by_property("sparse", &Value::Int64(2_101)),
            Vec::<NodeId>::new()
        );
    }

    /// A key repeated within one row keeps its first value, as the
    /// null-padded column form did.
    #[test]
    fn a_repeated_key_keeps_its_first_value() {
        let mut builder = IncrementalCompactStoreBuilder::new();
        let key = PropertyKey::new("name");
        let first = Value::from("first");
        let second = Value::from("second");
        builder
            .push_node(NodeId(1), ["A"], [(&key, &first), (&key, &second)])
            .unwrap();
        builder
            .push_node(NodeId(2), ["A"], [(&key, &second)])
            .unwrap();
        let store = builder.finish().unwrap();
        assert_eq!(store.get_node_property(NodeId(1), &key), Some(first));
        assert_eq!(store.get_node_property(NodeId(2), &key), Some(second));
    }

    /// String values longer than a section's `u16` name length round-trip
    /// whole through the streamed and the resident section writers, and a
    /// predicate against them still finds their rows.
    #[test]
    fn string_values_over_64_kib_round_trip_through_the_section() {
        use grafeo_common::storage::Section;

        let key = PropertyKey::new("record");
        let long = Value::from(format!("{}z", "a".repeat(70 * 1024)));
        let longest = Value::from(format!("{}z", "b".repeat(80 * 1024)));
        let short = Value::from("m");
        let build = || {
            let mut builder = IncrementalCompactStoreBuilder::new();
            for (id, value) in [(1, &long), (2, &short), (3, &longest)] {
                builder
                    .push_node(NodeId(id), ["Entity"], [(&key, value)])
                    .unwrap();
            }
            builder
        };
        let streamed = streamed_bytes(build());
        assert_eq!(streamed, section_bytes(build().finish().unwrap()));

        let mut restored = super::super::section::CompactStoreSection::empty();
        restored.deserialize(&streamed).unwrap();
        let store = restored.store().unwrap();
        for (id, value) in [(1, &long), (2, &short), (3, &longest)] {
            assert_eq!(
                store.get_node_property(NodeId(id), &key),
                Some(value.clone())
            );
        }
        assert_eq!(
            store.find_nodes_by_property(key.as_str(), &longest),
            vec![NodeId(3)]
        );
    }

    /// A label, property key, or edge type the section cannot name is a
    /// typed write error.
    #[test]
    fn a_name_over_64_kib_is_a_typed_section_error() {
        let label = "L".repeat(usize::from(u16::MAX) + 1);
        let mut builder = IncrementalCompactStoreBuilder::new();
        builder
            .push_node(NodeId(1), [label.as_str()], std::iter::empty())
            .unwrap();
        let err = builder.write_section(&mut Vec::new(), &[]).unwrap_err();
        assert_eq!(
            err.to_string(),
            "GRAFEO-V001: Invalid value: a compact section name is limited to 65535 bytes, \
             got 65536"
        );
    }

    /// The streamed section is single-use and write-only.
    #[test]
    fn incremental_section_refuses_a_second_write_and_reads() {
        use grafeo_common::storage::Section;
        let mut builder = IncrementalCompactStoreBuilder::new();
        fixture_builder(&mut builder);
        let mut section =
            super::super::section::IncrementalCompactStoreSection::new(builder, Vec::new());
        assert!(section.is_dirty());
        assert_eq!(
            section.serialize().unwrap(),
            section_bytes(fixture_incremental())
        );
        assert!(!section.is_dirty());
        let err = section.serialize().unwrap_err();
        assert!(err.to_string().contains("already written"), "{err}");
        let err = section.deserialize(&[]).unwrap_err();
        assert!(err.to_string().contains("write-only"), "{err}");
    }
}
