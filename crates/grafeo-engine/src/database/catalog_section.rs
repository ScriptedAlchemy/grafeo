//! Catalog section serializer for the `.grafeo` container format.
//!
//! Serializes schema definitions (node types, edge types, graph types, procedures),
//! index metadata (property, vector, text), and epoch state into the `CATALOG` section.

// Parts of this module are reserved for Phase 5 checkpoint integration.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use serde::{Deserialize, Serialize};

use grafeo_common::storage::section::{Section, SectionType};
use grafeo_common::utils::error::{Error, Result};

use crate::catalog::{
    Catalog, EdgeTypeDefinition, GraphTypeDefinition, NodeTypeDefinition, ProcedureDefinition,
};

/// Current catalog section format version.
///
/// `version` is the first field of [`CatalogSnapshot`] and a `u8`, which
/// bincode's standard config writes as a single leading byte — so a reader
/// can identify the payload format from `data[0]` before attempting a full
/// decode. [`CatalogSection::deserialize`] rejects any unsupported value
/// with a typed unsupported-version error: the section CRC has already
/// proven the bytes intact by the time the payload reaches this module, so
/// a foreign version byte means an incompatible revision wrote the store,
/// not corruption.
///
/// v2 adds the quantization mode and the caller's binding token to each
/// vector index entry, both of which the loader needs before it can
/// re-register an index without rebuilding it. v1 stores keep loading
/// through the retained [`CatalogSnapshotV1`] decoder.
const CATALOG_SECTION_VERSION: u8 = 2;

/// Previous catalog format, still readable.
const CATALOG_SECTION_VERSION_V1: u8 = 1;

// ── Snapshot types ──────────────────────────────────────────────────

#[derive(Serialize, Deserialize)]
struct CatalogSnapshot {
    version: u8,
    schema: SnapshotSchema,
    indexes: SnapshotIndexes,
    epoch: u64,
}

/// v1 snapshot, retained so files written before the vector-index
/// restore landed still open.
#[derive(Deserialize)]
struct CatalogSnapshotV1 {
    #[allow(dead_code)]
    version: u8,
    schema: SnapshotSchema,
    indexes: SnapshotIndexesV1,
    #[allow(dead_code)]
    epoch: u64,
}

#[derive(Deserialize, Default)]
struct SnapshotIndexesV1 {
    property_indexes: Vec<String>,
    #[allow(dead_code)]
    vector_indexes: Vec<SnapshotVectorIndexV1>,
    #[allow(dead_code)]
    text_indexes: Vec<SnapshotTextIndex>,
}

#[derive(Deserialize)]
struct SnapshotVectorIndexV1 {
    #[allow(dead_code)]
    label: String,
    #[allow(dead_code)]
    property: String,
    #[allow(dead_code)]
    dimensions: usize,
    #[allow(dead_code)]
    metric: grafeo_core::index::vector::DistanceMetric,
    #[allow(dead_code)]
    m: usize,
    #[allow(dead_code)]
    ef_construction: usize,
}

#[derive(Serialize, Deserialize, Default)]
struct SnapshotSchema {
    node_types: Vec<NodeTypeDefinition>,
    edge_types: Vec<EdgeTypeDefinition>,
    graph_types: Vec<GraphTypeDefinition>,
    procedures: Vec<ProcedureDefinition>,
    schemas: Vec<String>,
    graph_type_bindings: Vec<(String, String)>,
}

#[derive(Serialize, Deserialize, Default)]
struct SnapshotIndexes {
    property_indexes: Vec<String>,
    vector_indexes: Vec<SnapshotVectorIndex>,
    text_indexes: Vec<SnapshotTextIndex>,
}

#[derive(Serialize, Deserialize)]
struct SnapshotVectorIndex {
    label: String,
    property: String,
    dimensions: usize,
    metric: grafeo_core::index::vector::DistanceMetric,
    m: usize,
    ef_construction: usize,
    /// `true` when the index quantizes its vectors.
    ///
    /// A quantized index keeps its codebook inside the index, and the
    /// `VectorStore` section carries only the HNSW topology - so a
    /// quantized index cannot be restored from sections alone and must
    /// be rebuilt. Recording the mode is what lets the loader tell the
    /// two cases apart instead of silently restoring a quantized index
    /// as a full-precision one.
    quantized: bool,
    /// Opaque token the caller stamped on this index, if any.
    binding: Option<String>,
}

/// A vector index definition read back from the catalog, ready to be
/// re-registered before the `VectorStore` section is loaded into it.
///
/// Returned by [`CatalogSection::restored_vector_indexes`]. Quantized
/// indexes are not represented here: see [`SnapshotVectorIndex`].
#[derive(Clone, Debug)]
pub struct RestoredVectorIndex {
    /// Node label the index covers.
    pub label: String,
    /// Property holding the vectors.
    pub property: String,
    /// Vector dimensions.
    pub dimensions: usize,
    /// Distance metric.
    pub metric: grafeo_core::index::vector::DistanceMetric,
    /// HNSW links per node.
    pub m: usize,
    /// HNSW construction beam width.
    pub ef_construction: usize,
}

#[derive(Serialize, Deserialize)]
struct SnapshotTextIndex {
    label: String,
    property: String,
}

// ── Section implementation ──────────────────────────────────────────

/// Catalog section for the `.grafeo` container.
///
/// Serializes schema definitions and index metadata. The catalog is always
/// small (typically < 10 KB) and always kept in RAM.
pub struct CatalogSection {
    catalog: Arc<Catalog>,
    store: Arc<grafeo_core::graph::lpg::LpgStore>,
    epoch_fn: Box<dyn Fn() -> u64 + Send + Sync>,
    dirty: AtomicBool,
    /// Property index names read back by [`deserialize`](Self::deserialize).
    ///
    /// Creating an index scans the rows it covers, so the loader cannot
    /// act on these until the data sections are in - see
    /// [`restored_property_indexes`](Self::restored_property_indexes).
    restored_property_indexes: parking_lot::Mutex<Vec<String>>,
    /// Vector index definitions read back by [`deserialize`](Self::deserialize).
    ///
    /// Same deferral as the property indexes, for a sharper reason: the
    /// loader must register these *empty* and then let the `VectorStore`
    /// section fill in the persisted topology. Registering them here
    /// would race the section load, and building them the ordinary way
    /// would scan and re-link every vector - the rebuild this section
    /// exists to avoid.
    restored_vector_indexes: parking_lot::Mutex<Vec<RestoredVectorIndex>>,
}

impl CatalogSection {
    /// Create a new catalog section.
    ///
    /// The `epoch_fn` closure returns the current MVCC epoch. This avoids a
    /// dependency on `TransactionManager` which lives in the engine layer.
    pub fn new(
        catalog: Arc<Catalog>,
        store: Arc<grafeo_core::graph::lpg::LpgStore>,
        epoch_fn: impl Fn() -> u64 + Send + Sync + 'static,
    ) -> Self {
        Self {
            catalog,
            store,
            epoch_fn: Box::new(epoch_fn),
            dirty: AtomicBool::new(false),
            restored_property_indexes: parking_lot::Mutex::new(Vec::new()),
            restored_vector_indexes: parking_lot::Mutex::new(Vec::new()),
        }
    }

    /// Property index names this section carried, once deserialized.
    ///
    /// The loader replays them after the LPG section lands: without that
    /// every property index is silently lost across a reopen, and lookups
    /// that were O(1) fall back to a full scan.
    pub fn restored_property_indexes(&self) -> Vec<String> {
        self.restored_property_indexes.lock().clone()
    }

    /// Non-quantized vector index definitions this section carried.
    ///
    /// The loader registers each one as an empty index and then loads
    /// the `VectorStore` section into it. Without this the reopened
    /// store has no index entries, the section's restore loop finds
    /// nothing to fill, and the persisted HNSW topology is discarded -
    /// leaving vector search unavailable until something rebuilds it.
    pub fn restored_vector_indexes(&self) -> Vec<RestoredVectorIndex> {
        self.restored_vector_indexes.lock().clone()
    }

    /// Mark this section as dirty.
    #[allow(dead_code)] // Wired in Phase 5 checkpoint path
    pub fn mark_dirty(&self) {
        self.dirty.store(true, Ordering::Release);
    }

    fn collect_schema(&self) -> SnapshotSchema {
        SnapshotSchema {
            node_types: self.catalog.all_node_type_defs(),
            edge_types: self.catalog.all_edge_type_defs(),
            graph_types: self.catalog.all_graph_type_defs(),
            procedures: self.catalog.all_procedure_defs(),
            schemas: self.catalog.schema_names(),
            graph_type_bindings: self.catalog.all_graph_type_bindings(),
        }
    }

    fn collect_indexes(&self) -> SnapshotIndexes {
        let property_indexes = self.store.property_index_keys();

        #[cfg(feature = "vector-index")]
        let vector_indexes: Vec<SnapshotVectorIndex> = self
            .store
            .vector_index_entries()
            .into_iter()
            .filter_map(|(key, index)| {
                let (label, property) = key.split_once(':')?;
                let config = index.config();
                Some(SnapshotVectorIndex {
                    label: label.to_string(),
                    property: property.to_string(),
                    dimensions: config.dimensions,
                    metric: config.metric,
                    m: config.m,
                    ef_construction: config.ef_construction,
                    quantized: matches!(
                        &*index,
                        grafeo_core::index::vector::VectorIndexKind::Quantized(_)
                    ),
                    binding: self.store.vector_index_binding(label, property),
                })
            })
            .collect();
        #[cfg(not(feature = "vector-index"))]
        let vector_indexes = Vec::new();

        #[cfg(feature = "text-index")]
        let text_indexes: Vec<SnapshotTextIndex> = self
            .store
            .text_index_entries()
            .into_iter()
            .filter_map(|(key, _)| {
                let (label, property) = key.split_once(':')?;
                Some(SnapshotTextIndex {
                    label: label.to_string(),
                    property: property.to_string(),
                })
            })
            .collect();
        #[cfg(not(feature = "text-index"))]
        let text_indexes = Vec::new();

        SnapshotIndexes {
            property_indexes,
            vector_indexes,
            text_indexes,
        }
    }
}

impl Section for CatalogSection {
    fn section_type(&self) -> SectionType {
        SectionType::Catalog
    }

    fn version(&self) -> u8 {
        CATALOG_SECTION_VERSION
    }

    fn serialize(&self) -> Result<Vec<u8>> {
        let snapshot = CatalogSnapshot {
            version: CATALOG_SECTION_VERSION,
            schema: self.collect_schema(),
            indexes: self.collect_indexes(),
            epoch: (self.epoch_fn)(),
        };

        let config = bincode::config::standard();
        bincode::serde::encode_to_vec(&snapshot, config)
            .map_err(|e| Error::Internal(format!("Catalog section serialization failed: {e}")))
    }

    fn deserialize(&mut self, data: &[u8]) -> Result<()> {
        // Validate the format version before decoding anything else. A
        // mismatched snapshot shape would otherwise surface as opaque
        // bincode noise — or, when the divergent fields happen to be in
        // empty collections, decode silently under the wrong format.
        let found = *data.first().ok_or_else(|| {
            Error::Serialization(
                "Catalog section is empty: missing format version byte".to_string(),
            )
        })?;
        if found != CATALOG_SECTION_VERSION && found != CATALOG_SECTION_VERSION_V1 {
            return Err(Error::Serialization(format!(
                "unsupported catalog version {found} \
                 (supported {CATALOG_SECTION_VERSION_V1}, {CATALOG_SECTION_VERSION})"
            )));
        }

        let config = bincode::config::standard();

        // `version` is the snapshot's first field and a u8, so bincode's
        // standard config puts it in the leading byte. Read the format
        // off that rather than guessing from a failed decode: a v1
        // payload fed to the v2 decoder can mis-parse rather than error.
        if found == CATALOG_SECTION_VERSION_V1 {
            let (v1, _): (CatalogSnapshotV1, _) = bincode::serde::decode_from_slice(data, config)
                .map_err(|e| {
                    Error::Serialization(format!("Catalog section v1 deserialization failed: {e}"))
                })?;
            // v1 files record vector index metadata but carry neither the
            // quantization mode nor a binding token, so their indexes are
            // rebuilt the old way rather than restored. Only the property
            // indexes carry over.
            self.restore_schema(&v1.schema);
            self.restored_property_indexes
                .lock()
                .clone_from(&v1.indexes.property_indexes);
            self.restored_vector_indexes.lock().clear();
            return Ok(());
        }

        let (snapshot, _): (CatalogSnapshot, _) = bincode::serde::decode_from_slice(data, config)
            .map_err(|e| {
            Error::Serialization(format!("Catalog section deserialization failed: {e}"))
        })?;

        self.restore_schema(&snapshot.schema);

        // Index rebuilding scans the rows, so it has to wait until the
        // data sections are loaded. Hand the names to the loader instead.
        self.restored_property_indexes
            .lock()
            .clone_from(&snapshot.indexes.property_indexes);

        // Binding tokens are pure metadata: no rows to scan, so they can
        // land now. They are restored even for quantized indexes, whose
        // topology cannot be, so a caller can still see which generation
        // the index it is about to rebuild belonged to.
        #[cfg(feature = "vector-index")]
        {
            let mut restorable = Vec::new();
            for index in &snapshot.indexes.vector_indexes {
                if let Some(binding) = &index.binding {
                    self.store
                        .set_vector_index_binding(&index.label, &index.property, binding);
                }
                if index.quantized {
                    continue;
                }
                restorable.push(RestoredVectorIndex {
                    label: index.label.clone(),
                    property: index.property.clone(),
                    dimensions: index.dimensions,
                    metric: index.metric,
                    m: index.m,
                    ef_construction: index.ef_construction,
                });
            }
            *self.restored_vector_indexes.lock() = restorable;
        }

        Ok(())
    }

    fn is_dirty(&self) -> bool {
        self.dirty.load(Ordering::Acquire)
    }

    fn mark_clean(&self) {
        self.dirty.store(false, Ordering::Release);
    }

    fn memory_usage(&self) -> usize {
        // Catalog is tiny: schema defs + index metadata, typically < 10 KB
        4096
    }
}

impl CatalogSection {
    /// Replays schema definitions shared by every catalog format version.
    fn restore_schema(&self, schema: &SnapshotSchema) {
        for def in &schema.node_types {
            self.catalog.register_or_replace_node_type(def.clone());
        }
        for def in &schema.edge_types {
            self.catalog.register_or_replace_edge_type_def(def.clone());
        }
        for def in &schema.graph_types {
            let _ = self.catalog.register_graph_type(def.clone());
        }
        for def in &schema.procedures {
            self.catalog.replace_procedure(def.clone()).ok();
        }
        for name in &schema.schemas {
            let _ = self.catalog.register_schema_namespace(name.clone());
            let default_key = format!("{name}/__default__");
            let _ = self.store.create_graph(&default_key);
        }
        for (graph_name, type_name) in &schema.graph_type_bindings {
            let _ = self.catalog.bind_graph_type(graph_name, type_name.clone());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::{EdgeTypeDefinition, NodeTypeDefinition, TypedProperty};

    fn make_section() -> CatalogSection {
        let catalog = Arc::new(Catalog::new());
        let store = Arc::new(grafeo_core::graph::lpg::LpgStore::new().unwrap());
        CatalogSection::new(catalog, store, || 42)
    }

    #[test]
    fn empty_catalog_roundtrip() {
        let section = make_section();
        let bytes = section.serialize().expect("serialize empty catalog");
        assert!(!bytes.is_empty());

        let catalog2 = Arc::new(Catalog::new());
        let store2 = Arc::new(grafeo_core::graph::lpg::LpgStore::new().unwrap());
        let mut section2 = CatalogSection::new(catalog2, store2, || 0);
        section2
            .deserialize(&bytes)
            .expect("deserialize empty catalog");
    }

    #[test]
    fn catalog_with_node_types_roundtrip() {
        let section = make_section();
        section
            .catalog
            .register_or_replace_node_type(NodeTypeDefinition {
                name: "Person".to_string(),
                properties: vec![TypedProperty {
                    name: "name".to_string(),
                    data_type: crate::catalog::PropertyDataType::String,
                    nullable: false,
                    default_value: None,
                }],
                constraints: vec![],
                parent_types: vec![],
            });

        let bytes = section.serialize().unwrap();

        let catalog2 = Arc::new(Catalog::new());
        let store2 = Arc::new(grafeo_core::graph::lpg::LpgStore::new().unwrap());
        let mut section2 = CatalogSection::new(catalog2, store2, || 0);
        section2.deserialize(&bytes).unwrap();

        let types = section2.catalog.all_node_type_defs();
        assert_eq!(types.len(), 1);
        assert_eq!(types[0].name, "Person");
        assert_eq!(types[0].properties.len(), 1);
    }

    #[test]
    fn catalog_with_edge_types_roundtrip() {
        let section = make_section();
        section
            .catalog
            .register_or_replace_edge_type_def(EdgeTypeDefinition {
                name: "KNOWS".to_string(),
                properties: vec![],
                constraints: vec![],
                source_node_types: vec![],
                target_node_types: vec![],
            });

        let bytes = section.serialize().unwrap();

        let catalog2 = Arc::new(Catalog::new());
        let store2 = Arc::new(grafeo_core::graph::lpg::LpgStore::new().unwrap());
        let mut section2 = CatalogSection::new(catalog2, store2, || 0);
        section2.deserialize(&bytes).unwrap();

        let types = section2.catalog.all_edge_type_defs();
        assert_eq!(types.len(), 1);
        assert_eq!(types[0].name, "KNOWS");
    }

    #[test]
    fn catalog_section_type_and_version() {
        let section = make_section();
        assert_eq!(section.section_type(), SectionType::Catalog);
        assert_eq!(section.version(), CATALOG_SECTION_VERSION);
    }

    #[test]
    fn catalog_dirty_tracking() {
        let section = make_section();
        assert!(!section.is_dirty());

        section.mark_dirty();
        assert!(section.is_dirty());

        section.mark_clean();
        assert!(!section.is_dirty());
    }

    #[test]
    fn catalog_memory_usage() {
        let section = make_section();
        assert_eq!(section.memory_usage(), 4096);
    }

    #[test]
    fn catalog_deserialize_corrupt_data() {
        let mut section = make_section();
        let result = section.deserialize(&[0x01, 0xFE, 0xFD, 0x00]);
        assert!(result.is_err(), "corrupt data should fail deserialization");
    }

    #[test]
    fn catalog_deserialize_rejects_unsupported_version() {
        let section = make_section();
        let mut bytes = section.serialize().unwrap();
        bytes[0] = CATALOG_SECTION_VERSION + 1;

        let mut section2 = make_section();
        let err = section2
            .deserialize(&bytes)
            .expect_err("foreign version byte must fail");
        let msg = err.to_string();
        assert!(
            msg.contains("unsupported catalog version 3 (supported 1, 2)"),
            "error must name found and supported versions, got: {msg}"
        );
        assert_eq!(
            err.error_code().as_str(),
            "GRAFEO-X002",
            "version rejection is a serialization-class error, distinct \
             from the GRAFEO-X001 section CRC mismatch"
        );
    }

    #[test]
    fn catalog_deserialize_reads_v1_payload() {
        // Serialize-side mirror of the v1 snapshot shape. An empty `Vec`
        // encodes as a bare zero length under bincode's standard config,
        // so the element type of the empty vector-index list is
        // irrelevant to the bytes produced.
        #[derive(Serialize)]
        struct V1Indexes {
            property_indexes: Vec<String>,
            vector_indexes: Vec<u8>,
            text_indexes: Vec<SnapshotTextIndex>,
        }
        #[derive(Serialize)]
        struct V1Payload {
            version: u8,
            schema: SnapshotSchema,
            indexes: V1Indexes,
            epoch: u64,
        }

        let mut schema = SnapshotSchema::default();
        schema.node_types.push(NodeTypeDefinition {
            name: "Person".to_string(),
            properties: vec![],
            constraints: vec![],
            parent_types: vec![],
        });
        let payload = V1Payload {
            version: CATALOG_SECTION_VERSION_V1,
            schema,
            indexes: V1Indexes {
                property_indexes: vec!["Person:name".to_string()],
                vector_indexes: Vec::new(),
                text_indexes: Vec::new(),
            },
            epoch: 7,
        };
        let bytes =
            bincode::serde::encode_to_vec(&payload, bincode::config::standard()).unwrap();
        assert_eq!(bytes[0], CATALOG_SECTION_VERSION_V1);

        let mut section = make_section();
        section
            .deserialize(&bytes)
            .expect("v1 payload must keep loading");
        assert_eq!(section.catalog.all_node_type_defs().len(), 1);
        assert_eq!(
            section.restored_property_indexes(),
            vec!["Person:name".to_string()]
        );
        assert!(
            section.restored_vector_indexes().is_empty(),
            "v1 carries no restorable vector index metadata"
        );
    }

    #[test]
    fn catalog_deserialize_rejects_empty_payload() {
        let mut section = make_section();
        let err = section
            .deserialize(&[])
            .expect_err("empty payload must fail");
        assert!(
            err.to_string().contains("missing format version byte"),
            "unexpected error: {err}"
        );
    }
}
