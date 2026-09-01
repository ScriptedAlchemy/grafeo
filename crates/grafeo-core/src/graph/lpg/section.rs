//! LPG section serializer for the `.grafeo` container format.
//!
//! Implements the [`Section`] trait for LPG graph data (nodes, edges,
//! properties, named graphs). Uses the block-based binary format (v2)
//! defined in the `block` submodule for efficient serialization, CRC integrity
//! checking, and future mmap support.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use grafeo_common::storage::section::{Section, SectionType};
use grafeo_common::types::{EdgeId, EpochId, NodeId};
use grafeo_common::utils::error::{Error, Result};

use super::block::{self, BlockEdge, BlockNode, BlockSource};
use crate::graph::lpg::LpgStore;

/// Current LPG section format version (v2 = block-based).
const LPG_SECTION_VERSION: u8 = 2;

// ── Streaming source ────────────────────────────────────────────────

struct LpgStoreBlockSource {
    store: Arc<LpgStore>,
    node_ids: Vec<NodeId>,
    edge_ids: Vec<EdgeId>,
    named_graphs: Vec<(String, Arc<LpgStore>)>,
}

impl LpgStoreBlockSource {
    fn new(store: Arc<LpgStore>) -> Self {
        let node_ids = store.node_ids();
        let mut edge_ids = store.visible_edge_ids();
        edge_ids.sort_unstable();
        let named_graphs = store
            .graph_names()
            .into_iter()
            .filter_map(|name| store.graph(&name).map(|graph| (name, graph)))
            .collect();
        Self {
            store,
            node_ids,
            edge_ids,
            named_graphs,
        }
    }

    /// Refills `out` with the node's current labels and properties.
    ///
    /// Reads labels and property columns directly (shared `ArcStr` handles,
    /// no per-node property map), so the caller can reuse one scratch
    /// [`BlockNode`] across the whole visit.
    fn materialize_node_into(&self, id: NodeId, out: &mut BlockNode) -> Result<()> {
        if !self.store.node_visible(id) {
            return Err(Error::Internal(
                "LPG node disappeared while checkpointing".to_owned(),
            ));
        }
        out.id = id;
        out.labels.clear();
        out.properties.clear();

        self.store.collect_node_labels(id, &mut out.labels);
        out.labels.sort_unstable();

        #[cfg(feature = "temporal")]
        out.properties.extend(
            self.store
                .node_property_history(id)
                .into_iter()
                .map(|(key, entries)| (key, entries.into())),
        );

        #[cfg(not(feature = "temporal"))]
        self.store.for_each_node_property(id, |key, value| {
            out.properties
                .push((key.clone(), smallvec::smallvec![(EpochId::new(0), value)]));
        });

        out.properties
            .sort_by(|(left, _), (right, _)| left.cmp(right));
        Ok(())
    }

    /// Edge counterpart of [`Self::materialize_node_into`].
    fn materialize_edge_into(&self, id: EdgeId, out: &mut BlockEdge) -> Result<()> {
        let (src, dst, edge_type) = self.store.edge_header(id).ok_or_else(|| {
            Error::Internal("LPG edge disappeared while checkpointing".to_owned())
        })?;
        out.id = id;
        out.src = src;
        out.dst = dst;
        out.edge_type = edge_type;
        out.properties.clear();

        #[cfg(feature = "temporal")]
        out.properties.extend(
            self.store
                .edge_property_history(id)
                .into_iter()
                .map(|(key, entries)| (key, entries.into())),
        );

        #[cfg(not(feature = "temporal"))]
        self.store.for_each_edge_property(id, |key, value| {
            out.properties
                .push((key.clone(), smallvec::smallvec![(EpochId::new(0), value)]));
        });

        out.properties
            .sort_by(|(left, _), (right, _)| left.cmp(right));
        Ok(())
    }
}

impl BlockSource for LpgStoreBlockSource {
    fn node_count(&self) -> usize {
        self.node_ids.len()
    }

    fn edge_count(&self) -> usize {
        self.edge_ids.len()
    }

    fn named_graph_count(&self) -> usize {
        self.named_graphs.len()
    }

    fn visit_nodes(&self, visitor: &mut dyn FnMut(&BlockNode) -> Result<()>) -> Result<()> {
        let mut scratch = BlockNode {
            id: NodeId::new(0),
            labels: Vec::new(),
            properties: Vec::new(),
        };
        for id in &self.node_ids {
            self.materialize_node_into(*id, &mut scratch)?;
            visitor(&scratch)?;
        }
        Ok(())
    }

    fn visit_edges(&self, visitor: &mut dyn FnMut(&BlockEdge) -> Result<()>) -> Result<()> {
        let mut scratch = BlockEdge {
            id: EdgeId::new(0),
            src: NodeId::new(0),
            dst: NodeId::new(0),
            edge_type: arcstr::ArcStr::new(),
            properties: Vec::new(),
        };
        for id in &self.edge_ids {
            self.materialize_edge_into(*id, &mut scratch)?;
            visitor(&scratch)?;
        }
        Ok(())
    }

    fn visit_named_graphs(
        &self,
        visitor: &mut dyn FnMut(&str, &dyn BlockSource) -> Result<()>,
    ) -> Result<()> {
        for (name, store) in &self.named_graphs {
            let source = Self::new(Arc::clone(store));
            visitor(name, &source)?;
        }
        Ok(())
    }
}

fn populate_store(store: &LpgStore, nodes: &[BlockNode], edges: &[BlockEdge]) -> Result<()> {
    for node in nodes {
        let label_refs: Vec<&str> = node.labels.iter().map(|s| s.as_str()).collect();
        store.create_node_with_id(node.id, &label_refs)?;
        for (key, entries) in &node.properties {
            #[cfg(feature = "temporal")]
            for (epoch, value) in entries {
                store.set_node_property_at_epoch(node.id, key.as_str(), value.clone(), *epoch);
            }
            #[cfg(not(feature = "temporal"))]
            if let Some((_, value)) = entries.last() {
                store.set_node_property_prekeyed(node.id, key.clone(), value.clone());
            }
        }
    }
    for edge in edges {
        store.create_edge_with_id(edge.id, edge.src, edge.dst, &edge.edge_type)?;
        for (key, entries) in &edge.properties {
            #[cfg(feature = "temporal")]
            for (epoch, value) in entries {
                store.set_edge_property_at_epoch(edge.id, key.as_str(), value.clone(), *epoch);
            }
            #[cfg(not(feature = "temporal"))]
            if let Some((_, value)) = entries.last() {
                store.set_edge_property_prekeyed(edge.id, key.clone(), value.clone());
            }
        }
    }
    Ok(())
}

// ── Section implementation ──────────────────────────────────────────

/// LPG store section for the `.grafeo` container.
///
/// Wraps an `Arc<LpgStore>` and implements the [`Section`] trait for
/// serialization/deserialization of LPG graph data using the block-based
/// format (v2).
pub struct LpgStoreSection {
    store: Arc<LpgStore>,
    dirty: AtomicBool,
}

impl LpgStoreSection {
    /// Create a new LPG section wrapping the given store.
    pub fn new(store: Arc<LpgStore>) -> Self {
        Self {
            store,
            dirty: AtomicBool::new(false),
        }
    }

    /// Mark this section as dirty (has unsaved changes).
    pub fn mark_dirty(&self) {
        self.dirty.store(true, Ordering::Release);
    }

    /// Access the underlying store.
    #[must_use]
    pub fn store(&self) -> &Arc<LpgStore> {
        &self.store
    }
}

impl Section for LpgStoreSection {
    fn section_type(&self) -> SectionType {
        SectionType::LpgStore
    }

    fn version(&self) -> u8 {
        LPG_SECTION_VERSION
    }

    fn serialize(&self) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        self.serialize_into(&mut out)?;
        Ok(out)
    }

    /// Streams the block-format encoding straight into `sink`.
    ///
    /// Both entry points share this encoder, so the `Vec` and sink forms
    /// cannot produce different bytes.
    fn serialize_into(&self, sink: &mut dyn std::io::Write) -> Result<()> {
        let source = LpgStoreBlockSource::new(Arc::clone(&self.store));

        #[cfg(feature = "temporal")]
        let epoch = self.store.current_epoch().as_u64();
        #[cfg(not(feature = "temporal"))]
        let epoch = 0u64;

        block::write_source_blocks_into(sink, &source, epoch)
    }

    fn deserialize(&mut self, data: &[u8]) -> Result<()> {
        let store = &self.store;

        block::read_blocks(data, &mut |nodes, edges, named_graphs, epoch| {
            populate_store(store, &nodes, &edges)?;

            #[cfg(feature = "temporal")]
            store.sync_epoch(EpochId::new(epoch));
            #[cfg(not(feature = "temporal"))]
            let _ = epoch;

            for graph in &named_graphs {
                store
                    .create_graph(&graph.name)
                    .map_err(|e| grafeo_common::utils::error::Error::Internal(e.to_string()))?;
                if let Some(graph_store) = store.graph(&graph.name) {
                    populate_store(&graph_store, &graph.nodes, &graph.edges)?;
                    #[cfg(feature = "temporal")]
                    graph_store.sync_epoch(EpochId::new(epoch));
                }
            }

            Ok(())
        })
    }

    fn is_dirty(&self) -> bool {
        self.dirty.load(Ordering::Acquire)
    }

    fn mark_clean(&self) {
        self.dirty.store(false, Ordering::Release);
    }

    fn memory_usage(&self) -> usize {
        let (store, indexes, mvcc, string_pool) = self.store.memory_breakdown();
        store.total_bytes + indexes.total_bytes + mvcc.total_bytes + string_pool.total_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use grafeo_common::types::{NodeId, PropertyKey, Value};

    #[test]
    fn lpg_section_round_trip() {
        let store = Arc::new(LpgStore::new().unwrap());
        store.create_node(&["Person"]);
        store.create_node(&["Person"]);
        let n1 = NodeId::new(1);
        let n2 = NodeId::new(2);
        store.set_node_property(n1, "name", Value::String("Alix".into()));
        store.set_node_property(n2, "name", Value::String("Gus".into()));
        store.create_edge(n1, n2, "KNOWS");

        let section = LpgStoreSection::new(Arc::clone(&store));
        let bytes = section.serialize().expect("serialize should succeed");
        assert!(!bytes.is_empty());
        assert!(block::is_block_format(&bytes));

        // Deserialize into a fresh store
        let store2 = Arc::new(LpgStore::new().unwrap());
        let mut section2 = LpgStoreSection::new(store2);
        section2
            .deserialize(&bytes)
            .expect("deserialize should succeed");

        assert_eq!(section2.store().node_count(), 2);
        assert_eq!(section2.store().edge_count(), 1);
    }

    /// The sink path and the `Vec` path must produce identical bytes:
    /// a container written through either one has to be readable by the
    /// same deserializer, and a divergence would only show up as a CRC
    /// failure on a much later open.
    #[test]
    fn lpg_section_sink_and_vec_paths_are_byte_identical() {
        let store = Arc::new(LpgStore::new().unwrap());
        // Enough shape to exercise every block kind: labels, node and
        // edge property columns, string interning, and a named graph.
        for i in 0..256 {
            let id = store.create_node(&["Person", "Indexed"]);
            store.set_node_property(id, "name", Value::String(arcstr::format!("person-{i}")));
            store.set_node_property(id, "age", Value::Int64(i));
            store.set_node_property(id, "active", Value::Bool(i % 2 == 0));
        }
        for i in 1..256u64 {
            let e = store.create_edge(NodeId::new(i), NodeId::new(i + 1), "KNOWS");
            store.set_edge_property(e, "weight", Value::Float64(i as f64));
        }
        store.create_graph("social").unwrap();
        if let Some(g) = store.graph("social") {
            let n = g.create_node(&["Friend"]);
            g.set_node_property(n, "nick", Value::String("gus".into()));
        }

        let section = LpgStoreSection::new(Arc::clone(&store));
        let via_vec = section.serialize().expect("serialize");
        let mut via_sink = Vec::new();
        section
            .serialize_into(&mut via_sink)
            .expect("serialize_into");

        assert_eq!(
            via_sink.len(),
            via_vec.len(),
            "sink and Vec paths must write the same number of bytes"
        );
        assert!(
            via_sink == via_vec,
            "sink and Vec paths must write identical bytes"
        );

        // And the sink bytes are actually loadable.
        let reloaded = Arc::new(LpgStore::new().unwrap());
        let mut section2 = LpgStoreSection::new(Arc::clone(&reloaded));
        section2.deserialize(&via_sink).expect("deserialize");
        assert_eq!(reloaded.node_count(), 256);
        assert_eq!(reloaded.edge_count(), 255);
    }

    #[test]
    fn live_store_stream_matches_the_canonical_block_encoding() {
        let store = Arc::new(LpgStore::new().unwrap());
        let first = store.create_node(&["Person", "Indexed"]);
        let second = store.create_node(&["Person"]);
        store.set_node_property(first, "name", Value::String("alix".into()));
        store.set_node_property(first, "active", Value::Bool(true));
        let edge = store.create_edge(first, second, "KNOWS");
        store.set_edge_property(edge, "weight", Value::Float64(0.75));
        store.create_graph("social").unwrap();
        let named_store = store.graph("social").unwrap();
        let friend = named_store.create_node(&["Friend"]);
        named_store.set_node_property(friend, "nick", Value::String("gus".into()));

        let streamed = LpgStoreSection::new(store).serialize().unwrap();
        let canonical = block::write_blocks(
            &[
                BlockNode {
                    id: first,
                    labels: vec!["Indexed".into(), "Person".into()],
                    properties: vec![
                        (
                            "active".into(),
                            smallvec::smallvec![(EpochId::new(0), Value::Bool(true))],
                        ),
                        (
                            "name".into(),
                            smallvec::smallvec![(EpochId::new(0), Value::String("alix".into()))],
                        ),
                    ],
                },
                BlockNode {
                    id: second,
                    labels: vec!["Person".into()],
                    properties: vec![],
                },
            ],
            &[BlockEdge {
                id: edge,
                src: first,
                dst: second,
                edge_type: "KNOWS".into(),
                properties: vec![(
                    "weight".into(),
                    smallvec::smallvec![(EpochId::new(0), Value::Float64(0.75))],
                )],
            }],
            &[block::BlockNamedGraph {
                name: "social".into(),
                nodes: vec![BlockNode {
                    id: friend,
                    labels: vec!["Friend".into()],
                    properties: vec![(
                        "nick".into(),
                        smallvec::smallvec![(EpochId::new(0), Value::String("gus".into()))],
                    )],
                }],
                edges: vec![],
            }],
            0,
        )
        .unwrap();

        assert_eq!(streamed, canonical);
    }

    #[test]
    fn lpg_section_dirty_tracking() {
        let store = Arc::new(LpgStore::new().unwrap());
        let section = LpgStoreSection::new(store);

        assert!(!section.is_dirty());
        section.mark_dirty();
        assert!(section.is_dirty());
        section.mark_clean();
        assert!(!section.is_dirty());
    }

    #[test]
    fn lpg_section_type() {
        let store = Arc::new(LpgStore::new().unwrap());
        let section = LpgStoreSection::new(store);
        assert_eq!(section.section_type(), SectionType::LpgStore);
        assert_eq!(section.version(), LPG_SECTION_VERSION);
    }

    #[test]
    fn lpg_section_empty_round_trip() {
        let store = Arc::new(LpgStore::new().unwrap());
        let section = LpgStoreSection::new(Arc::clone(&store));
        let bytes = section.serialize().unwrap();

        let store2 = Arc::new(LpgStore::new().unwrap());
        let mut section2 = LpgStoreSection::new(store2);
        section2.deserialize(&bytes).unwrap();
        assert_eq!(section2.store().node_count(), 0);
        assert_eq!(section2.store().edge_count(), 0);
    }

    #[test]
    fn lpg_section_properties_preserved() {
        let store = Arc::new(LpgStore::new().unwrap());
        let n = store.create_node(&["Person"]);
        store.set_node_property(n, "name", Value::String("Alix".into()));
        store.set_node_property(n, "age", Value::Int64(30));
        store.set_node_property(n, "active", Value::Bool(true));

        let section = LpgStoreSection::new(Arc::clone(&store));
        let bytes = section.serialize().unwrap();

        let store2 = Arc::new(LpgStore::new().unwrap());
        let mut section2 = LpgStoreSection::new(Arc::clone(&store2));
        section2.deserialize(&bytes).unwrap();

        let node = store2.get_node(n).unwrap();
        let name_key: PropertyKey = "name".into();
        let age_key: PropertyKey = "age".into();
        let active_key: PropertyKey = "active".into();
        assert_eq!(
            node.properties.get(&name_key),
            Some(&Value::String("Alix".into()))
        );
        assert_eq!(node.properties.get(&age_key), Some(&Value::Int64(30)));
        assert_eq!(node.properties.get(&active_key), Some(&Value::Bool(true)));
    }

    #[test]
    fn lpg_section_named_graphs() {
        let store = Arc::new(LpgStore::new().unwrap());
        store.create_node(&["Root"]);
        store.create_graph("social").unwrap();

        if let Some(g) = store.graph("social") {
            g.create_node(&["Friend"]);
        }

        let section = LpgStoreSection::new(Arc::clone(&store));
        let bytes = section.serialize().unwrap();

        let store2 = Arc::new(LpgStore::new().unwrap());
        let mut section2 = LpgStoreSection::new(Arc::clone(&store2));
        section2.deserialize(&bytes).unwrap();

        assert_eq!(store2.node_count(), 1);
        assert!(store2.graph("social").is_some());
        assert_eq!(store2.graph("social").unwrap().node_count(), 1);
    }

    #[test]
    fn lpg_section_crc_integrity() {
        let store = Arc::new(LpgStore::new().unwrap());
        store.create_node(&["Test"]);

        let section = LpgStoreSection::new(Arc::clone(&store));
        let mut bytes = section.serialize().unwrap();

        // Corrupt a byte
        let last = bytes.len() - 1;
        bytes[last] ^= 0xFF;

        let store2 = Arc::new(LpgStore::new().unwrap());
        let mut section2 = LpgStoreSection::new(store2);
        assert!(section2.deserialize(&bytes).is_err());
    }
}
