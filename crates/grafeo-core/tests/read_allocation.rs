//! Allocation metadata must describe the same rows that the native readers serve.

#![cfg(feature = "lpg")]

use grafeo_common::types::{EdgeId, NodeId, PropertyKey, Value};
use grafeo_core::graph::lpg::LpgStore;
use grafeo_core::graph::traits::GraphStore;

fn fixture() -> (LpgStore, [NodeId; 3], EdgeId, Value, Value) {
    let store = LpgStore::new().unwrap();
    let nodes = std::array::from_fn(|_| store.create_node(&["Indexed"]));
    let text = Value::from("exact tool output ".repeat(1024));
    let bytes = Value::from(vec![0xab_u8; 4096]);
    for (index, node) in nodes.iter().enumerate() {
        store.set_node_property(*node, "payload", text.clone());
        store.set_node_property(*node, "bytes", bytes.clone());
        store.set_node_property(*node, "score", Value::Int64(if index < 2 { 7 } else { 11 }));
        for ordinal in 0..12 {
            store.set_node_property(*node, &format!("field_{ordinal}"), Value::Int64(ordinal));
        }
    }
    let edge = store.create_edge(nodes[0], nodes[1], "LINK");
    store.set_edge_property(edge, "payload", text.clone());
    store.set_edge_property(edge, "bytes", bytes.clone());
    (store, nodes, edge, text, bytes)
}

fn assert_rows(
    store: &dyn GraphStore,
    node_id: NodeId,
    edge_id: EdgeId,
    text: &Value,
    bytes: &Value,
) {
    let text_len = text.as_str().unwrap().len();
    assert!(store.node_allocation_bytes(node_id).unwrap() >= text_len);
    assert!(store.edge_allocation_bytes(edge_id).unwrap() >= text_len);
    assert!(
        store
            .node_property_allocation_bytes(node_id, &PropertyKey::new("payload"))
            .unwrap()
            >= text_len
    );
    let node = store.get_node(node_id).unwrap();
    assert_eq!(node.get_property("payload"), Some(text));
    assert_eq!(node.get_property("bytes"), Some(bytes));
    assert_eq!(node.properties.len(), 15);
    let edge = store.get_edge(edge_id).unwrap();
    assert_eq!(edge.get_property("payload"), Some(text));
    assert_eq!(edge.get_property("bytes"), Some(bytes));
    assert_eq!(edge.properties.len(), 2);
    assert!(store.label_allocation_bytes().unwrap() >= "Indexed".len());
    assert_eq!(store.all_labels(), ["Indexed"]);
    assert_eq!(store.node_allocation_bytes(NodeId::INVALID), Some(0));
    assert_eq!(store.edge_allocation_bytes(EdgeId::INVALID), Some(0));
}

#[test]
fn lpg_metadata_and_direct_materialization_preserve_rows() {
    let (store, nodes, edge, text, bytes) = fixture();
    assert_rows(&store, nodes[0], edge, &text, &bytes);
    let key = PropertyKey::new("score");
    assert_eq!(
        store.property_lookup_allocation_bytes(&key, &Value::Int64(7)),
        None
    );
    store.create_property_index("score");
    let demand = store
        .property_lookup_allocation_bytes(&key, &Value::Int64(7))
        .unwrap();
    let mut matches = store.find_nodes_by_property("score", &Value::Int64(7));
    matches.sort_unstable();
    assert_eq!(matches, nodes[..2]);
    assert!(demand >= matches.len() * std::mem::size_of::<NodeId>());
    assert_eq!(
        store.property_lookup_allocation_bytes(&key, &Value::Int64(99)),
        Some(0)
    );
}

#[cfg(feature = "compact-store")]
#[test]
fn compact_metadata_counts_decoding_and_index_matches() {
    use grafeo_core::graph::compact::from_graph_store_preserving_ids;

    let (store, nodes, edge, text, bytes) = fixture();
    let compact = from_graph_store_preserving_ids(&store).unwrap();
    assert_rows(&compact, nodes[0], edge, &text, &bytes);
    let score = PropertyKey::new("score");
    assert_eq!(
        compact.property_lookup_allocation_bytes(&score, &Value::Int64(7)),
        None
    );
    compact.enable_property_indexes([score.clone(), PropertyKey::new("payload")]);
    let matching = compact
        .property_lookup_allocation_bytes(&score, &Value::Int64(7))
        .unwrap();
    let absent = compact
        .property_lookup_allocation_bytes(&score, &Value::Int64(99))
        .unwrap();
    let mut matches = compact.find_nodes_by_property("score", &Value::Int64(7));
    matches.sort_unstable();
    assert_eq!(matches, nodes[..2]);
    assert!(matching >= absent + matches.len() * std::mem::size_of::<NodeId>());
    let payload = PropertyKey::new("payload");
    assert!(
        compact
            .property_lookup_allocation_bytes(&payload, &text)
            .unwrap()
            >= text.as_str().unwrap().len()
    );
    assert_eq!(compact.find_nodes_by_property("payload", &text).len(), 3);
}

#[cfg(feature = "compact-store")]
#[test]
fn layered_metadata_covers_base_and_overlay_reads() {
    use grafeo_core::graph::compact::{from_graph_store_preserving_ids, layered::LayeredStore};
    use grafeo_core::graph::traits::GraphStoreMut;

    let (store, nodes, edge, text, bytes) = fixture();
    let compact = from_graph_store_preserving_ids(&store).unwrap();
    compact.enable_property_indexes([PropertyKey::new("score")]);
    let layered = LayeredStore::new(compact, nodes[2].as_u64(), edge.as_u64()).unwrap();
    layered.overlay_store().create_property_index("score");
    let base_demand = layered.node_allocation_bytes(nodes[0]).unwrap();
    assert_rows(&layered, nodes[0], edge, &text, &bytes);
    let changed_text = Value::from("updated exact output ".repeat(2048));
    layered.set_node_property(nodes[0], "payload", changed_text.clone());
    assert!(layered.node_allocation_bytes(nodes[0]).unwrap() > base_demand);
    assert_eq!(
        layered.get_node(nodes[0]).unwrap().get_property("payload"),
        Some(&changed_text)
    );
    let overlay = layered.create_node(&["Indexed"]);
    layered.set_node_property(overlay, "score", Value::Int64(7));
    let mut matches = layered.find_nodes_by_property("score", &Value::Int64(7));
    matches.sort_unstable();
    assert_eq!(matches, [nodes[0], nodes[1], overlay]);
    assert!(
        layered
            .property_lookup_allocation_bytes(&PropertyKey::new("score"), &Value::Int64(7))
            .unwrap()
            >= matches.len() * std::mem::size_of::<NodeId>()
    );
}

#[cfg(feature = "temporal")]
#[test]
fn direct_materialization_keeps_epoch_property_visibility() {
    let (store, nodes, edge, text, _) = fixture();
    let before = store.current_epoch();
    let after = store.new_epoch();
    let replacement = Value::from("replacement");
    store.set_node_property(nodes[0], "payload", replacement.clone());
    store.set_edge_property(edge, "payload", replacement.clone());
    assert_eq!(
        store
            .get_node_at_epoch(nodes[0], before)
            .unwrap()
            .get_property("payload"),
        Some(&text)
    );
    assert_eq!(
        store
            .get_edge_at_epoch(edge, before)
            .unwrap()
            .get_property("payload"),
        Some(&text)
    );
    assert_eq!(
        store
            .get_node_at_epoch(nodes[0], after)
            .unwrap()
            .get_property("payload"),
        Some(&replacement)
    );
    assert_eq!(
        store
            .get_edge_at_epoch(edge, after)
            .unwrap()
            .get_property("payload"),
        Some(&replacement)
    );
}
