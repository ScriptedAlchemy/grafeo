//! Allocation probe for transactional mutation batches.
//!
//! Mirrors the batch shape TraceDecay's graph publication applies: N batches,
//! each one transaction creating `nodes` property-bearing nodes plus `edges`
//! property-bearing edges, committed through the session API with WAL on.
//! Run under a heap profiler (valgrind --tool=dhat) to attribute per-batch
//! allocation traffic to engine call sites.
//!
//!   cargo build --example batch_alloc_probe --release \
//!     --features lpg,wal,grafeo-file
//!   valgrind --tool=dhat ./target/release/examples/batch_alloc_probe

use std::time::Instant;

use grafeo_common::types::Value;
use grafeo_engine::{Config, GrafeoDB};

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn main() {
    let batches = env_usize("PROBE_BATCHES", 8);
    let nodes = env_usize("PROBE_NODES", 2_000);
    let edges = env_usize("PROBE_EDGES", 2_000);

    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("probe.grafeo");
    let db = GrafeoDB::with_config(Config::persistent(&path)).expect("open database");

    for batch in 0..batches {
        let started = Instant::now();
        let mut session = db.session();
        session.begin_transaction().expect("begin transaction");
        let mut node_ids = Vec::with_capacity(nodes);
        for i in 0..nodes {
            let key = format!("entity:{batch}:{i}");
            let id = session
                .create_node_with_props(
                    &["Entity", "Generation"],
                    [
                        ("key", Value::String(key.into())),
                        ("kind", Value::String("symbol".into())),
                        (
                            "path",
                            Value::String(
                                format!("crates/example/src/module_{i}/file.rs").into(),
                            ),
                        ),
                        (
                            "sequence",
                            Value::Int64(i64::try_from(i).expect("probe sizes fit i64")),
                        ),
                        (
                            "payload",
                            Value::String("x".repeat(96).into()),
                        ),
                    ],
                )
                .expect("create node");
            node_ids.push(id);
        }
        for i in 0..edges {
            let from = node_ids[i % node_ids.len()];
            let to = node_ids[(i * 7 + 1) % node_ids.len()];
            session
                .create_edge_with_props(
                    from,
                    to,
                    "RELATES",
                    [
                        ("key", Value::String(format!("relation:{batch}:{i}").into())),
                        (
                            "ordinal",
                            Value::Int64(i64::try_from(i).expect("probe sizes fit i64")),
                        ),
                    ],
                )
                .expect("create edge");
        }
        session.commit().expect("commit batch");
        println!(
            "batch,{batch},elapsed_ms,{:.1}",
            started.elapsed().as_secs_f64() * 1_000.0
        );
    }
    db.close().expect("close database");
    println!("done");
}
