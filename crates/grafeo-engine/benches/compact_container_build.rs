//! Cost of producing a compacted single-file database from a row stream.
//!
//! Two paths build the same rows and leave the same kind of container:
//!
//! - `direct`: `IncrementalCompactStoreBuilder` → `GrafeoDB::write_compact_container`
//! - `lpg`: persistent `GrafeoDB` (WAL off) → row inserts → `compact()` → `close()`
//!
//! Each path runs in its own child process so its peak RSS (`VmHWM`) is its
//! own, and reports wall time, CPU time (user + system, all threads) and
//! peak RSS. Numbers print as one JSON line per path.
//!
//! ```bash
//! cargo bench -p grafeo-engine --features "lpg,gql,grafeo-file,compact-store,mmap" \
//!     --bench compact_container_build -- [--nodes N] [--edges N] [--paths direct,lpg]
//! ```
//!
//! Defaults to 1.1 M nodes and 1.1 M edges (2.2 M rows), the shape of a
//! large code graph: string-heavy node properties with a 200-byte binary
//! payload, integer-and-string edge properties.

// reason: bench indices are small, known values.
#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
#![allow(missing_docs)]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use grafeo_common::types::{EdgeId, NodeId, PropertyKey, Value};
use grafeo_core::graph::compact::IncrementalCompactStoreBuilder;
use grafeo_engine::{Config, GrafeoDB};

const DEFAULT_NODES: u64 = 1_100_000;
const DEFAULT_EDGES: u64 = 1_100_000;
const NODE_LABEL: &str = "entity";
const EDGE_TYPE: &str = "relation";

struct Args {
    nodes: u64,
    edges: u64,
    paths: Vec<String>,
    phase: Option<String>,
}

fn parse_args() -> Args {
    let mut args = Args {
        nodes: DEFAULT_NODES,
        edges: DEFAULT_EDGES,
        paths: vec!["direct".to_owned(), "lpg".to_owned()],
        phase: None,
    };
    let mut iter = std::env::args().skip(1);
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--nodes" => {
                args.nodes = iter
                    .next()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(args.nodes);
            }
            "--edges" => {
                args.edges = iter
                    .next()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(args.edges);
            }
            "--paths" => {
                if let Some(list) = iter.next() {
                    args.paths = list.split(',').map(str::to_owned).collect();
                }
            }
            "--phase" => args.phase = iter.next(),
            // cargo bench passes `--bench`; ignore it and anything unknown.
            _ => {}
        }
    }
    args
}

/// `(node id, labels, properties)` for row `index`: a symbol-like entity.
fn node_row(index: u64) -> (NodeId, Vec<(PropertyKey, Value)>) {
    let properties = vec![
        (
            PropertyKey::new("identity"),
            Value::from(format!(
                "symbol:crate/module_{}/item_{index:07}",
                index % 4_900
            )),
        ),
        (
            PropertyKey::new("kind"),
            Value::from(["function", "struct", "method", "field"][(index % 4) as usize]),
        ),
        (
            PropertyKey::new("name"),
            Value::from(format!("item_{index:07}")),
        ),
        (
            PropertyKey::new("file"),
            Value::from(format!("crates/module_{}/src/lib.rs", index % 4_900)),
        ),
        (
            PropertyKey::new("line"),
            Value::Int64((index % 3_000).cast_signed()),
        ),
        (
            PropertyKey::new("payload"),
            Value::Bytes(vec![(index % 251) as u8; 200].into()),
        ),
    ];
    (NodeId(index + 1), properties)
}

/// `(edge id, src, dst, properties)` for row `index`: a call-like relation.
fn edge_row(index: u64, nodes: u64) -> (EdgeId, NodeId, NodeId, Vec<(PropertyKey, Value)>) {
    let src = NodeId((index % nodes) + 1);
    let dst = NodeId(((index * 7 + 13) % nodes) + 1);
    let properties = vec![
        (
            PropertyKey::new("identity"),
            Value::from(format!("call:{index:07}")),
        ),
        (
            PropertyKey::new("kind"),
            Value::from(["calls", "references", "contains"][(index % 3) as usize]),
        ),
        (
            PropertyKey::new("weight"),
            Value::Int64((index % 17).cast_signed()),
        ),
    ];
    (EdgeId(index + 1), src, dst, properties)
}

fn run_direct(path: &Path, nodes: u64, edges: u64) -> (u64, u64) {
    let mut builder = IncrementalCompactStoreBuilder::new();
    for index in 0..nodes {
        let (id, properties) = node_row(index);
        builder
            .push_node(id, [NODE_LABEL], properties.iter().map(|(k, v)| (k, v)))
            .unwrap();
    }
    for index in 0..edges {
        let (id, src, dst, properties) = edge_row(index, nodes);
        builder
            .push_edge(
                id,
                EDGE_TYPE,
                src,
                dst,
                properties.iter().map(|(k, v)| (k, v)),
            )
            .unwrap();
    }
    GrafeoDB::write_compact_container(path, builder, ["identity".to_owned()]).unwrap();
    reopen_counts(path)
}

fn run_lpg(path: &Path, nodes: u64, edges: u64) -> (u64, u64) {
    let mut db = GrafeoDB::with_config(Config {
        wal_enabled: false,
        ..Config::persistent(path)
    })
    .unwrap();
    db.create_property_index("identity");
    {
        let store = db.graph_store_mut().expect("writable store");
        // The LPG allocates its own ids; map the fixture's ids onto them.
        let mut ids = Vec::with_capacity(nodes as usize);
        for index in 0..nodes {
            let (_, properties) = node_row(index);
            let id = store.create_node(&[NODE_LABEL]);
            for (key, value) in properties {
                store.set_node_property(id, key.as_str(), value);
            }
            ids.push(id);
        }
        for index in 0..edges {
            let (_, src, dst, properties) = edge_row(index, nodes);
            let id = store.create_edge(
                ids[(src.0 - 1) as usize],
                ids[(dst.0 - 1) as usize],
                EDGE_TYPE,
            );
            for (key, value) in properties {
                store.set_edge_property(id, key.as_str(), value);
            }
        }
    }
    db.compact().unwrap();
    db.close().unwrap();
    reopen_counts(path)
}

fn reopen_counts(path: &Path) -> (u64, u64) {
    let db = GrafeoDB::with_config(Config::read_only(path)).unwrap();
    let store = db.graph_store();
    (store.node_count() as u64, store.edge_count() as u64)
}

/// `(user + system CPU seconds, peak RSS bytes)` of this process.
fn process_costs() -> (f64, u64) {
    let stat = std::fs::read_to_string("/proc/self/stat").unwrap_or_default();
    // Fields after the command name, which may contain spaces.
    let after = stat.rsplit(')').next().unwrap_or("");
    let fields: Vec<&str> = after.split_whitespace().collect();
    let ticks: f64 = fields
        .get(11)
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(0.0)
        + fields
            .get(12)
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(0.0);
    let cpu = ticks / 100.0;
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let peak_kib: u64 = status
        .lines()
        .find_map(|line| line.strip_prefix("VmHWM:"))
        .and_then(|rest| rest.trim().trim_end_matches("kB").trim().parse().ok())
        .unwrap_or(0);
    (cpu, peak_kib * 1024)
}

fn run_phase(phase: &str, nodes: u64, edges: u64) {
    let dir = tempfile::TempDir::new().unwrap();
    let path: PathBuf = dir.path().join(format!("{phase}.grafeo"));
    let started = Instant::now();
    let (node_count, edge_count) = match phase {
        "direct" => run_direct(&path, nodes, edges),
        "lpg" => run_lpg(&path, nodes, edges),
        other => panic!("unknown phase {other}"),
    };
    let wall = started.elapsed().as_secs_f64();
    let (cpu, peak_rss) = process_costs();
    let file_bytes = std::fs::metadata(&path).map_or(0, |m| m.len());
    assert_eq!(
        (node_count, edge_count),
        (nodes, edges),
        "{phase} lost rows"
    );
    println!(
        "{{\"path\":\"{phase}\",\"nodes\":{nodes},\"edges\":{edges},\"wall_s\":{wall:.2},\"cpu_s\":{cpu:.2},\"peak_rss_bytes\":{peak_rss},\"file_bytes\":{file_bytes}}}"
    );
}

fn main() {
    let args = parse_args();
    if let Some(phase) = args.phase {
        run_phase(&phase, args.nodes, args.edges);
        return;
    }
    let exe = std::env::current_exe().unwrap();
    for phase in &args.paths {
        let status = Command::new(&exe)
            .args([
                "--phase",
                phase,
                "--nodes",
                &args.nodes.to_string(),
                "--edges",
                &args.edges.to_string(),
            ])
            .status()
            .unwrap();
        assert!(status.success(), "{phase} path failed");
    }
}
