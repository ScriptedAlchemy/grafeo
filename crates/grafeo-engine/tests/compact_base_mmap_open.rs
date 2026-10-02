//! Reopening a compacted database maps its `CompactStore` section
//! instead of copying it onto the heap.
//!
//! `CompactStoreSection::deserialize_from_bytes` re-bases every column
//! codec onto slices of the `Bytes` it is handed, so handing it a
//! `Bytes::from_owner(mmap)` builds the base without copying the file at
//! all. The container-open path used to call the `&[u8]` entry point over
//! a heap `Vec` read from the file, paying two copies for nothing.
//!
//! The mapping is shared, so it must be released before anything rewrites
//! the container. These tests cover both halves: that reads work through
//! the mapping, and that a write cycle detaches from it without
//! corrupting the base.

#![cfg(all(
    feature = "compact-store",
    feature = "lpg",
    feature = "grafeo-file",
    feature = "gql"
))]

use std::path::Path;

use grafeo_engine::{Config, GrafeoDB};

const PEOPLE: usize = 512;

/// Creates a persistent database, compacts it so the container carries a
/// `CompactStore` section, and closes it.
fn seed_compacted_db(path: &Path) {
    let mut db = GrafeoDB::with_config(Config::persistent(path)).expect("open");
    {
        let session = db.session();
        for i in 0..PEOPLE {
            session
                .execute(&format!(
                    "INSERT (:Person {{name: 'person-{i}', age: {age}}})",
                    age = i % 90
                ))
                .expect("insert");
        }
    }
    db.compact().expect("compact");
    db.close().expect("close");
}

/// Reads every seeded name back, sorted, so callers can compare whole
/// datasets rather than spot-check.
fn read_names(db: &GrafeoDB) -> Vec<String> {
    let session = db.session();
    let result = session
        .execute("MATCH (p:Person) RETURN p.name")
        .expect("query");
    let mut names: Vec<String> = result
        .rows()
        .iter()
        .filter_map(|row| match &row[0] {
            grafeo_common::types::Value::String(s) => Some(s.to_string()),
            _ => None,
        })
        .collect();
    names.sort();
    names
}

fn expected_names() -> Vec<String> {
    let mut names: Vec<String> = (0..PEOPLE).map(|i| format!("person-{i}")).collect();
    names.sort();
    names
}

#[test]
fn reopen_serves_reads_through_the_container_mapping() {
    let dir = tempfile::TempDir::new().expect("tempdir");
    let path = dir.path().join("mmap_open.grafeo");
    seed_compacted_db(&path);

    let db = GrafeoDB::with_config(Config::persistent(&path)).expect("reopen");
    assert!(
        db.compact_base_is_mmap_backed(),
        "the compact base should open through a mapping, not a heap copy"
    );
    assert_eq!(read_names(&db), expected_names());
    db.close().expect("close");
}

/// The regression this guards: a container write rewrites the section
/// payloads the mapping points at, and `mmap_section` maps shared, so a
/// retained mapping would start serving the new bytes at stale offsets.
/// Without the detach that shows up as garbage or a CRC failure on the
/// *next* open — far from the cause.
///
/// A checkpoint is the write, and it is deliberately the whole test:
/// inserting new rows first would drag in an unrelated pre-existing
/// defect (the overlay id allocator restarts at 1 after a reload of a
/// compacted database, so the first post-reopen insert shadows base node
/// 1 — reproducible on the unmodified base revision with the heap open
/// path).
#[test]
fn write_cycle_after_a_mapped_open_leaves_the_store_intact() {
    let dir = tempfile::TempDir::new().expect("tempdir");
    let path = dir.path().join("mmap_open_then_write.grafeo");
    seed_compacted_db(&path);

    {
        let db = GrafeoDB::with_config(Config::persistent(&path)).expect("reopen");
        assert!(db.compact_base_is_mmap_backed());

        db.wal_checkpoint().expect("checkpoint");
        assert!(
            !db.compact_base_is_mmap_backed(),
            "the checkpoint must have released the container mapping"
        );
        // The base has to survive being rewritten underneath itself.
        assert_eq!(
            read_names(&db),
            expected_names(),
            "reads must still work after the container was rewritten"
        );

        db.close().expect("close");
    }

    let db = GrafeoDB::with_config(Config::persistent(&path)).expect("reopen after write");
    assert_eq!(
        read_names(&db),
        expected_names(),
        "the rewritten container must still hold the whole store"
    );
    db.close().expect("close");
}

/// A read-only open never writes, so the mapping simply lives for the
/// lifetime of the handle.
#[test]
fn read_only_open_serves_reads_through_the_mapping() {
    use grafeo_engine::config::AccessMode;

    let dir = tempfile::TempDir::new().expect("tempdir");
    let path = dir.path().join("mmap_open_ro.grafeo");
    seed_compacted_db(&path);

    let db =
        GrafeoDB::with_config(Config::persistent(&path).with_access_mode(AccessMode::ReadOnly))
            .expect("read-only reopen");
    assert!(db.compact_base_is_mmap_backed());
    assert_eq!(read_names(&db), expected_names());
    db.close().expect("close");
}

/// Repeated open/close cycles must be stable: each one maps, detaches at
/// the checkpoint, and rewrites a container the next open can map again.
#[test]
fn repeated_open_close_cycles_stay_consistent() {
    let dir = tempfile::TempDir::new().expect("tempdir");
    let path = dir.path().join("mmap_cycles.grafeo");
    seed_compacted_db(&path);

    for cycle in 0..3 {
        let db = GrafeoDB::with_config(Config::persistent(&path))
            .unwrap_or_else(|e| panic!("reopen cycle {cycle}: {e}"));
        assert_eq!(read_names(&db), expected_names(), "cycle {cycle}");
        db.close().expect("close");
    }
}

// ── Measurement probes ──────────────────────────────────────────────
//
// Both probes are `#[ignore]`d: they measure rather than assert, and
// `VmHWM` is process-wide and monotonic, so a scenario has to own its
// process. They are split into phases for the same reason — building a
// fixture in the process that then measures the open leaves the
// allocator holding a large free arena, which makes the open look free
// no matter which path it took.
//
// Open-path comparison (build once, then one fresh process per path):
//
// ```text
// export P=/tmp/probe.grafeo GRAFEO_ATREST_ROWS=2000000
// GRAFEO_ATREST_PATH=$P GRAFEO_ATREST_PHASE=build \
//   cargo test -p grafeo-engine --features compact-store \
//   --test compact_base_mmap_open -- --ignored --nocapture --exact \
//   compact_base_open_probe
// GRAFEO_ATREST_PATH=$P GRAFEO_COMPACT_BASE_MMAP=1 \
//   cargo test ... --exact compact_base_open_probe
// GRAFEO_ATREST_PATH=$P GRAFEO_COMPACT_BASE_MMAP=0 \
//   cargo test ... --exact compact_base_open_probe
// ```
//
// Publish peak (compare the same command across revisions):
//
// ```text
// GRAFEO_ATREST_ROWS=2000000 cargo test ... --exact publish_peak_probe
// ```

const PROBE_LABELS: [&str; 4] = ["Symbol", "File", "Module", "Chunk"];

fn probe_rows() -> usize {
    std::env::var("GRAFEO_ATREST_ROWS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1_000_000)
}

/// Populates `db` through the store API rather than through GQL: one
/// statement per row would make a million-row fixture take hours and
/// would measure the parser, not the storage layer.
fn build_probe_fixture(db: &GrafeoDB, rows: usize) {
    use grafeo_common::types::Value;

    let store = db.graph_store_mut().expect("mutable store");
    let mut ids = Vec::with_capacity(rows);
    for index in 0..rows {
        let node = store.create_node(&[PROBE_LABELS[index % PROBE_LABELS.len()]]);
        store.set_node_property(node, "name", Value::String(format!("sym-{index}").into()));
        // reason: probe row index stays well inside i64
        #[allow(clippy::cast_possible_wrap)]
        store.set_node_property(node, "idx", Value::Int64(index as i64));
        ids.push(node);
    }
    for window in ids.windows(2).step_by(4) {
        store.create_edge(window[0], window[1], "calls");
    }
}

fn mib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

fn kib_to_mib(kib: i64) -> f64 {
    kib as f64 / 1024.0
}

/// Open-path probe. `GRAFEO_ATREST_PHASE=build` writes the fixture at
/// `GRAFEO_ATREST_PATH`; any other value opens it and reports wall time
/// and resident set. `GRAFEO_COMPACT_BASE_MMAP` selects the path.
#[test]
#[ignore = "measurement probe: one scenario per process, driven by env"]
fn compact_base_open_probe() {
    let Ok(path) = std::env::var("GRAFEO_ATREST_PATH") else {
        println!("set GRAFEO_ATREST_PATH; see module docs");
        return;
    };
    let path = std::path::PathBuf::from(path);
    let rows = probe_rows();

    if std::env::var("GRAFEO_ATREST_PHASE").as_deref() == Ok("build") {
        let _ = std::fs::remove_file(&path);
        let mut db = GrafeoDB::with_config(Config::persistent(&path)).expect("open");
        build_probe_fixture(&db, rows);
        db.compact().expect("compact");
        db.close().expect("close");
        println!(
            "phase=build rows={rows} on_disk={:.1}MiB",
            mib(std::fs::metadata(&path).expect("metadata").len())
        );
        return;
    }

    let on_disk = std::fs::metadata(&path).expect("fixture must exist").len();
    let rss_before = proc_status_kib("VmRSS");

    let started = std::time::Instant::now();
    let db = GrafeoDB::with_config(Config::persistent(&path)).expect("reopen");
    let open_wall = started.elapsed();

    let rss_after = proc_status_kib("VmRSS");
    let hwm = proc_status_kib("VmHWM");
    let mapped = db.compact_base_is_mmap_backed();

    // A fast open that answers nothing is not a fast open.
    let store = db.graph_store();
    let node_count = store.node_count();
    assert_eq!(node_count, rows, "the reopened store must hold every row");

    println!(
        "phase=open mode={} rows={rows} on_disk={:.1}MiB open_wall={open_wall:?} \
         VmRSS {:.1} -> {:.1} MiB (delta {:.1} MiB) VmHWM {:.1} MiB",
        if mapped { "mmap" } else { "heap" },
        mib(on_disk),
        kib_to_mib(rss_before),
        kib_to_mib(rss_after),
        kib_to_mib(rss_after - rss_before),
        kib_to_mib(hwm),
    );

    db.close().expect("close");
}

/// Publish-peak probe: builds a store in RAM, then measures the extra
/// resident memory the persist itself costs. Compare the printed
/// `publish_peak` against the container size, and across revisions.
#[test]
#[ignore = "measurement probe: one scenario per process, driven by env"]
fn publish_peak_probe() {
    let rows = probe_rows();
    let dir = tempfile::TempDir::new().expect("tempdir");
    let path = dir.path().join("publish.grafeo");

    let db = GrafeoDB::with_config(Config::persistent(&path)).expect("open");
    build_probe_fixture(&db, rows);

    let rss_built = proc_status_kib("VmRSS");
    let hwm_built = proc_status_kib("VmHWM");

    let started = std::time::Instant::now();
    db.close().expect("close");
    let close_wall = started.elapsed();

    let hwm_after = proc_status_kib("VmHWM");
    let on_disk = std::fs::metadata(&path).expect("metadata").len();

    println!(
        "rows={rows} on_disk={:.1}MiB close_wall={close_wall:?} \
         VmRSS_built={:.1}MiB VmHWM_built={:.1}MiB VmHWM_after_publish={:.1}MiB \
         publish_peak={:.1}MiB ({:.1}x on-disk)",
        mib(on_disk),
        kib_to_mib(rss_built),
        kib_to_mib(hwm_built),
        kib_to_mib(hwm_after),
        kib_to_mib(hwm_after),
        kib_to_mib(hwm_after) / mib(on_disk),
    );
}

/// Reads a `/proc/self/status` field in KiB. Returns 0 off Linux.
fn proc_status_kib(field: &str) -> i64 {
    let Ok(status) = std::fs::read_to_string("/proc/self/status") else {
        return 0;
    };
    status
        .lines()
        .find(|line| line.starts_with(field))
        .and_then(|line| {
            line.split_whitespace()
                .nth(1)
                .and_then(|v| v.parse::<i64>().ok())
        })
        .unwrap_or(0)
}
