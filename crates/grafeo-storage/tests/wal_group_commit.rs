//! WAL group-commit durability and buffering contracts.

#![cfg(feature = "wal")]

use std::fs;
use std::process::Command;
use std::thread;
use std::time::{Duration, Instant};

use grafeo_common::types::{NodeId, TransactionId};
use grafeo_storage::wal::{DurabilityMode, WalConfig, WalManager, WalRecord, WalRecovery};

const CHILD_MODE_ENV: &str = "GRAFEO_WAL_GROUP_COMMIT_CHILD";
const CHILD_WAL_DIR_ENV: &str = "GRAFEO_WAL_GROUP_COMMIT_DIR";
const CHILD_READY_FILE_ENV: &str = "GRAFEO_WAL_GROUP_COMMIT_READY";

fn sync_wal(path: &std::path::Path) -> WalManager {
    WalManager::with_config(
        path,
        WalConfig {
            durability: DurabilityMode::Sync,
            ..WalConfig::default()
        },
    )
    .expect("open synchronous WAL")
}

fn create_node(id: u64, label: &str) -> WalRecord {
    WalRecord::CreateNode {
        id: NodeId::new(id),
        labels: vec![label.to_owned()],
    }
}

#[test]
fn mutations_stay_buffered_until_the_group_commit() {
    let dir = tempfile::tempdir().expect("create WAL directory");
    let wal = sync_wal(dir.path());

    for id in 0..100 {
        wal.log(&create_node(id, "buffered"))
            .expect("append mutation");
    }

    let bytes_before_commit = fs::metadata(wal.path()).expect("WAL metadata").len();
    assert_eq!(
        bytes_before_commit, 0,
        "sub-group mutations must not each flush the bounded WAL buffer"
    );

    wal.log(&WalRecord::TransactionCommit {
        transaction_id: TransactionId::new(1),
    })
    .expect("commit group");
    assert!(
        fs::metadata(wal.path()).expect("WAL metadata").len() > 0,
        "the commit must flush the complete group"
    );
    drop(wal);

    let recovered = WalRecovery::new(dir.path()).recover().expect("recover WAL");
    assert_eq!(recovered.len(), 101);
}

#[test]
fn group_commit_kill_child() {
    if std::env::var_os(CHILD_MODE_ENV).is_none() {
        return;
    }

    let wal_dir = std::env::var_os(CHILD_WAL_DIR_ENV).expect("child WAL directory");
    let ready_file = std::env::var_os(CHILD_READY_FILE_ENV).expect("child ready file");
    let wal = sync_wal(std::path::Path::new(&wal_dir));

    wal.log(&create_node(1, "durable"))
        .expect("append durable mutation");
    wal.log(&WalRecord::TransactionCommit {
        transaction_id: TransactionId::new(1),
    })
    .expect("commit durable group");

    wal.log(&create_node(2, "not-durable"))
        .expect("append uncommitted mutation");
    fs::write(&ready_file, b"ready").expect("publish child readiness");

    loop {
        thread::park_timeout(Duration::from_mins(1));
    }
}

#[test]
fn process_kill_between_group_commits_recovers_last_durable_group() {
    let dir = tempfile::tempdir().expect("create test directory");
    let wal_dir = dir.path().join("wal");
    let ready_file = dir.path().join("ready");
    let current_exe = std::env::current_exe().expect("resolve test executable");
    let mut child = Command::new(current_exe)
        .arg("--exact")
        .arg("group_commit_kill_child")
        .arg("--nocapture")
        .env(CHILD_MODE_ENV, "1")
        .env(CHILD_WAL_DIR_ENV, &wal_dir)
        .env(CHILD_READY_FILE_ENV, &ready_file)
        .spawn()
        .expect("spawn WAL child");

    let deadline = Instant::now() + Duration::from_secs(10);
    while !ready_file.exists() {
        assert!(
            Instant::now() < deadline,
            "child did not reach the between-groups crash point"
        );
        assert!(
            child.try_wait().expect("poll child").is_none(),
            "child exited before the crash point"
        );
        thread::sleep(Duration::from_millis(10));
    }

    child.kill().expect("kill child between group commits");
    let status = child.wait().expect("reap child");
    assert!(!status.success(), "child must be killed, not exit cleanly");

    let recovered = WalRecovery::new(&wal_dir)
        .recover()
        .expect("recover killed WAL");
    assert_eq!(
        recovered.len(),
        2,
        "only one mutation and its commit survive"
    );
    match &recovered[0] {
        WalRecord::CreateNode { id, labels } => {
            assert_eq!(*id, NodeId::new(1));
            assert_eq!(labels, &["durable"]);
        }
        other => panic!("expected durable mutation, got {other:?}"),
    }
    assert!(matches!(
        recovered[1],
        WalRecord::TransactionCommit { transaction_id }
            if transaction_id == TransactionId::new(1)
    ));
}
