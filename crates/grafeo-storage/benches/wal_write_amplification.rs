//! One-shot WAL syscall probe for a 10,000-mutation transaction.
//!
//! Build it with `cargo bench -p grafeo-storage --bench
//! wal_write_amplification --no-run`, then run the emitted executable under:
//!
//! ```text
//! strace -qq -c -e trace=write,writev,fsync,fdatasync <executable>
//! ```
//!
//! The workload intentionally uses a payload close to the observed TraceDecay
//! mutation frame size. The syscall summary is the benchmark result: write
//! calls and durability syncs per 10,000 mutations.

use grafeo_common::types::{NodeId, TransactionId, Value};
use grafeo_storage::wal::{DurabilityMode, WalConfig, WalManager, WalRecord, WalRecovery};

const MUTATIONS: u64 = 10_000;
const PAYLOAD_BYTES: usize = 300;

fn main() {
    let dir = tempfile::tempdir().expect("create WAL benchmark directory");
    let wal = WalManager::with_config(
        dir.path(),
        WalConfig {
            durability: DurabilityMode::Sync,
            ..WalConfig::default()
        },
    )
    .expect("open WAL");
    let payload = "x".repeat(PAYLOAD_BYTES);

    for id in 0..MUTATIONS {
        wal.log(&WalRecord::SetNodeProperty {
            id: NodeId::new(id),
            key: "payload".to_owned(),
            value: Value::String(payload.clone().into()),
        })
        .expect("append mutation");
    }
    wal.log(&WalRecord::TransactionCommit {
        transaction_id: TransactionId::new(1),
    })
    .expect("commit mutation group");
    drop(wal);

    let recovered = WalRecovery::new(dir.path()).recover().expect("recover WAL");
    let expected_records = usize::try_from(MUTATIONS).expect("benchmark size fits usize") + 1;
    assert_eq!(recovered.len(), expected_records);
}
