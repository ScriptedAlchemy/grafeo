//! Compression and encoding for graph property storage.
//!
//! Graph properties can take up a lot of space - especially string-heavy data like
//! names and labels. This module provides several encoding strategies to shrink
//! your data without losing information.
//!
//! | Data type | Best codec | Typical savings |
//! | --------- | ---------- | --------------- |
//! | Sorted integers (IDs, timestamps) | [`DeltaBitPacked`] | 5-20x smaller |
//! | Small integers (ages, counts) | [`BitPackedInts`] | 2-16x smaller |
//! | Repeated strings (labels, categories) | [`DictionaryEncoding`] | 2-50x smaller |
//! | Booleans (flags, markers) | [`BitVector`] | 8x smaller |
//!
//! Use [`CodecSelector`] to automatically pick the best codec for your data,
//! or choose manually when you know your data characteristics.
//!
//! # Example
//!
//! ```no_run
//! use grafeo_core::codec::{TypeSpecificCompressor, CodecSelector};
//!
//! // Compress sorted integers
//! let values: Vec<u64> = (100..200).collect();
//! let compressed = TypeSpecificCompressor::compress_integers(&values).unwrap();
//! println!("Compression ratio: {:.1}x", compressed.compression_ratio());
//!
//! // Compress booleans
//! let bools = vec![true, false, true, true, false];
//! let compressed = TypeSpecificCompressor::compress_booleans(&bools).unwrap();
//! ```

pub mod bitpack;
pub mod bitvec;
pub mod block;
pub mod delta;
pub mod dictionary;
#[cfg(feature = "tiered-storage")]
pub mod epoch_store;
pub mod runlength;
pub mod selector;
#[cfg(feature = "succinct-indexes")]
pub mod succinct;

// Re-export commonly used types
pub use bitpack::{BitPackedInts, DeltaBitPacked};
pub use bitvec::{BitVector, BitVectorBuilder};
pub use block::{BlockEntry, DEFAULT_BLOCK_ROWS};
pub use delta::{DeltaEncoding, zigzag_decode, zigzag_encode};
pub use dictionary::{DictionaryBuilder, DictionaryEncoding};
pub use runlength::{Run, RunLengthAnalyzer, RunLengthEncoding, SignedRunLengthEncoding};
pub use selector::{
    CodecSelector, CompressedData, CompressionCodec, CompressionMetadata, TypeSpecificCompressor,
};

// Tiered storage exports (feature-gated)
#[cfg(feature = "tiered-storage")]
pub use epoch_store::{
    CompressedEpochBlock, CompressionType, EpochBlockHeader, EpochStore, EpochStoreStats,
    IndexEntry, ZoneMap,
};

// Succinct data structure exports (feature-gated)
#[cfg(feature = "succinct-indexes")]
pub use succinct::{EliasFano, SuccinctBitVector, WaveletTree};

/// The buffer a deserialized store's `Bytes`-backed codecs are views into.
///
/// A store read from a section keeps its column bodies as slices of that
/// section's buffer, which is mapped from the file or, on the read path, one
/// heap copy. Its owner accounts for the buffer once; a codec owns only the
/// buffers outside it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SectionSpan {
    start: usize,
    end: usize,
}

impl SectionSpan {
    /// The span of `section`.
    #[must_use]
    pub fn of(section: &[u8]) -> Self {
        let start = section.as_ptr().addr();
        Self {
            start,
            end: start + section.len(),
        }
    }

    /// The buffer's length.
    #[must_use]
    pub fn len(&self) -> usize {
        self.end - self.start
    }

    /// Whether no section backs the store.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.start == self.end
    }

    /// Heap bytes `bytes` holds outside this section.
    #[must_use]
    pub fn owned_bytes(&self, bytes: &[u8]) -> usize {
        let start = bytes.as_ptr().addr();
        if bytes.is_empty() || (start >= self.start && start + bytes.len() <= self.end) {
            0
        } else {
            bytes.len()
        }
    }
}
