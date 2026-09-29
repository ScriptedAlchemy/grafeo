//! Exact heap footprints of the containers graph storage holds, computed from
//! each container's own length and capacity and the layout its allocation
//! was made with.
//!
//! These are the bytes the global allocator was asked for. Allocator size
//! classes and page slack are the allocator's own overhead and are not
//! counted here.

use std::mem::{align_of, size_of};
use std::sync::Arc;

use arcstr::ArcStr;

/// The two reference counts every `Arc` and [`ArcStr`] allocation starts with.
const REFCOUNT_HEADER_BYTES: usize = 2 * size_of::<usize>();

/// Bytes a `Vec<T>`'s buffer occupies: its capacity, not its length.
#[must_use]
pub fn vec_bytes<T>(vec: &Vec<T>) -> usize {
    vec.capacity() * size_of::<T>()
}

/// Bytes the allocation behind an `Arc<[T]>` of `len` elements occupies:
/// the reference counts, then the elements, padded to the allocation's
/// alignment.
#[must_use]
pub fn arc_slice_bytes<T>(len: usize) -> usize {
    let align = align_of::<usize>().max(align_of::<T>());
    let header = REFCOUNT_HEADER_BYTES.next_multiple_of(align_of::<T>());
    (header + len * size_of::<T>()).next_multiple_of(align)
}

/// Bytes an `Arc<str>` allocation occupies.
#[must_use]
pub fn arc_str_bytes(value: &Arc<str>) -> usize {
    arc_slice_bytes::<u8>(value.len())
}

/// Bytes an [`ArcStr`] allocation occupies: unpadded, and none for a static
/// string.
#[must_use]
pub fn arcstr_bytes(value: &ArcStr) -> usize {
    if ArcStr::is_static(value) {
        0
    } else {
        REFCOUNT_HEADER_BYTES + value.len()
    }
}

/// Bytes a `String`'s buffer occupies.
#[must_use]
pub fn string_bytes(value: &String) -> usize {
    value.capacity()
}

/// Control bytes the SwissTable probes per group: SSE2 groups on x86, NEON
/// groups on aarch64, and one machine word otherwise.
const GROUP_WIDTH: usize = if cfg!(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature = "sse2"
)) {
    16
} else if cfg!(all(target_arch = "aarch64", target_feature = "neon")) {
    8
} else {
    size_of::<usize>()
};

/// Bytes a std `HashMap`'s table occupies. The std map is a SwissTable whose
/// `capacity()` is all but one of a power-of-two bucket count below eight
/// buckets and seven eighths of it from eight up; its one allocation holds
/// every bucket's entry, padded to the control alignment, then a control
/// byte per bucket and one trailing group.
#[must_use]
pub fn std_hash_map_bytes<K, V, S>(map: &std::collections::HashMap<K, V, S>) -> usize {
    swiss_table_bytes::<(K, V)>(map.capacity())
}

/// Bytes a std `HashSet`'s table occupies; see [`std_hash_map_bytes`].
#[must_use]
pub fn std_hash_set_bytes<T, S>(set: &std::collections::HashSet<T, S>) -> usize {
    swiss_table_bytes::<T>(set.capacity())
}

fn swiss_table_bytes<T>(capacity: usize) -> usize {
    if capacity == 0 {
        return 0;
    }
    let buckets = if capacity < 8 {
        capacity + 1
    } else {
        capacity / 7 * 8
    };
    let control_align = align_of::<T>().max(GROUP_WIDTH);
    (size_of::<T>() * buckets).next_multiple_of(control_align) + buckets + GROUP_WIDTH
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arc_str_allocations_carry_their_counts_and_padding() {
        assert_eq!(arc_str_bytes(&Arc::from("abc")), 24);
        assert_eq!(arc_str_bytes(&Arc::from("abcdefghi")), 32);
        assert_eq!(arc_slice_bytes::<u64>(3), 40);
    }

    #[test]
    fn arcstr_allocations_are_unpadded_and_statics_are_free() {
        assert_eq!(arcstr_bytes(&ArcStr::from("abcdefghi")), 25);
        assert_eq!(arcstr_bytes(&arcstr::literal!("abcdefghi")), 0);
    }

    /// The std tables are the same SwissTable hashbrown implements, so the
    /// layout derived from a capacity must be what hashbrown allocates for it.
    #[test]
    fn a_swiss_table_is_charged_what_it_allocates() {
        for entries in [1_usize, 3, 4, 7, 8, 13, 14, 15, 100, 1_000, 30_000] {
            let map: hashbrown::HashMap<u64, (u16, u64)> =
                (0..entries as u64).map(|key| (key, (0, key))).collect();
            assert_eq!(
                swiss_table_bytes::<(u64, (u16, u64))>(map.capacity()),
                map.allocation_size(),
                "{entries} entries"
            );
            let set: hashbrown::HashSet<String> = (0..entries).map(|key| key.to_string()).collect();
            assert_eq!(
                swiss_table_bytes::<String>(set.capacity()),
                set.allocation_size(),
                "{entries} strings"
            );
        }
        assert_eq!(
            std_hash_map_bytes(&std::collections::HashMap::<u64, u64>::new()),
            0
        );
    }

    #[test]
    fn a_vec_is_charged_its_capacity() {
        let mut vec = Vec::<u64>::with_capacity(10);
        vec.push(1);
        assert_eq!(vec_bytes(&vec), 80);
    }
}
