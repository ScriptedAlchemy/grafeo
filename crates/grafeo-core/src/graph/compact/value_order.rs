//! Row orders that serve indexed property lookups on a
//! [`CompactStore`](super::CompactStore).
//!
//! An indexed node column carries its non-null rows sorted by value (ties in
//! row order), so an equality lookup is a binary search over the column's
//! own encoded values. The order is a flat array of row numbers: a sealed
//! section stores it next to the column and a mapped store serves it in
//! place, so reopening a sealed store builds no index.
//!
//! Only codecs whose stored form orders the values they decode to carry an
//! order: dictionary strings (by encoded entry bytes), bit-packed unsigned
//! integers, and raw signed integers. Other columns answer through the
//! zone-map-pruned scan.

use std::cmp::Ordering;
use std::sync::Arc;

use bytes::Bytes;
use grafeo_common::memory::heap::vec_bytes;
use grafeo_common::types::Value;

use super::column::ColumnCodec;
use super::dict_value::encode_dict_entry;
use crate::codec::BitVector;
use crate::codec::pages::SectionPages;

/// A column's non-null rows in ascending value order.
#[derive(Debug)]
pub(crate) enum RowOrder {
    Inline(Vec<u32>),
    /// LE u32 rows in place in a mapped section.
    Mapped {
        rows: Bytes,
        pages: Arc<SectionPages>,
    },
}

impl RowOrder {
    /// Serves an order in place from a section.
    ///
    /// # Errors
    ///
    /// A region that is not whole LE u32 rows.
    pub(crate) fn mapped(rows: Bytes, pages: &Arc<SectionPages>) -> Result<Self, String> {
        if !rows.len().is_multiple_of(4) {
            return Err(format!(
                "row order of {} bytes is not whole rows",
                rows.len()
            ));
        }
        Ok(Self::Mapped {
            rows,
            pages: Arc::clone(pages),
        })
    }

    pub(crate) fn len(&self) -> usize {
        match self {
            Self::Inline(rows) => rows.len(),
            Self::Mapped { rows, .. } => rows.len() / 4,
        }
    }

    pub(crate) fn row(&self, index: usize) -> Option<usize> {
        match self {
            Self::Inline(rows) => rows.get(index).map(|&row| row as usize),
            Self::Mapped { rows, pages } => {
                let start = index.checked_mul(4)?;
                let bytes = rows.get(start..start + 4)?;
                if !pages.verify_slice(bytes) {
                    return None;
                }
                Some(u32::from_le_bytes(bytes.try_into().ok()?) as usize)
            }
        }
    }

    /// Appends the order as LE u32 rows, the form a section stores.
    pub(crate) fn write_to(&self, out: &mut Vec<u8>) {
        out.reserve(self.len() * 4);
        for index in 0..self.len() {
            if let Some(row) = self.row(index) {
                // reason: rows come from a u32 order
                #[allow(clippy::cast_possible_truncation)]
                out.extend_from_slice(&(row as u32).to_le_bytes());
            }
        }
    }

    /// Heap bytes held outside a mapped section.
    pub(crate) fn heap_bytes(&self) -> usize {
        match self {
            Self::Inline(rows) => vec_bytes(rows),
            Self::Mapped { .. } => 0,
        }
    }
}

/// Sorts the rows of `codec` that `null_mask` does not mark absent by value,
/// or `None` when the codec has no ordered equality index.
pub(crate) fn build_row_order(
    codec: &ColumnCodec,
    null_mask: Option<&BitVector>,
) -> Option<RowOrder> {
    let len = u32::try_from(codec.len()).ok()?;
    let present = |row: &u32| {
        let row = *row as usize;
        !null_mask.is_some_and(|mask| mask.get(row).unwrap_or(false))
            && !matches!(codec, ColumnCodec::Dict(dict) if dict.is_null(row))
    };
    let mut rows: Vec<u32> = (0..len).filter(present).collect();
    match codec {
        ColumnCodec::Dict(dict) => {
            let mut by_entry: Vec<u32> = (0..u32::try_from(dict.dictionary_size()).ok()?).collect();
            by_entry.sort_unstable_by(|&a, &b| dict.entry(a as usize).cmp(&dict.entry(b as usize)));
            let mut rank = vec![0u32; by_entry.len()];
            for (position, &code) in by_entry.iter().enumerate() {
                // reason: position < dictionary size, which fits u32
                #[allow(clippy::cast_possible_truncation)]
                let position = position as u32;
                rank[code as usize] = position;
            }
            rows.sort_by_key(|&row| {
                dict.code_at(row as usize)
                    .and_then(|code| rank.get(code as usize).copied())
                    .unwrap_or(u32::MAX)
            });
        }
        ColumnCodec::BitPacked(packed) => {
            rows.sort_by_key(|&row| packed.get(row as usize));
        }
        ColumnCodec::RawI64(store) => {
            rows.sort_by_key(|&row| store.get(row as usize));
        }
        _ => return None,
    }
    Some(RowOrder::Inline(rows))
}

/// The rows of `codec` whose value equals `value`, in row order, found by
/// binary search over `order`.
pub(crate) fn find_eq(codec: &ColumnCodec, order: &RowOrder, value: &Value) -> Vec<usize> {
    let entry;
    let compare: Box<dyn Fn(usize) -> Option<Ordering> + '_> = match (codec, value) {
        (ColumnCodec::Dict(dict), Value::String(_) | Value::Bytes(_)) => {
            entry = encode_dict_entry(value);
            Box::new(|row| {
                let code = dict.code_at(row)?;
                Some(dict.entry(code as usize)?.cmp(entry.as_str()))
            })
        }
        (ColumnCodec::BitPacked(packed), &Value::Int64(target)) => {
            let Ok(target) = u64::try_from(target) else {
                return Vec::new();
            };
            Box::new(move |row| Some(packed.get(row)?.cmp(&target)))
        }
        (ColumnCodec::RawI64(store), &Value::Int64(target)) => {
            Box::new(move |row| Some(store.get(row)?.cmp(&target)))
        }
        _ => return Vec::new(),
    };

    let (mut low, mut high) = (0, order.len());
    while low < high {
        let mid = low + (high - low) / 2;
        let Some(ordering) = order.row(mid).and_then(&compare) else {
            return Vec::new();
        };
        if ordering == Ordering::Less {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    let mut rows = Vec::new();
    for index in low..order.len() {
        match order.row(index) {
            Some(row) if compare(row) == Some(Ordering::Equal) => rows.push(row),
            _ => break,
        }
    }
    rows
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::{BitPackedInts, DictionaryBuilder};

    #[test]
    fn dictionary_order_finds_every_equal_row_in_row_order() {
        let mut builder = DictionaryBuilder::new();
        for value in ["pear", "apple", "fig", "apple", "pear", "apple"] {
            builder.add(value);
        }
        let codec = ColumnCodec::Dict(builder.build());
        let order = build_row_order(&codec, None).unwrap();
        let rows: Vec<usize> = (0..order.len()).map(|i| order.row(i).unwrap()).collect();
        assert_eq!(rows, vec![1, 3, 5, 2, 0, 4]);
        assert_eq!(
            find_eq(&codec, &order, &Value::from("apple")),
            vec![1, 3, 5]
        );
        assert_eq!(find_eq(&codec, &order, &Value::from("pear")), vec![0, 4]);
        assert_eq!(
            find_eq(&codec, &order, &Value::from("kiwi")),
            Vec::<usize>::new()
        );
        assert_eq!(
            find_eq(&codec, &order, &Value::Int64(1)),
            Vec::<usize>::new()
        );
    }

    #[test]
    fn masked_rows_are_left_out_of_the_order() {
        let codec = ColumnCodec::BitPacked(BitPackedInts::pack(&[5, 0, 5, 3]));
        let mask = BitVector::from_bools(&[false, true, true, false]);
        let order = build_row_order(&codec, Some(&mask)).unwrap();
        assert_eq!(find_eq(&codec, &order, &Value::Int64(5)), vec![0]);
        assert_eq!(
            find_eq(&codec, &order, &Value::Int64(0)),
            Vec::<usize>::new()
        );
        assert_eq!(find_eq(&codec, &order, &Value::Int64(3)), vec![3]);
        assert_eq!(
            find_eq(&codec, &order, &Value::Int64(-3)),
            Vec::<usize>::new()
        );
    }
}
