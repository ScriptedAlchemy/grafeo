//! Original-id maps of an id-preserving [`CompactStore`](super::CompactStore).
//!
//! The forward map is the `(id, table, position)` records in ascending id
//! order, answered by binary search; the reverse map is one id per table
//! position. Both are the exact bytes a section stores, so a store read from
//! a mapped section serves them in place instead of rebuilding hash maps
//! over every node and edge.

use std::sync::Arc;

use bytes::Bytes;
use grafeo_common::memory::heap::vec_bytes;

use crate::codec::pages::SectionPages;

/// Bytes of one forward record: `id u64, table u16, position u64`, LE.
pub(crate) const RECORD_BYTES: usize = 18;

/// One forward record.
pub(crate) type IdRecord = (u64, u16, u64);

/// Preserved ids to `(table, position)` and back.
#[derive(Debug)]
pub(crate) struct IdMap {
    forward: Forward,
    reverse: Vec<Reverse>,
}

#[derive(Debug)]
enum Forward {
    Inline(Vec<IdRecord>),
    Mapped {
        records: Bytes,
        pages: Arc<SectionPages>,
    },
}

#[derive(Debug)]
enum Reverse {
    Inline(Vec<u64>),
    Mapped {
        ids: Bytes,
        pages: Arc<SectionPages>,
    },
}

impl IdMap {
    /// Builds both directions from each table's position-ordered ids.
    /// Positions holding `invalid` are padding and resolve from no id.
    pub(crate) fn from_positions(tables: Vec<Vec<u64>>, invalid: u64) -> Self {
        let mut records: Vec<IdRecord> = Vec::with_capacity(tables.iter().map(Vec::len).sum());
        for (table, ids) in tables.iter().enumerate() {
            let Ok(table) = u16::try_from(table) else {
                continue;
            };
            records.extend(
                ids.iter()
                    .enumerate()
                    .filter(|&(_, &id)| id != invalid)
                    .map(|(position, &id)| (id, table, position as u64)),
            );
        }
        records.sort_unstable_by_key(|&(id, _, _)| id);
        Self {
            forward: Forward::Inline(records),
            reverse: tables.into_iter().map(Reverse::Inline).collect(),
        }
    }

    /// Builds both directions from id-ordered records, as legacy sections
    /// store them, for `table_count` tables. Positions no record names are
    /// `invalid` padding.
    pub(crate) fn from_records(records: Vec<IdRecord>, table_count: usize, invalid: u64) -> Self {
        let mut reverse: Vec<Vec<u64>> = vec![Vec::new(); table_count];
        for &(id, table, position) in &records {
            let (Some(ids), Ok(position)) =
                (reverse.get_mut(table as usize), usize::try_from(position))
            else {
                continue;
            };
            if ids.len() <= position {
                ids.resize(position + 1, invalid);
            }
            ids[position] = id;
        }
        Self {
            forward: Forward::Inline(records),
            reverse: reverse.into_iter().map(Reverse::Inline).collect(),
        }
    }

    /// Serves both directions in place from a section: `records` holds
    /// [`RECORD_BYTES`]-wide records in ascending id order and `reverse`
    /// one LE u64 id per position of each table.
    ///
    /// # Errors
    ///
    /// A region whose length is not a whole number of entries.
    pub(crate) fn mapped(
        records: Bytes,
        reverse: Vec<Bytes>,
        pages: &Arc<SectionPages>,
    ) -> Result<Self, String> {
        if !records.len().is_multiple_of(RECORD_BYTES) {
            return Err(format!(
                "id map of {} bytes is not whole records",
                records.len()
            ));
        }
        if let Some(table) = reverse.iter().position(|ids| !ids.len().is_multiple_of(8)) {
            return Err(format!("reverse id map of table {table} is not whole ids"));
        }
        Ok(Self {
            forward: Forward::Mapped {
                records,
                pages: Arc::clone(pages),
            },
            reverse: reverse
                .into_iter()
                .map(|ids| Reverse::Mapped {
                    ids,
                    pages: Arc::clone(pages),
                })
                .collect(),
        })
    }

    /// Number of mapped ids.
    pub(crate) fn len(&self) -> usize {
        match &self.forward {
            Forward::Inline(records) => records.len(),
            Forward::Mapped { records, .. } => records.len() / RECORD_BYTES,
        }
    }

    /// The record at `index` in id order.
    pub(crate) fn record(&self, index: usize) -> Option<IdRecord> {
        match &self.forward {
            Forward::Inline(records) => records.get(index).copied(),
            Forward::Mapped { records, pages } => {
                let start = index.checked_mul(RECORD_BYTES)?;
                let bytes = records.get(start..start + RECORD_BYTES)?;
                if !pages.verify_slice(bytes) {
                    return None;
                }
                let u64_at = |at: usize| {
                    let mut word = [0u8; 8];
                    word.copy_from_slice(&bytes[at..at + 8]);
                    u64::from_le_bytes(word)
                };
                Some((
                    u64_at(0),
                    u16::from_le_bytes([bytes[8], bytes[9]]),
                    u64_at(10),
                ))
            }
        }
    }

    /// The `(table, position)` holding `id`.
    pub(crate) fn resolve(&self, id: u64) -> Option<(u16, u64)> {
        let (mut low, mut high) = (0, self.len());
        while low < high {
            let mid = low + (high - low) / 2;
            let (found, table, position) = self.record(mid)?;
            match found.cmp(&id) {
                std::cmp::Ordering::Less => low = mid + 1,
                std::cmp::Ordering::Greater => high = mid,
                std::cmp::Ordering::Equal => return Some((table, position)),
            }
        }
        None
    }

    /// The id stored at `position` of `table`.
    pub(crate) fn original(&self, table: u16, position: u64) -> Option<u64> {
        let position = usize::try_from(position).ok()?;
        match self.reverse.get(table as usize)? {
            Reverse::Inline(ids) => ids.get(position).copied(),
            Reverse::Mapped { ids, pages } => {
                let start = position.checked_mul(8)?;
                let bytes = ids.get(start..start + 8)?;
                if !pages.verify_slice(bytes) {
                    return None;
                }
                let mut word = [0u8; 8];
                word.copy_from_slice(bytes);
                Some(u64::from_le_bytes(word))
            }
        }
    }

    /// Number of positions table `table` maps.
    pub(crate) fn table_len(&self, table: usize) -> usize {
        match self.reverse.get(table) {
            Some(Reverse::Inline(ids)) => ids.len(),
            Some(Reverse::Mapped { ids, .. }) => ids.len() / 8,
            None => 0,
        }
    }

    /// Number of tables the reverse map covers.
    pub(crate) fn table_count(&self) -> usize {
        self.reverse.len()
    }

    /// The highest mapped id.
    pub(crate) fn max_id(&self) -> Option<u64> {
        self.len()
            .checked_sub(1)
            .and_then(|last| self.record(last))
            .map(|(id, _, _)| id)
    }

    /// Every mapped id in ascending order.
    pub(crate) fn ids(&self) -> impl Iterator<Item = u64> + '_ {
        (0..self.len()).filter_map(|index| self.record(index).map(|(id, _, _)| id))
    }

    /// Heap bytes held outside a mapped section.
    pub(crate) fn heap_bytes(&self) -> usize {
        let forward = match &self.forward {
            Forward::Inline(records) => vec_bytes(records),
            Forward::Mapped { .. } => 0,
        };
        vec_bytes(&self.reverse)
            + forward
            + self
                .reverse
                .iter()
                .map(|reverse| match reverse {
                    Reverse::Inline(ids) => vec_bytes(ids),
                    Reverse::Mapped { .. } => 0,
                })
                .sum::<usize>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const INVALID: u64 = u64::MAX;

    #[test]
    fn positions_and_records_build_the_same_map() {
        let tables = vec![vec![40, 7, 19], vec![3, INVALID, 88]];
        let from_positions = IdMap::from_positions(tables.clone(), INVALID);
        let records: Vec<IdRecord> = (0..from_positions.len())
            .map(|index| from_positions.record(index).unwrap())
            .collect();
        assert_eq!(
            records,
            vec![(3, 1, 0), (7, 0, 1), (19, 0, 2), (40, 0, 0), (88, 1, 2)]
        );
        let from_records = IdMap::from_records(records, 2, INVALID);
        for map in [&from_positions, &from_records] {
            assert_eq!(map.resolve(19), Some((0, 2)));
            assert_eq!(map.resolve(88), Some((1, 2)));
            assert_eq!(map.resolve(20), None);
            assert_eq!(map.original(1, 1), Some(INVALID));
            assert_eq!(map.original(0, 0), Some(40));
            assert_eq!(map.original(1, 3), None);
            assert_eq!(map.max_id(), Some(88));
            assert_eq!(map.ids().collect::<Vec<_>>(), vec![3, 7, 19, 40, 88]);
        }
    }
}
