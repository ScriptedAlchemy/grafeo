//! Heap accounting the compact store's tables share.

use std::collections::HashMap;

use grafeo_common::memory::heap::{
    arc_slice_bytes, arcstr_bytes, std_hash_map_bytes, string_bytes, vec_bytes,
};
use grafeo_common::types::{PropertyKey, Value};

use super::schema::{ColumnDef, EdgeSchema, TableSchema};
use super::zone_map::ZoneMap;
use crate::statistics::{ColumnStatistics, Statistics};

/// Heap bytes a value read out of a compact column owns. Columnar codecs
/// decode only scalars, strings, bytes, and vectors, and scalars own none.
pub(super) fn value_bytes(value: &Value) -> usize {
    match value {
        Value::String(text) => arcstr_bytes(text),
        Value::Bytes(bytes) => arc_slice_bytes::<u8>(bytes.len()),
        Value::Vector(items) => arc_slice_bytes::<f32>(items.len()),
        _ => 0,
    }
}

pub(super) fn key_bytes(key: &PropertyKey) -> usize {
    arcstr_bytes(key.as_arc())
}

pub(super) fn zone_map_bytes(zone_map: &ZoneMap) -> usize {
    zone_map.min.as_ref().map_or(0, value_bytes) + zone_map.max.as_ref().map_or(0, value_bytes)
}

fn column_defs_bytes(columns: &Vec<ColumnDef>) -> usize {
    vec_bytes(columns)
        + columns
            .iter()
            .map(|column| arcstr_bytes(&column.name))
            .sum::<usize>()
}

pub(super) fn table_schema_bytes(schema: &TableSchema) -> usize {
    arcstr_bytes(&schema.label) + column_defs_bytes(&schema.columns)
}

pub(super) fn edge_schema_bytes(schema: &EdgeSchema) -> usize {
    arcstr_bytes(&schema.edge_type)
        + arcstr_bytes(&schema.src_label)
        + arcstr_bytes(&schema.dst_label)
        + column_defs_bytes(&schema.property_columns)
}

fn column_statistics_bytes(statistics: &ColumnStatistics) -> usize {
    statistics.min_value.as_ref().map_or(0, value_bytes)
        + statistics.max_value.as_ref().map_or(0, value_bytes)
        + statistics.histogram.as_ref().map_or(0, |histogram| {
            vec_bytes(histogram.bucket_storage())
                + histogram
                    .buckets()
                    .iter()
                    .map(|bucket| value_bytes(&bucket.lower) + value_bytes(&bucket.upper))
                    .sum::<usize>()
        })
}

fn properties_bytes(properties: &HashMap<String, ColumnStatistics>) -> usize {
    std_hash_map_bytes(properties)
        + properties
            .iter()
            .map(|(key, column)| string_bytes(key) + column_statistics_bytes(column))
            .sum::<usize>()
}

/// Heap bytes an `Arc<Statistics>` holds: the shared allocation and every
/// map, key, and bound inside it.
pub(super) fn statistics_bytes(statistics: &Statistics) -> usize {
    arc_slice_bytes::<Statistics>(1)
        + std_hash_map_bytes(&statistics.labels)
        + statistics
            .labels
            .iter()
            .map(|(label, label_statistics)| {
                string_bytes(label) + properties_bytes(&label_statistics.properties)
            })
            .sum::<usize>()
        + std_hash_map_bytes(&statistics.edge_types)
        + statistics
            .edge_types
            .iter()
            .map(|(edge_type, edge_statistics)| {
                string_bytes(edge_type) + properties_bytes(&edge_statistics.properties)
            })
            .sum::<usize>()
        + properties_bytes(&statistics.properties)
}
