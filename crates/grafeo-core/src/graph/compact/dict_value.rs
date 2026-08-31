//! Lossless `Value` <-> dictionary-entry mapping for the `Dict` codec.
//!
//! The dictionary codec stores strings. Before this module existed, every
//! non-string scalar that fell back to `Dict` was stringified with `Display`
//! and decoded as `Value::String` — for `Value::Bytes` that is silent data
//! corruption: the bytes went in and a string came out. Property graphs built
//! by real applications carry serialized byte payloads on most nodes, so the
//! `Dict` fallback must round-trip them exactly.
//!
//! The mapping stays inside the existing string dictionary: a `Value::Bytes`
//! entry is stored as a marked hex string, and a genuine string that happens
//! to begin with the marker prefix is escaped with a second marker. The
//! marker begins with a NUL byte, which no meaningful user string starts
//! with, so in practice escaping never fires — but the escape keeps the
//! mapping bijective rather than merely unlikely to collide.
//!
//! Whether a dictionary entry *is* marked is decided by the section format
//! version, not by the entry bytes alone: sections older than the marker
//! format stored every entry as a raw string, so a legacy entry that
//! collides with the marker prefix must not be reinterpreted as a payload.
//! The section loader normalizes such entries through
//! [`escape_legacy_entry`] before the store serves them, after which the
//! always-on decoding below is unambiguous for every store in memory.

use arcstr::ArcStr;
use std::sync::Arc;

use grafeo_common::types::Value;

/// Shared prefix of every marked dictionary entry.
pub(crate) const DICT_MARKER_PREFIX: &str = "\u{0}gfo1:";
/// A `Value::Bytes` entry: marker followed by lowercase hex of the payload.
const DICT_BYTES_MARKER: &str = "\u{0}gfo1:b:";
/// A genuine string that itself begins with [`DICT_MARKER_PREFIX`].
const DICT_STRING_ESCAPE_MARKER: &str = "\u{0}gfo1:s:";

fn encode_hex(bytes: &[u8]) -> String {
    const TABLE: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(TABLE[usize::from(byte >> 4)] as char);
        out.push(TABLE[usize::from(byte & 0x0f)] as char);
    }
    out
}

fn decode_hex(hex: &str) -> Option<Vec<u8>> {
    fn nibble(byte: u8) -> Option<u8> {
        match byte {
            b'0'..=b'9' => Some(byte - b'0'),
            b'a'..=b'f' => Some(byte - b'a' + 10),
            _ => None,
        }
    }
    let hex = hex.as_bytes();
    if !hex.len().is_multiple_of(2) {
        return None;
    }
    hex.chunks_exact(2)
        .map(|pair| Some(nibble(pair[0])? << 4 | nibble(pair[1])?))
        .collect()
}

/// Encodes one value into its dictionary-entry string.
///
/// `Value::Bytes` becomes a marked hex string; a string beginning with the
/// marker prefix is escaped; every other value keeps its previous encoding
/// (`Display` for non-string scalars in mixed columns, empty string for
/// nulls), so existing stores and columns are byte-identical.
pub(crate) fn encode_dict_entry(value: &Value) -> String {
    match value {
        Value::Null => String::new(),
        Value::Bytes(bytes) => format!("{DICT_BYTES_MARKER}{}", encode_hex(bytes)),
        Value::String(s) if s.starts_with(DICT_MARKER_PREFIX) => {
            format!("{DICT_STRING_ESCAPE_MARKER}{s}")
        }
        Value::String(s) => s.to_string(),
        other => format!("{other}"),
    }
}

/// Escapes a raw legacy dictionary entry that collides with the marker
/// prefix.
///
/// Sections written before the marker mapping stored every entry as a raw
/// string. Rewriting a colliding entry into the escape form makes
/// [`decode_dict_entry`] return exactly the original string, and keeps
/// equality lookups — which encode the query through
/// [`encode_dict_entry`] — finding it. Returns `None` for the entire
/// population of realistic entries, which do not begin with a NUL byte.
pub(crate) fn escape_legacy_entry(entry: &str) -> Option<String> {
    entry
        .starts_with(DICT_MARKER_PREFIX)
        .then(|| format!("{DICT_STRING_ESCAPE_MARKER}{entry}"))
}

/// Decodes one dictionary entry back into the `Value` it encodes.
///
/// Unmarked entries decode as strings exactly as before the marker
/// existed. A marked entry that fails its own decoding (impossible for
/// entries this module wrote) is returned as the literal string rather
/// than dropped.
pub(crate) fn decode_dict_entry(entry: &str) -> Value {
    if let Some(hex) = entry.strip_prefix(DICT_BYTES_MARKER) {
        if let Some(bytes) = decode_hex(hex) {
            return Value::Bytes(Arc::from(bytes.into_boxed_slice()));
        }
        return Value::String(ArcStr::from(entry));
    }
    if let Some(escaped) = entry.strip_prefix(DICT_STRING_ESCAPE_MARKER) {
        return Value::String(ArcStr::from(escaped));
    }
    Value::String(ArcStr::from(entry))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bytes_round_trip_exactly() {
        let payload: Vec<u8> = (0u8..=255).collect();
        let value = Value::Bytes(Arc::from(payload.clone().into_boxed_slice()));
        let encoded = encode_dict_entry(&value);
        assert!(encoded.starts_with(DICT_BYTES_MARKER));
        assert_eq!(decode_dict_entry(&encoded), value);
    }

    #[test]
    fn empty_bytes_round_trip() {
        let value = Value::Bytes(Arc::from(Vec::new().into_boxed_slice()));
        assert_eq!(decode_dict_entry(&encode_dict_entry(&value)), value);
    }

    #[test]
    fn plain_strings_are_unchanged() {
        let value = Value::String(ArcStr::from("plain"));
        assert_eq!(encode_dict_entry(&value), "plain");
        assert_eq!(decode_dict_entry("plain"), value);
    }

    #[test]
    fn marker_prefixed_string_escapes_and_round_trips() {
        let tricky = format!("{DICT_MARKER_PREFIX}not-actually-bytes");
        let value = Value::String(ArcStr::from(tricky.as_str()));
        let encoded = encode_dict_entry(&value);
        assert!(encoded.starts_with(DICT_STRING_ESCAPE_MARKER));
        assert_eq!(decode_dict_entry(&encoded), value);
    }

    #[test]
    fn corrupt_marked_entry_degrades_to_the_literal_string() {
        let entry = format!("{DICT_BYTES_MARKER}zz-not-hex");
        assert_eq!(
            decode_dict_entry(&entry),
            Value::String(ArcStr::from(entry.as_str()))
        );
    }

    #[test]
    fn legacy_colliding_entry_escapes_to_the_original_string() {
        let raw = format!("{DICT_BYTES_MARKER}00");
        let escaped = escape_legacy_entry(&raw).expect("colliding entry must escape");
        // Decoding hands back the raw legacy string, not a Bytes payload.
        assert_eq!(
            decode_dict_entry(&escaped),
            Value::String(ArcStr::from(raw.as_str()))
        );
        // The escaped form is exactly what the current encoder produces for
        // that string, so equality lookups keep finding the row.
        assert_eq!(
            escaped,
            encode_dict_entry(&Value::String(ArcStr::from(raw.as_str())))
        );

        assert_eq!(escape_legacy_entry("plain"), None);
        assert_eq!(escape_legacy_entry(""), None);
    }
}
