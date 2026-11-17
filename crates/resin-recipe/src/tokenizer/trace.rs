//! Streaming UTF-8 decoding and byte ownership for the supported stock decoder chains.
use super::bytes::{TokenBytes, fallback_byte};
use crate::{AttributedText, Error, TokenSpan, Tokenizer};
use serde_json::Value;
use std::collections::VecDeque;

const MAX_UTF8_HOLD_BYTES: usize = 3;

#[derive(Clone, Copy)]
enum Kind {
    ByteLevel,
    ByteFallback,
}
struct Token {
    bytes: Vec<u8>,
    fallback: bool,
}
struct Piece {
    index: u32,
    token: Token,
    start: usize,
}
pub(super) struct Trace {
    kind: Kind,
    bytes: TokenBytes,
    pending: VecDeque<Piece>,
    raw: Vec<u8>,
}
impl Trace {
    pub fn new(tokenizer: &Tokenizer) -> Result<Self, Error> {
        let bytes = TokenBytes::new(tokenizer);
        let kind = kind(&bytes.decoder).ok_or_else(|| {
            Error::new(
                "recipe attribution requires a supported ByteLevel or ByteFallback decoder chain",
            )
        })?;
        Ok(Self {
            kind,
            bytes,
            pending: VecDeque::new(),
            raw: Vec::new(),
        })
    }
    pub fn clear(&mut self) {
        self.pending.clear();
        self.raw.clear();
    }
    pub fn push(
        &mut self,
        tokenizer: &Tokenizer,
        ids: &[u32],
        token_start: u32,
    ) -> Result<(), Error> {
        for (offset, &id) in ids.iter().enumerate() {
            let index = u32::try_from(offset)
                .ok()
                .and_then(|offset| token_start.checked_add(offset))
                .ok_or_else(|| Error::new("generated token index exceeds u32"))?;
            let spelling = tokenizer.id_to_token(id).unwrap_or_default();
            let fallback = !self.bytes.special.contains(&id)
                && matches!(self.kind, Kind::ByteFallback)
                && fallback_byte(&spelling).is_some();
            let bytes = self
                .bytes
                .get(id, &spelling)
                .ok_or_else(|| Error::new("unsupported token byte representation"))?;
            u32::try_from(bytes.len()).map_err(|_| Error::new("token byte length exceeds u32"))?;
            // An empty piece only matters as the boundary that ends a `<0xNN>` run.
            if bytes.is_empty() && (matches!(self.kind, Kind::ByteLevel) || self.pending.is_empty())
            {
                continue;
            }
            let token = Token { bytes, fallback };
            self.pending.push_back(Piece {
                index,
                token,
                start: 0,
            });
        }
        Ok(())
    }
    /// Decodes every byte whose text is final. Only a trailing incomplete
    /// scalar prefix, at most `MAX_UTF8_HOLD_BYTES`, waits for later tokens;
    /// `flush` also replaces that prefix.
    pub fn decode_pending(&mut self, flush: bool) -> Result<AttributedText, Error> {
        if self.pending.is_empty() {
            return Ok(AttributedText::default());
        }
        self.raw.clear();
        for piece in &self.pending {
            self.raw
                .extend_from_slice(&piece.token.bytes[piece.start..]);
        }
        let (emit_len, valid) = match self.kind {
            Kind::ByteLevel => utf8_emit_len(&self.raw, flush),
            Kind::ByteFallback => {
                // Only the trailing run of `<0xNN>` tokens can still change meaning.
                let tail: usize = self
                    .pending
                    .iter()
                    .rev()
                    .take_while(|piece| piece.token.fallback)
                    .map(|piece| piece.token.bytes.len() - piece.start)
                    .sum();
                let tail_start = self.raw.len() - tail;
                let emit = tail_start + utf8_emit_len(&self.raw[tail_start..], flush).0;
                // An empty piece ends a `<0xNN>` run, so bytes on either side never fuse.
                let fused = self
                    .pending
                    .iter()
                    .any(|piece| piece.token.bytes.is_empty());
                (
                    emit,
                    !fused && std::str::from_utf8(&self.raw[..emit]).is_ok(),
                )
            }
        };
        if emit_len == 0 {
            if self.raw.is_empty() {
                self.pending.clear();
            }
            return Ok(AttributedText::default());
        }
        if valid {
            let text = std::str::from_utf8(&self.raw[..emit_len])
                .map_err(|_| Error::new("decoded bytes are not valid UTF-8"))?
                .to_owned();
            u32::try_from(text.len()).map_err(|_| Error::new("decoded text length exceeds u32"))?;
            let mut spans = Vec::with_capacity(self.pending.len());
            let mut offset = 0;
            for piece in &self.pending {
                let end = (offset + piece.token.bytes.len() - piece.start).min(text.len());
                if end > offset {
                    copied_spans(piece, offset, end, &text, &mut spans);
                }
                offset = end;
                if offset == text.len() {
                    break;
                }
            }
            self.consume(emit_len);
            return Ok(AttributedText { text, spans });
        }
        // The uncommon invalid-byte path preserves each decoder's exact replacement
        // policy: ByteLevel is lossy UTF-8; ByteFallback replaces every failed byte.
        let pieces = self.pending.make_contiguous();
        let mut decoded = AttributedText::default();
        let (mut offset, mut raw_offset) = (0, 0);
        while offset < pieces.len() && raw_offset < emit_len {
            let fallback = matches!(self.kind, Kind::ByteFallback) && pieces[offset].token.fallback;
            let end = match self.kind {
                Kind::ByteLevel => pieces.len(),
                Kind::ByteFallback if fallback => {
                    offset
                        + pieces[offset..]
                            .iter()
                            .take_while(|piece| piece.token.fallback)
                            .count()
                }
                Kind::ByteFallback => offset + 1,
            };
            let raw_end = (raw_offset
                + pieces[offset..end]
                    .iter()
                    .map(|piece| piece.token.bytes.len() - piece.start)
                    .sum::<usize>())
            .min(emit_len);
            decode(
                &pieces[offset..end],
                &self.raw[raw_offset..raw_end],
                fallback,
                &mut decoded,
            )?;
            offset = end;
            raw_offset = raw_end;
        }
        self.consume(emit_len);
        Ok(decoded)
    }
    fn consume(&mut self, mut bytes: usize) {
        while let Some(piece) = self.pending.front_mut() {
            let length = piece.token.bytes.len() - piece.start;
            if length > bytes {
                piece.start += bytes;
                break;
            }
            bytes -= length;
            self.pending.pop_front();
        }
    }
}

/// Returns how many leading bytes are final and whether they are all valid UTF-8.
/// Unless `flush`, a trailing incomplete scalar prefix (1 to 3 bytes) stays pending.
fn utf8_emit_len(bytes: &[u8], flush: bool) -> (usize, bool) {
    let first = match std::str::from_utf8(bytes) {
        Ok(_) => return (bytes.len(), true),
        Err(error) => error,
    };
    if flush {
        return (bytes.len(), false);
    }
    let Some(invalid_len) = first.error_len() else {
        // Short buffers wait whole so a scalar split across tokens keeps one source run.
        return if bytes.len() <= MAX_UTF8_HOLD_BYTES {
            (0, true)
        } else {
            (first.valid_up_to(), true)
        };
    };
    let mut source = first.valid_up_to() + invalid_len;
    while source < bytes.len() {
        match std::str::from_utf8(&bytes[source..]) {
            Ok(_) => return (bytes.len(), false),
            Err(error) => {
                source += error.valid_up_to();
                match error.error_len() {
                    Some(len) => source += len,
                    None => break,
                }
            }
        }
    }
    (source, false)
}

/// A token has at most a leading split scalar, a linear middle, and a trailing
/// split scalar. Whole ordinary tokens therefore need just one span.
fn copied_spans(
    piece: &Piece,
    mut start: usize,
    end: usize,
    text: &str,
    spans: &mut Vec<TokenSpan>,
) {
    let source_base = piece.start;
    let text_base = start;
    if !text.is_char_boundary(start) {
        let mut scalar_start = start;
        while !text.is_char_boundary(scalar_start) {
            scalar_start -= 1;
        }
        let mut scalar_end = start;
        while !text.is_char_boundary(scalar_end) {
            scalar_end += 1;
        }
        let fragment_end = end.min(scalar_end);
        spans.push(TokenSpan {
            token: piece.index,
            source: source_base as u32..(source_base + fragment_end - text_base) as u32,
            text: scalar_start as u32..scalar_end as u32,
        });
        start = fragment_end;
    }
    let mut stable_end = end;
    while !text.is_char_boundary(stable_end) {
        stable_end -= 1;
    }
    if start < stable_end {
        spans.push(TokenSpan {
            token: piece.index,
            source: (source_base + start - text_base) as u32
                ..(source_base + stable_end - text_base) as u32,
            text: start as u32..stable_end as u32,
        });
        start = stable_end;
    }
    if start < end {
        let mut scalar_end = end;
        while !text.is_char_boundary(scalar_end) {
            scalar_end += 1;
        }
        spans.push(TokenSpan {
            token: piece.index,
            source: (source_base + start - text_base) as u32
                ..(source_base + end - text_base) as u32,
            text: start as u32..scalar_end as u32,
        });
    }
}
fn kind(value: &Value) -> Option<Kind> {
    let children = if value["type"] == "Sequence" {
        value["decoders"].as_array()?.as_slice()
    } else {
        std::slice::from_ref(value)
    };
    match children {
        [decoder] if decoder["type"] == "ByteLevel" => Some(Kind::ByteLevel),
        [decoder] if decoder["type"] == "ByteFallback" => Some(Kind::ByteFallback),
        [decoder, fuse] if decoder["type"] == "ByteFallback" && fuse["type"] == "Fuse" => {
            Some(Kind::ByteFallback)
        }
        [replace, decoder, fuse]
            if replace["type"] == "Replace"
                && replace["pattern"]["String"] == "▁"
                && replace["content"] == " "
                && decoder["type"] == "ByteFallback"
                && fuse["type"] == "Fuse" =>
        {
            Some(Kind::ByteFallback)
        }
        _ => None,
    }
}
fn decode(
    pieces: &[Piece],
    bytes: &[u8],
    fallback: bool,
    result: &mut AttributedText,
) -> Result<(), Error> {
    let (mut source, mut piece_index, mut piece_offset) = (0, 0, 0);
    while source < bytes.len() {
        let (valid_len, invalid_len) = match std::str::from_utf8(&bytes[source..]) {
            Ok(_) => (bytes.len() - source, 0),
            Err(error) => (
                error.valid_up_to(),
                error
                    .error_len()
                    .unwrap_or(bytes.len() - source - error.valid_up_to()),
            ),
        };
        // Each ill-formed subpart is fixed by its own bytes, so replacing it alone
        // keeps the output independent of how pushes split the token stream.
        let (replacements, replacement_len) = if fallback {
            (invalid_len, 1)
        } else {
            (usize::from(invalid_len > 0), invalid_len)
        };
        let valid = std::str::from_utf8(&bytes[source..source + valid_len]).unwrap();
        for (character, source_len) in valid
            .chars()
            .map(|character| (character, character.len_utf8()))
            .chain(std::iter::repeat_n(('�', replacement_len), replacements))
        {
            let event_start = u32::try_from(result.text.len())
                .map_err(|_| Error::new("decoded text length exceeds u32"))?;
            result.text.push(character);
            let event_end = u32::try_from(result.text.len())
                .map_err(|_| Error::new("decoded text length exceeds u32"))?;
            let end = source + source_len;
            while source < end {
                let piece = &pieces[piece_index];
                let length = piece.token.bytes.len() - piece.start;
                if source >= piece_offset + length {
                    piece_offset += length;
                    piece_index += 1;
                    continue;
                }
                let fragment_end = end.min(piece_offset + length);
                result.push_span(TokenSpan {
                    token: piece.index,
                    source: (piece.start + source - piece_offset) as u32
                        ..(piece.start + fragment_end - piece_offset) as u32,
                    text: event_start..event_end,
                });
                source = fragment_end;
            }
        }
    }
    Ok(())
}
