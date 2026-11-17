mod bytes;
mod trace;
pub use bytes::TokenBytes;

use std::sync::Arc;

pub use tokenizers::Tokenizer;

use crate::{AttributedText, Decoder, Error};

/// Streaming decoder for the supported stock tokenizers, with owned state and a
/// shared tokenizer.
///
/// Emits finalized text on every push and holds back at most 3 incomplete
/// trailing UTF-8 bytes; definitively invalid bytes become U+FFFD immediately.
pub struct TokenizerDecoder {
    tokenizer: Arc<Tokenizer>,
    trace: trace::Trace,
}

impl TokenizerDecoder {
    pub fn new(tokenizer: Arc<Tokenizer>) -> Result<Self, Error> {
        let trace = trace::Trace::new(&tokenizer)?;
        Ok(Self { tokenizer, trace })
    }
}

impl Decoder for TokenizerDecoder {
    fn push(&mut self, ids: &[u32], token_start: u32) -> Result<AttributedText, Error> {
        if ids.is_empty() {
            return Ok(AttributedText::default());
        }
        self.trace.push(&self.tokenizer, ids, token_start)?;
        self.trace.decode_pending(false)
    }

    fn finish(&mut self) -> Result<AttributedText, Error> {
        let result = self.trace.decode_pending(true);
        self.reset();
        result
    }

    fn reset(&mut self) {
        self.trace.clear();
    }
}
