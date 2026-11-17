use crate::{
    AttributedText, Decoder, Error,
    parser::{EventSink, Grammar, Parser},
};

/// Decode generated tokens as plain text, with no reasoning or tool grammar.
///
/// Stop tokens are dropped from the text but never end the response, so a
/// request that ignores stop tokens keeps producing text past them.
pub type PlainParser = Parser<PlainGrammar>;

pub struct PlainGrammar {
    stop_token_ids: Vec<u32>,
}

impl PlainParser {
    pub fn new(decoder: Box<dyn Decoder>, stop_token_ids: Vec<u32>) -> Self {
        Self::with_grammar(PlainGrammar { stop_token_ids }, decoder)
    }
}

impl Grammar for PlainGrammar {
    fn channel(&self) -> Option<&'static str> {
        Some("content")
    }
    fn is_control(&self, id: u32) -> bool {
        self.stop_token_ids.contains(&id)
    }
    fn text(&mut self, text: AttributedText, events: &mut EventSink) -> Result<(), Error> {
        if !text.is_empty() {
            events.text(text, false);
        }
        Ok(())
    }
    fn control(&mut self, _: u32, _: &mut EventSink) -> Result<(), Error> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Event, FinishReason, ResponseParser};

    const STOP: u32 = 0;

    struct Letters;
    impl Decoder for Letters {
        fn push(&mut self, ids: &[u32], _: u32) -> Result<AttributedText, Error> {
            Ok(AttributedText::plain(
                ids.iter()
                    .map(|&id| char::from_u32(u32::from(b'a') + id).unwrap())
                    .collect::<String>(),
            ))
        }
        fn finish(&mut self) -> Result<AttributedText, Error> {
            Ok(AttributedText::default())
        }
        fn reset(&mut self) {}
    }

    fn text(events: Vec<crate::AttributedEvent>) -> String {
        events
            .into_iter()
            .map(|event| match event.event {
                Event::TextDelta { delta } => delta,
                other => panic!("plain output must be text, got {other:?}"),
            })
            .collect()
    }

    #[test]
    fn stop_tokens_vanish_from_text_and_do_not_end_the_response() {
        let mut parser = PlainParser::new(Box::new(Letters), vec![STOP]);
        let mut output = text(parser.feed(&[1, 2]).unwrap());
        output += &text(parser.feed(&[STOP, 3, STOP, 4]).unwrap());
        output += &text(parser.finish(FinishReason::Length).unwrap());
        assert_eq!(output, "bcde");
    }

    #[test]
    fn without_stop_tokens_every_token_is_text() {
        let mut parser = PlainParser::new(Box::new(Letters), vec![]);
        assert_eq!(text(parser.feed(&[STOP, 1]).unwrap()), "ab");
    }
}
