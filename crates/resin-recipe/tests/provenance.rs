use resin_recipe::deepseek::DeepSeekV41Recipe;
use resin_recipe::gemma::Gemma4Recipe;
use resin_recipe::gpt_oss::GptOssRecipe;
use resin_recipe::inkling2::Inkling2Recipe;
use resin_recipe::lfm2::Lfm2Recipe;
use resin_recipe::qwen::Qwen38Recipe;
use resin_recipe::{
    AttributedEvent, Decoder, Event, FinishReason, Recipe, ResponseParser, Tokenizer,
    TokenizerDecoder, TokenizerEncoder,
};
use serde_json::{Map, json};
use std::{collections::BTreeMap, sync::Arc};

const CONTROLS: &[&str] = &[
    "<|im_start|>",
    "<|im_end|>",
    "<|fim_suffix|>",
    "<|ghissue|>",
    "<|meta_sep|>",
    "<|im_sep|>",
    "<|meta_start|>",
    "<|message_model|>",
    "<|content_text|>",
    "<|content_thinking|>",
    "<|content_invoke_tool_json|>",
    "<|content_invoke_tool_text|>",
    "<|content_tool_error|>",
    "<|content_model_end_sampling|>",
    "<|end_message|>",
    "<｜begin▁of▁sentence｜>",
    "<｜end▁of▁sentence｜>",
    "<think>",
    "</think>",
    "｜DSML｜",
    "<|endoftext|>",
    "<tool_call>",
    "</tool_call>",
    "<|turn>",
    "<turn|>",
    "<|channel>",
    "<channel|>",
    "<|tool_call>",
    "<tool_call|>",
    "<|tool_response>",
    "<tool_response|>",
    "<|\"|>",
    "<eos>",
    "<|tool_call_start|>",
    "<|tool_call_end|>",
];
fn byte_character(byte: u8) -> char {
    if (33..=126).contains(&byte) || (161..=172).contains(&byte) || (174..=255).contains(&byte) {
        return char::from(byte);
    }
    let offset = (0..byte)
        .filter(|b| !(33..=126).contains(b) && !(161..=172).contains(b) && !(174..=255).contains(b))
        .count();
    char::from_u32(256 + offset as u32).unwrap()
}
fn tokenizer(fallback: bool, extra: &[&str]) -> Arc<Tokenizer> {
    let mut vocab = Map::new();
    for byte in 0..=255_u8 {
        vocab.insert(
            if fallback {
                format!("<0x{byte:02X}>")
            } else {
                byte_character(byte).to_string()
            },
            json!(byte),
        );
    }
    for (index, text) in CONTROLS.iter().chain(extra).enumerate() {
        vocab.insert((*text).to_owned(), json!(256 + index));
    }
    let decoder = if fallback {
        json!({"type":"Sequence","decoders":[{"type":"Replace","pattern":{"String":"▁"},"content":" "},{"type":"ByteFallback"},{"type":"Fuse"}]})
    } else {
        json!({"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false})
    };
    Arc::new(Tokenizer::from_bytes(serde_json::to_vec(&json!({"version":"1.0","added_tokens":[],"normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":decoder,"model":{"type":"BPE","vocab":vocab,"merges":[]}})).unwrap()).unwrap())
}
fn bytes(text: &str) -> Vec<u32> {
    text.bytes().map(u32::from).collect()
}
fn id(tokenizer: &Tokenizer, text: &str) -> u32 {
    tokenizer.token_to_id(text).unwrap()
}
fn token_parts(tokenizer: &Tokenizer, parts: &[&str]) -> Vec<u32> {
    parts
        .iter()
        .flat_map(|part| {
            tokenizer
                .token_to_id(part)
                .map_or_else(|| bytes(part), |id| vec![id])
        })
        .collect()
}

fn recipe<R: Recipe + Default + 'static>() -> Box<dyn Recipe> {
    Box::new(R::default())
}

fn parser(
    model: &dyn Recipe,
    tokenizer: Arc<Tokenizer>,
    prompt: &[u32],
) -> Box<dyn ResponseParser> {
    model
        .create_parser(
            &TokenizerEncoder::new(tokenizer.clone()).unwrap(),
            Box::new(TokenizerDecoder::new(tokenizer).unwrap()),
            &json!({}),
            prompt,
        )
        .unwrap()
}

type Ownership = BTreeMap<usize, Vec<(usize, usize, usize, usize, usize)>>;
fn normalize(events: Vec<AttributedEvent>) -> (String, String, Ownership) {
    let (mut content, mut reasoning) = (String::new(), String::new());
    let mut owners = BTreeMap::<_, Vec<_>>::new();
    for event in events {
        let target = match event.event {
            Event::TextDelta { delta } => {
                content.push_str(&delta);
                (&content, delta.len(), 0)
            }
            Event::ReasoningDelta { delta } => {
                reasoning.push_str(&delta);
                (&reasoning, delta.len(), 1)
            }
            _ => {
                assert!(event.spans.is_empty());
                continue;
            }
        };
        let base = target.0.len() - target.1;
        for span in event.spans {
            assert!(span.text.end as usize <= target.1);
            assert!(span.source.start <= span.source.end);
            owners.entry(span.token as usize).or_default().push((
                span.source.start as usize,
                span.source.end as usize,
                base + span.text.start as usize,
                base + span.text.end as usize,
                target.2,
            ));
        }
    }
    (content, reasoning, owners)
}

#[test]
fn all_model_grammars_preserve_unicode_and_source_indices_across_batching() {
    for (model, prompt, before, transition) in [
        (
            recipe::<GptOssRecipe>(),
            &[
                "<|im_start|>",
                "assistant",
                "<|meta_sep|>",
                "analysis",
                "<|im_sep|>",
            ][..],
            &[][..],
            &["<|im_end|>", "<|im_start|>", "assistant", "<|im_sep|>"][..],
        ),
        (
            recipe::<Inkling2Recipe>(),
            &["<|content_thinking|>"],
            &[],
            &["<|content_text|>"],
        ),
        (
            recipe::<DeepSeekV41Recipe>(),
            &["<think>"],
            &[],
            &["</think>"],
        ),
        (recipe::<Qwen38Recipe>(), &["<think>"], &[], &["</think>"]),
        (
            recipe::<Gemma4Recipe>(),
            &["<|channel>"],
            &[],
            &["<channel|>"],
        ),
        (recipe::<Lfm2Recipe>(), &[], &["<think>"], &["</think>"]),
    ] {
        for fallback in [false, true] {
            let tokenizer = tokenizer(fallback, &[]);
            let prompt = token_parts(&tokenizer, prompt);
            let mut generated = token_parts(&tokenizer, before);
            generated.extend(bytes("hi 🦀!"));
            generated.extend(token_parts(&tokenizer, transition));
            generated.extend(bytes("ok 🧠!"));
            let mut expected = None;
            for chunk_size in [1, 2, 3, 7, generated.len()] {
                let mut decoder = parser(model.as_ref(), tokenizer.clone(), &prompt);
                let mut events = Vec::new();
                for chunk in generated.chunks(chunk_size) {
                    events.extend(decoder.feed(chunk).unwrap());
                }
                events.extend(decoder.finish(FinishReason::Length).unwrap());
                let actual = normalize(events);
                assert_eq!(
                    (&actual.0, &actual.1),
                    (&"ok 🧠!".to_owned(), &"hi 🦀!".to_owned())
                );
                assert_eq!(actual.2.len(), "hi 🦀!ok 🧠!".len());
                if let Some(expected) = &expected {
                    assert_eq!(&actual, expected);
                } else {
                    expected = Some(actual);
                }
            }
        }
    }
}

#[test]
fn qwen_trim_retains_delayed_internal_whitespace_and_marks_partial_tokens() {
    let tokenizer = tokenizer(false, &["ĠhelloĠ", "worldĠ"]);
    let mut parser = parser(recipe::<Qwen38Recipe>().as_ref(), tokenizer.clone(), &[]);
    let first = id(&tokenizer, "ĠhelloĠ");
    let second = id(&tokenizer, "worldĠ");
    let mut events = parser.feed(&[first]).unwrap();
    events.extend(parser.feed(&[second]).unwrap());
    events.extend(parser.finish(FinishReason::Length).unwrap());
    let (text, _, owners) = normalize(events);
    assert_eq!(text, "hello world");
    assert_eq!(owners[&0], vec![(1, 6, 0, 5, 0), (6, 7, 5, 6, 0)]);
    assert_eq!(owners[&1], vec![(0, 5, 6, 11, 0)]);
}

#[test]
fn gemma_header_body_inside_one_token_and_deepseek_marker_prefix_are_attributed() {
    let tokenizer = tokenizer(false, &["analysisĊbody", "aĊĊ<", "x"]);
    let mut gemma = parser(recipe::<Gemma4Recipe>().as_ref(), tokenizer.clone(), &[]);
    let events = gemma
        .feed(&[
            id(&tokenizer, "<|channel>"),
            id(&tokenizer, "analysisĊbody"),
        ])
        .unwrap();
    let (_, reasoning, owners) = normalize(events);
    assert_eq!(reasoning, "body");
    assert_eq!(owners[&1], vec![(9, 13, 0, 4, 1)]);
    let mut deepseek = parser(
        recipe::<DeepSeekV41Recipe>().as_ref(),
        tokenizer.clone(),
        &[],
    );
    let mut events = deepseek.feed(&[id(&tokenizer, "aĊĊ<")]).unwrap();
    events.extend(deepseek.feed(&[id(&tokenizer, "x")]).unwrap());
    events.extend(deepseek.finish(FinishReason::Length).unwrap());
    let (text, _, owners) = normalize(events);
    assert_eq!(text, "a\n\n<x");
    assert_eq!(owners[&0], vec![(0, 1, 0, 1, 0), (1, 4, 1, 4, 0)]);
}

#[test]
fn incomplete_utf8_flush_owns_replacements() {
    for fallback in [false, true] {
        let tokenizer = tokenizer(fallback, &[]);
        let mut decoder = TokenizerDecoder::new(tokenizer).unwrap();
        assert!(decoder.push(&[0xF0, 0x9F], 7).unwrap().is_empty());
        let tail = decoder.finish().unwrap();
        assert_eq!(tail.text, if fallback { "��" } else { "�" });
        assert_eq!(
            tail.spans.iter().map(|s| s.token).collect::<Vec<_>>(),
            vec![7, 8]
        );
        assert!(tail.spans.iter().all(|s| s.source == (0..1)));
    }
}

#[test]
fn transformed_tools_have_no_fabricated_provenance_and_serde_keeps_flat_events() {
    let tokenizer = tokenizer(false, &[]);
    for (model, open, body, close) in [
        (
            recipe::<Inkling2Recipe>(),
            "<|content_invoke_tool_json|>",
            r#"{"name":"f","args":{"a":1}}"#,
            "<|content_text|>",
        ),
        (
            recipe::<Qwen38Recipe>(),
            "<tool_call>",
            "<function=f><parameter=a>1</parameter></function>",
            "</tool_call>",
        ),
        (
            recipe::<Gemma4Recipe>(),
            "<|tool_call>",
            "call:f{a:1}",
            "<tool_call|>",
        ),
        (
            recipe::<Lfm2Recipe>(),
            "<|tool_call_start|>",
            "[f(a=1)]",
            "<|tool_call_end|>",
        ),
    ] {
        let mut parser = parser(model.as_ref(), tokenizer.clone(), &[]);
        let mut ids = vec![id(&tokenizer, open)];
        ids.extend(bytes(body));
        ids.push(id(&tokenizer, close));
        let events = parser.feed(&ids).unwrap();
        assert!(
            events
                .iter()
                .any(|event| matches!(event.event, Event::ToolCallStart { .. }))
        );
        assert!(events.iter().all(|event| event.spans.is_empty()));
    }
    let mut parser = parser(recipe::<Qwen38Recipe>().as_ref(), tokenizer, &[]);
    let event = parser.feed(&bytes("x")).unwrap().pop().unwrap();
    let value = serde_json::to_value(&event).unwrap();
    assert_eq!(value["type"], "text_delta");
    assert_eq!(value["delta"], "x");
    assert_eq!(value["token_spans"][0], json!([0, 0, 1, 0, 1]));
    assert_eq!(
        serde_json::from_value::<AttributedEvent>(value).unwrap(),
        event
    );
    let plain: AttributedEvent = Event::TextDelta { delta: "x".into() }.into();
    assert_eq!(
        serde_json::to_value(plain).unwrap(),
        json!({"type":"text_delta","delta":"x"})
    );
}

#[test]
fn source_bytes_share_decoder_conventions_and_unsupported_chains_fail_upfront() {
    use resin_recipe::TokenBytes;
    let tokenizer = tokenizer(true, &["▁hello", "<0x0000>", "<0xGG>"]);
    let bytes = TokenBytes::new(&tokenizer);
    for (spelling, expected) in [
        ("▁hello", b" hello".as_slice()),
        ("<0x0000>", b"<0x0000>"),
        ("<0xGG>", b"<0xGG>"),
    ] {
        assert_eq!(
            bytes.get(id(&tokenizer, spelling), spelling).unwrap(),
            expected
        );
    }
    let tokenizer = Tokenizer::from_bytes(br#"{"version":"1.0","added_tokens":[],"decoder":{"type":"Fuse"},"model":{"type":"BPE","vocab":{"hello":0},"merges":[]}}"#).unwrap();
    assert!(TokenizerDecoder::new(Arc::new(tokenizer)).is_err());
}

#[test]
fn qwen_and_deepseek_tool_channel_hints_exclude_deferred_content() {
    let tokenizer = tokenizer(false, &[]);
    let mut qwen = parser(recipe::<Qwen38Recipe>().as_ref(), tokenizer.clone(), &[]);
    assert_eq!(qwen.token_channel(u32::from(b'x')), None);
    let open = id(&tokenizer, "<tool_call>");
    assert_eq!(qwen.token_channel(open), Some("control"));
    qwen.feed(&[open]).unwrap();
    assert_eq!(qwen.token_channel(u32::from(b'x')), Some("tool"));
    qwen.feed(&bytes("<function=f></function>")).unwrap();
    qwen.feed(&[id(&tokenizer, "</tool_call>")]).unwrap();
    assert_eq!(qwen.token_channel(u32::from(b' ')), Some("tool"));
    qwen.feed(&[id(&tokenizer, "<think>")]).unwrap();
    assert_eq!(qwen.token_channel(u32::from(b'x')), None);

    let mut deepseek = parser(
        recipe::<DeepSeekV41Recipe>().as_ref(),
        tokenizer.clone(),
        &[],
    );
    deepseek.feed(&bytes("\n\n<")).unwrap();
    assert_eq!(deepseek.token_channel(u32::from(b'x')), None);
    deepseek.feed(&[id(&tokenizer, "｜DSML｜")]).unwrap();
    assert_eq!(deepseek.token_channel(u32::from(b'x')), None);
    deepseek.feed(&bytes(" calls>")).unwrap();
    assert_eq!(deepseek.token_channel(u32::from(b'<')), Some("tool"));
}

#[test]
fn compact_spans_keep_split_scalars_and_move_full_buffers() {
    use resin_recipe::{AttributedText, TokenSpan};
    assert_eq!(std::mem::size_of::<TokenSpan>(), 20);
    let first = format!("a{}", byte_character(0xC3));
    let second = format!("{}{}", byte_character(0xA9), byte_character(b' '));
    let tokenizer = tokenizer(false, &[&first, &second]);
    let mut decoder = TokenizerDecoder::new(tokenizer.clone()).unwrap();
    assert!(
        decoder
            .push(&[id(&tokenizer, &first)], 0)
            .unwrap()
            .is_empty()
    );
    let mut text = decoder.push(&[id(&tokenizer, &second)], 1).unwrap();
    assert_eq!(text.text, "aé ");
    assert_eq!(
        text.spans,
        vec![
            TokenSpan {
                token: 0,
                source: 0..1,
                text: 0..1
            },
            TokenSpan {
                token: 0,
                source: 1..2,
                text: 1..3
            },
            TokenSpan {
                token: 1,
                source: 0..1,
                text: 1..3
            },
            TokenSpan {
                token: 1,
                source: 1..2,
                text: 3..4
            },
        ]
    );
    let (text_ptr, spans_ptr) = (text.text.as_ptr(), text.spans.as_ptr());
    let whole = text.take(text.len());
    assert_eq!(
        (whole.text.as_ptr(), whole.spans.as_ptr()),
        (text_ptr, spans_ptr)
    );
    let mut empty = AttributedText::default();
    empty.push(whole);
    assert_eq!(
        (empty.text.as_ptr(), empty.spans.as_ptr()),
        (text_ptr, spans_ptr)
    );
    let head = empty.take(1);
    assert_eq!(head.text, "a");
    assert_eq!(empty.text, "é ");
    assert_eq!(
        empty.spans[0],
        TokenSpan {
            token: 0,
            source: 1..2,
            text: 0..2
        }
    );
}

#[test]
fn invalid_byte_groups_preserve_the_exact_native_replacement_policy() {
    for fallback in [false, true] {
        let mut decoder = TokenizerDecoder::new(tokenizer(fallback, &[])).unwrap();
        let mut text = decoder.push(&[b'a' as u32, 0xF0, 0x9F], 0).unwrap();
        text.push(decoder.finish().unwrap());
        assert_eq!(text.text, if fallback { "a��" } else { "a�" });
        assert_eq!(
            text.spans.iter().map(|s| s.token).collect::<Vec<_>>(),
            [0, 1, 2]
        );
        assert!(text.spans.iter().all(|s| s.source == (0..1)));
    }
}

const REPLACEMENT: &str = "\u{fffd}";

fn push_chunks(
    decoder: &mut TokenizerDecoder,
    chunks: &[&[u32]],
) -> Vec<resin_recipe::AttributedText> {
    let mut start = 0;
    chunks
        .iter()
        .map(|chunk| {
            let text = decoder.push(chunk, start).unwrap();
            start += chunk.len() as u32;
            text
        })
        .collect()
}

fn compositions(length: usize) -> Vec<Vec<usize>> {
    if length == 0 {
        return vec![vec![]];
    }
    (1..=length)
        .flat_map(|first| {
            compositions(length - first)
                .into_iter()
                .map(move |mut rest| {
                    rest.insert(0, first);
                    rest
                })
        })
        .collect()
}

#[test]
fn split_multibyte_characters_across_tokens_and_after_invalid_bytes() {
    for fallback in [false, true] {
        let tokenizer = tokenizer(fallback, &[]);
        for scalar in ["é", "€", "\u{1f980}"] {
            let ids = bytes(scalar);
            for split in compositions(ids.len()) {
                let mut decoder = TokenizerDecoder::new(tokenizer.clone()).unwrap();
                let mut offset = 0;
                let chunks: Vec<&[u32]> = split
                    .iter()
                    .map(|&length| {
                        offset += length;
                        &ids[offset - length..offset]
                    })
                    .collect();
                let mut emitted = push_chunks(&mut decoder, &chunks);
                let last = emitted.pop().unwrap();
                assert!(
                    emitted.iter().all(|text| text.is_empty()),
                    "{scalar} {split:?}"
                );
                assert_eq!(last.text, scalar);
                let spans: Vec<_> = last
                    .spans
                    .iter()
                    .map(|span| (span.token, span.source.clone(), span.text.clone()))
                    .collect();
                let expected: Vec<_> = (0..ids.len() as u32)
                    .map(|token| (token, 0..1, 0..scalar.len() as u32))
                    .collect();
                assert_eq!(spans, expected);
                assert!(decoder.finish().unwrap().is_empty());
            }
        }

        let lead = format!("hello{}", byte_character(0xF0));
        let tokenizer = crate::tokenizer(fallback, &[&lead]);
        let mut decoder = TokenizerDecoder::new(tokenizer.clone()).unwrap();
        let head: Vec<u32> = if fallback {
            bytes("hello").into_iter().chain([0xF0]).collect()
        } else {
            vec![id(&tokenizer, &lead)]
        };
        let first = decoder.push(&head, 0).unwrap();
        assert_eq!(first.text, "hello");
        let second = decoder
            .push(&[0x9F, 0xA6, 0x80], head.len() as u32)
            .unwrap();
        assert_eq!(second.text, "\u{1f980}");
        assert_eq!(second.spans.len(), 4);
        assert_eq!(second.spans[0].source, if fallback { 0..1 } else { 5..6 });
        assert!(second.spans.iter().all(|span| span.text == (0..4)));

        let mut decoder = TokenizerDecoder::new(tokenizer).unwrap();
        let texts = push_chunks(&mut decoder, &[&[0xFF, 0xFF, 0xF0, 0x9F], &[0xA6, 0x80]]);
        assert_eq!(texts[0].text, "\u{fffd}\u{fffd}");
        assert_eq!(texts[1].text, "\u{1f980}");
        assert!(decoder.finish().unwrap().is_empty());
    }
}

#[test]
fn long_invalid_utf8_runs_emit_replacements_without_unbounded_holdback() {
    for fallback in [false, true] {
        let tokenizer = tokenizer(fallback, &[]);
        for byte in [0xFF_u32, 0x80] {
            let mut decoder = TokenizerDecoder::new(tokenizer.clone()).unwrap();
            for index in 0..1024 {
                let text = decoder.push(&[byte], index).unwrap();
                assert_eq!(text.text, REPLACEMENT);
                assert_eq!(text.spans.len(), 1);
                assert_eq!(text.spans[0].token, index);
            }
            assert!(decoder.finish().unwrap().is_empty());
        }

        let mut decoder = TokenizerDecoder::new(tokenizer.clone()).unwrap();
        let expected = " \u{fffd}";
        for index in 0..1024 {
            let text = decoder.push(&[0x20, 0xFF], index * 2).unwrap();
            assert_eq!(text.text, expected);
        }
        assert!(decoder.finish().unwrap().is_empty());
    }

    let tokenizer = tokenizer(false, &[]);
    let mut decoder = TokenizerDecoder::new(tokenizer).unwrap();
    let mut actual = String::new();
    for index in 0..1024 {
        let text = decoder.push(&[0x20, 0xF0], index * 2).unwrap();
        actual.push_str(&text.text);
        match index {
            0 => assert_eq!(text.text, ""),
            1 => assert_eq!(text.text, " \u{fffd} "),
            _ => assert_eq!(text.text, "\u{fffd} "),
        }
    }
    let tail = decoder.finish().unwrap();
    assert_eq!(tail.text, REPLACEMENT);
    actual.push_str(&tail.text);
    let raw: Vec<u8> = (0..1024).flat_map(|_| [0x20, 0xF0]).collect();
    assert_eq!(actual, String::from_utf8_lossy(&raw));
}

#[test]
fn worst_case_holdback_is_three_bytes() {
    for fallback in [false, true] {
        let mut decoder = TokenizerDecoder::new(tokenizer(fallback, &[])).unwrap();
        let steps = push_chunks(&mut decoder, &[&[0xF0], &[0x9F], &[0xA6], &[0xF0]]);
        assert!(steps[..3].iter().all(|text| text.is_empty()));
        assert_eq!(
            steps[3].text,
            if fallback {
                "\u{fffd}\u{fffd}\u{fffd}"
            } else {
                REPLACEMENT
            }
        );
        assert_eq!(decoder.finish().unwrap().text, REPLACEMENT);
    }
}

#[test]
fn final_flush_and_control_boundaries_drain_incomplete_tails() {
    for fallback in [false, true] {
        let tokenizer = tokenizer(fallback, &[]);
        let mut decoder = TokenizerDecoder::new(tokenizer.clone()).unwrap();
        for tail in [&[0xC2_u8][..], &[0xE2, 0x82], &[0xF0, 0x9F, 0xA6]] {
            let ids: Vec<u32> = tail.iter().map(|&byte| u32::from(byte)).collect();
            assert!(decoder.push(&ids, 0).unwrap().is_empty());
            let flushed = decoder.finish().unwrap();
            let expected = if fallback { tail.len() } else { 1 };
            assert_eq!(flushed.text, REPLACEMENT.repeat(expected));
            assert!(decoder.finish().unwrap().is_empty());
            assert!(decoder.push(&[b'a' as u32], 0).unwrap().text == "a");
            decoder.reset();
        }

        for (control, prompt) in [("</think>", Some("<think>")), ("<|im_end|>", None)] {
            let prompt: Vec<u32> = prompt
                .map(|text| id(&tokenizer, text))
                .into_iter()
                .collect();
            let mut parser = parser(
                recipe::<Qwen38Recipe>().as_ref(),
                tokenizer.clone(),
                &prompt,
            );
            let events = parser.feed(&[0xF0, 0x9F, id(&tokenizer, control)]).unwrap();
            let (content, reasoning, _) = normalize(events);
            let expected = REPLACEMENT.repeat(if fallback { 2 } else { 1 });
            assert_eq!(format!("{reasoning}{content}"), expected, "{control}");
        }
    }
}

fn decode_in_chunks(tokenizer: &Arc<Tokenizer>, ids: &[u32], split: &[usize]) -> String {
    let mut decoder = TokenizerDecoder::new(tokenizer.clone()).unwrap();
    let (mut text, mut offset) = (String::new(), 0);
    for &length in split {
        offset += length;
        text.push_str(&decoder.push(&ids[offset - length..offset], 0).unwrap().text);
    }
    text.push_str(&decoder.finish().unwrap().text);
    text
}

#[test]
fn decoded_text_does_not_depend_on_push_boundaries_or_fuse_across_empty_tokens() {
    for fallback in [false, true] {
        let tokenizer = tokenizer(fallback, &[""]);
        let empty = id(&tokenizer, "");
        let cases: [(&[u32], &str, &str); 7] = [
            (&[0xC3, empty, 0xA9], "é", "\u{fffd}\u{fffd}"),
            (&[0xFF, 0xF0, 0x9F, 0xA6, 0x80], "\u{fffd}🦀", "\u{fffd}🦀"),
            (&[0x20, 0xFF], " \u{fffd}", " \u{fffd}"),
            (&[0x61, 0xF0, 0x9F], "a\u{fffd}", "a\u{fffd}\u{fffd}"),
            (
                &[0xF0, 0x9F, empty, 0xA6, 0x80],
                "🦀",
                "\u{fffd}\u{fffd}\u{fffd}\u{fffd}",
            ),
            (
                &[0xE2, 0x82, 0xFF, 0xAC],
                "\u{fffd}\u{fffd}\u{fffd}",
                "\u{fffd}\u{fffd}\u{fffd}\u{fffd}",
            ),
            (&[empty, 0xC3, 0xA9, empty, 0xC3], "é\u{fffd}", "é\u{fffd}"),
        ];
        for (ids, byte_level, byte_fallback) in cases {
            let expected = if fallback { byte_fallback } else { byte_level };
            for split in compositions(ids.len()) {
                assert_eq!(
                    decode_in_chunks(&tokenizer, ids, &split),
                    expected,
                    "fallback={fallback} {ids:x?} {split:?}"
                );
            }
        }
    }
}

#[test]
fn holdback_stops_at_three_bytes_and_valid_text_after_invalid_bytes_emits() {
    for fallback in [false, true] {
        let tokenizer = tokenizer(fallback, &[]);
        let mut decoder = TokenizerDecoder::new(tokenizer.clone()).unwrap();
        let texts = push_chunks(&mut decoder, &[&[0x61, 0xF0], &[0x9F, 0xA6], &[0x80]]);
        let texts: Vec<_> = texts.iter().map(|text| text.text.as_str()).collect();
        assert_eq!(texts, ["", "a", "\u{1f980}"]);

        let mut decoder = TokenizerDecoder::new(tokenizer).unwrap();
        assert_eq!(
            decoder.push(&[0xFF, 0x61, 0x62], 0).unwrap().text,
            "\u{fffd}ab"
        );
        assert!(decoder.finish().unwrap().is_empty());
    }
}

#[test]
fn flushed_tail_of_a_partly_emitted_token_keeps_its_source_offset() {
    let lead = format!("hello{}", byte_character(0xF0));
    let tokenizer = tokenizer(false, &[&lead]);
    let mut decoder = TokenizerDecoder::new(tokenizer.clone()).unwrap();
    assert_eq!(
        decoder.push(&[id(&tokenizer, &lead)], 0).unwrap().text,
        "hello"
    );
    let tail = decoder.finish().unwrap();
    assert_eq!(tail.text, REPLACEMENT);
    assert_eq!(tail.spans.len(), 1);
    assert_eq!(
        (tail.spans[0].token, tail.spans[0].source.clone()),
        (0, 5..6)
    );
}

#[test]
fn byte_fallback_short_tokens_do_not_wait_and_empty_tokens_end_runs() {
    let tokenizer = tokenizer(true, &["a", ""]);
    let (short, empty) = (id(&tokenizer, "a"), id(&tokenizer, ""));
    let mut decoder = TokenizerDecoder::new(tokenizer.clone()).unwrap();
    assert_eq!(decoder.push(&[short, 0xF0], 0).unwrap().text, "a");
    assert_eq!(decoder.push(&[empty], 2).unwrap().text, REPLACEMENT);
    assert_eq!(decoder.push(&[0xC3], 3).unwrap().text, "");
    assert_eq!(decoder.push(&[empty], 4).unwrap().text, REPLACEMENT);
    assert!(decoder.finish().unwrap().is_empty());
}
