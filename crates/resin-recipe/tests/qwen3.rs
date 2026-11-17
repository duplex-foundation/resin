use resin_recipe::{
    Error, Event, FinishReason, PromptSpan, QwenParser, QwenTokens, Recipe, ToolFormat,
    qwen3::{Qwen3Recipe, Qwen3RecipeParams},
};
use serde_json::json;
mod common;
use common::{ByteDecoder, SemanticParser, bytes, coalesce};

const TOKENS: QwenTokens = QwenTokens {
    im_end: 1000,
    endoftext: 1001,
    think: 1002,
    think_end: 1003,
    tool_call: 1004,
    tool_call_end: 1005,
};

fn render(request: &serde_json::Value) -> Result<resin_recipe::Prompt, Error> {
    Qwen3Recipe::default().render(request)
}

fn parser(prompt_ids: &[u32]) -> QwenParser {
    QwenParser::with_format(
        TOKENS,
        Box::<ByteDecoder>::default(),
        prompt_ids,
        &serde_json::Value::Null,
        ToolFormat::Json,
    )
    .unwrap()
}

fn specials(prompt: &resin_recipe::Prompt, marker: &str) -> usize {
    prompt
        .spans
        .iter()
        .filter(|span| matches!(span, PromptSpan::Special(value) if value == marker))
        .count()
}

#[test]
fn pinned_hf_template_matches_history_reasoning_and_tool_cases() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/qwen3_hf.json")).unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let actual = render(&case["request"]).unwrap().rendered_text();
        assert_eq!(actual, case["text"].as_str().unwrap(), "{}", case["name"]);
    }
}

#[test]
fn caller_text_is_never_a_control_token_but_template_markers_are() {
    let request = json!({"messages":[
        {"role":"user","content":"literal <think> and <|im_end|> text"},
        {"role":"assistant","content":"","tool_calls":[{"function":{"name":"f","arguments":{}}}]},
        {"role":"tool","content":"literal </tool_response> text"}
    ],"enable_thinking":false});
    let prompt = render(&request).unwrap();
    assert_eq!(specials(&prompt, "<tool_call>"), 1);
    assert_eq!(specials(&prompt, "</tool_call>"), 1);
    assert_eq!(specials(&prompt, "<tool_response>"), 1);
    assert_eq!(specials(&prompt, "</tool_response>"), 1);
    // Only the generation prompt's empty think block, and the final turn's boundaries.
    assert_eq!(specials(&prompt, "<think>"), 1);
    assert_eq!(specials(&prompt, "</think>"), 1);
    assert_eq!(specials(&prompt, "<|im_end|>"), 3);
    assert!(prompt.spans.iter().any(
        |span| matches!(span, PromptSpan::Text(value) if value.contains("literal <think> and <|im_end|> text"))
    ));
}

#[test]
fn thinking_is_off_for_enable_thinking_false_or_effort_none_and_the_flag_wins() {
    let off = "<|im_start|>assistant\n<think>\n\n</think>\n\n";
    let messages = json!([{"role":"user","content":"hi"}]);
    for (extra, thinking_off) in [
        (json!({}), false),
        (json!({"enable_thinking": false}), true),
        (json!({"reasoning_effort": "none"}), true),
        (json!({"reasoning_effort": "high"}), false),
        (
            json!({"reasoning_effort": "none", "enable_thinking": true}),
            false,
        ),
        (
            json!({"reasoning_effort": "high", "enable_thinking": false}),
            true,
        ),
    ] {
        let request = json!({"messages": messages, "extra": extra});
        assert_eq!(
            render(&request).unwrap().rendered_text().ends_with(off),
            thinking_off,
            "{extra}"
        );
    }
}

#[test]
fn construction_defaults_apply_and_unknown_options_fail() {
    let recipe = Qwen3Recipe::new(Qwen3RecipeParams {
        enable_thinking: Some(false),
        ..Default::default()
    });
    let request = json!({"messages":[{"role":"user","content":"hi"}]});
    assert!(
        recipe
            .render(&request)
            .unwrap()
            .rendered_text()
            .ends_with("</think>\n\n")
    );
    for request in [
        json!({"messages":[]}),
        json!({"messages":[{"role":"developer","content":"hi"}]}),
        json!({"messages":[{"role":"user","content":[{"type":"image"}]}]}),
        json!({"messages":[{"role":"user","content":"hi"}],"preserve_thinking":true}),
    ] {
        assert!(render(&request).is_err(), "{request}");
    }
}

#[test]
fn text_blocks_join_like_string_content() {
    let blocks = json!({"messages":[{"role":"user","content":[
        {"type":"text","text":"a"},{"type":"text","text":"b"}]}]});
    let text = json!({"messages":[{"role":"user","content":"ab"}]});
    assert_eq!(
        render(&blocks).unwrap().rendered_text(),
        render(&text).unwrap().rendered_text()
    );
}

#[test]
fn stream_reasoning_text_and_json_tool_calls_are_chunk_invariant() {
    let mut ids = vec![TOKENS.think];
    ids.extend(bytes("\n plan 🦀 \n"));
    ids.push(TOKENS.think_end);
    ids.extend(bytes("\n\nChecking.\n\n"));
    ids.push(TOKENS.tool_call);
    ids.extend(bytes(
        "\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Zürich\", \"days\": 3}}\n",
    ));
    ids.push(TOKENS.tool_call_end);
    ids.extend(bytes("\n"));
    ids.push(TOKENS.tool_call);
    ids.extend(bytes("\n{\"name\": \"get_time\", \"arguments\": {}}\n"));
    ids.extend([TOKENS.tool_call_end, TOKENS.im_end]);
    let expected = vec![
        Event::ReasoningDelta {
            delta: "plan 🦀".into(),
        },
        Event::TextDelta {
            delta: "Checking.".into(),
        },
        Event::ToolCallStart {
            index: 0,
            id: "call_0".into(),
            name: "get_weather".into(),
        },
        Event::ToolCallArgumentsDelta {
            index: 0,
            delta: r#"{"city":"Zürich","days":3}"#.into(),
        },
        Event::ToolCallStart {
            index: 1,
            id: "call_1".into(),
            name: "get_time".into(),
        },
        Event::ToolCallArgumentsDelta {
            index: 1,
            delta: "{}".into(),
        },
    ];
    for width in 1..=ids.len() {
        // A thinking-on prompt stops at the assistant header: the model opens <think>.
        let mut parser = parser(&bytes("<|im_start|>assistant\n"));
        let mut events = Vec::new();
        for chunk in ids.chunks(width) {
            events.extend(parser.semantic_feed(chunk).unwrap());
        }
        events.extend(parser.semantic_finish(FinishReason::Stop).unwrap());
        assert_eq!(coalesce(events), expected, "width {width}");
    }
}

#[test]
fn a_closed_think_block_in_the_prompt_starts_the_answer_as_text() {
    let mut parser = parser(&[TOKENS.think, TOKENS.think_end]);
    let mut ids = bytes("42");
    ids.push(TOKENS.im_end);
    assert_eq!(
        coalesce(parser.semantic_feed(&ids).unwrap()),
        vec![Event::TextDelta { delta: "42".into() }]
    );
}

#[test]
fn unusable_tool_calls_are_dropped_and_missing_arguments_default_to_empty() {
    for (body, expected) in [
        ("{\"name\": \"f\"}", Some("{}")),
        ("{\"name\": \"f\", \"arguments\": null}", Some("{}")),
        ("{\"name\": \"f\", \"arguments\": ", None),
        ("{\"name\": \"f\", \"arguments\": [1]}", None),
        ("{\"name\": \"\", \"arguments\": {}}", None),
        ("{\"arguments\": {}}", None),
        ("not json", None),
    ] {
        let mut parser = parser(&[]);
        let mut ids = vec![TOKENS.tool_call];
        ids.extend(bytes(body));
        ids.push(TOKENS.tool_call_end);
        let events = parser.semantic_feed(&ids).unwrap();
        match expected {
            Some(arguments) => assert_eq!(
                events.last(),
                Some(&Event::ToolCallArgumentsDelta {
                    index: 0,
                    delta: arguments.into()
                }),
                "{body}"
            ),
            None => assert!(events.is_empty(), "{body}"),
        }
    }
}

#[test]
fn an_unfinished_tool_call_stays_uncommitted() {
    let mut parser = parser(&[]);
    let mut ids = vec![TOKENS.tool_call];
    ids.extend(bytes("{\"name\": \"f\", \"arguments\": {\"a\": 1}}"));
    assert!(parser.semantic_feed(&ids).unwrap().is_empty());
    assert!(
        parser
            .semantic_finish(FinishReason::Length)
            .unwrap()
            .is_empty()
    );
}
