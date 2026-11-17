//! Qwen3 dense (non-3.5) conversation format.
//!
//! Renders the pinned `Qwen/Qwen3-32B` chat template: JSON tool calls, and a
//! generation prompt that stops at `<|im_start|>assistant\n` unless thinking is
//! off. Qwen3.5 and later use [`crate::qwen`]. Content is text only; markers
//! inside message text are encoded as ordinary text, never as control tokens.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    Error, MessagePolicy, Prompt,
    qwen::{QwenParser, QwenTokens, ToolFormat},
    render::{function, list, messages, text_content},
};

const TOOLS_HEADER: &str = "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>";
const TOOLS_FOOTER: &str = "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>";

/// Per-request controls, matching the checkpoint template's variables.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Qwen3RecipeParams {
    pub add_generation_prompt: Option<bool>,
    /// `false` closes an empty `<think>` block in the generation prompt.
    pub enable_thinking: Option<bool>,
    /// `"none"` turns thinking off, like `enable_thinking: false`; every other
    /// value leaves it on. `enable_thinking` wins when both are set.
    pub reasoning_effort: Option<String>,
}

#[derive(Clone, Debug, Default)]
pub struct Qwen3Recipe {
    params: Qwen3RecipeParams,
}

impl Qwen3Recipe {
    pub fn new(params: Qwen3RecipeParams) -> Self {
        Self { params }
    }
}

fn start(prompt: &mut Prompt, role: &str) {
    prompt.special("<|im_start|>").text(role).text("\n");
}

fn end(prompt: &mut Prompt) {
    prompt.special("<|im_end|>").text("\n");
}

fn content(message: &Value) -> Result<String, Error> {
    text_content(&message["content"], |_| {
        Err(Error::new("Qwen3 renders text content only"))
    })
}

/// A tool-response user turn is part of the running tool exchange, not a new query.
fn is_tool_response(text: &str) -> bool {
    text.starts_with("<tool_response>") && text.ends_with("</tool_response>")
}

/// Split an assistant message into the template's `(reasoning, content)`.
fn split_reasoning(message: &Value, content: String) -> (String, String) {
    if let Some(reasoning) = message["reasoning_content"].as_str() {
        return (reasoning.to_owned(), content);
    }
    let Some((before, _)) = content.split_once("</think>") else {
        return (String::new(), content);
    };
    let before = before.trim_end_matches('\n');
    let reasoning = before
        .rsplit_once("<think>")
        .map_or(before, |(_, reasoning)| reasoning)
        .trim_start_matches('\n');
    let answer = content
        .rsplit_once("</think>")
        .map_or("", |(_, answer)| answer)
        .trim_start_matches('\n');
    (reasoning.to_owned(), answer.to_owned())
}

fn render_tool_calls(prompt: &mut Prompt, message: &Value, content: &str) -> Result<(), Error> {
    for (index, call) in list(&message["tool_calls"]).iter().enumerate() {
        if index > 0 || !content.is_empty() {
            prompt.text("\n");
        }
        let call = function(call);
        let name = call["name"]
            .as_str()
            .ok_or_else(|| Error::new("Qwen3 tool calls require a name"))?;
        let arguments = match &call["arguments"] {
            Value::String(text) => text.clone(),
            Value::Null => "null".to_owned(),
            value => crate::json::spaced(value),
        };
        prompt
            .special("<tool_call>")
            .text("\n{\"name\": ")
            .text(serde_json::to_string(name).expect("strings serialize"))
            .text(", \"arguments\": ")
            .text(arguments)
            .text("}\n")
            .special("</tool_call>");
    }
    Ok(())
}

impl crate::Recipe for Qwen3Recipe {
    fn message_policy(&self) -> MessagePolicy {
        MessagePolicy {
            roles: &["system", "user", "assistant", "tool"],
            ..MessagePolicy::STANDARD
        }
    }

    fn render(&self, request: &Value) -> Result<Prompt, Error> {
        let options = crate::recipe::options("Qwen3", &self.params, request)?;
        let messages = messages(request, "Qwen3")?;
        let tools = list(&request["tools"]);
        let contents = messages
            .iter()
            .map(content)
            .collect::<Result<Vec<_>, _>>()?;
        let first_system = messages[0]["role"] == "system";

        let mut prompt = Prompt::new();
        if !tools.is_empty() {
            start(&mut prompt, "system");
            if first_system {
                prompt.text(&contents[0]).text("\n\n");
            }
            prompt.text(TOOLS_HEADER);
            for tool in tools {
                prompt.text("\n").text(crate::json::spaced(tool));
            }
            prompt.template(TOOLS_FOOTER, &["<tool_call>", "</tool_call>"]);
            end(&mut prompt);
        } else if first_system {
            start(&mut prompt, "system");
            prompt.text(&contents[0]);
            end(&mut prompt);
        }

        let last_query = messages
            .iter()
            .zip(&contents)
            .rposition(|(message, content)| message["role"] == "user" && !is_tool_response(content))
            .unwrap_or(messages.len() - 1);
        for (index, message) in messages.iter().enumerate() {
            let role = message["role"].as_str().unwrap_or("");
            let content = &contents[index];
            match role {
                "system" if index == 0 => {}
                "system" | "user" => {
                    start(&mut prompt, role);
                    prompt.text(content);
                    end(&mut prompt);
                }
                "assistant" => {
                    let (reasoning, content) = split_reasoning(message, content.clone());
                    start(&mut prompt, "assistant");
                    let last = index + 1 == messages.len();
                    if index > last_query && (last || !reasoning.is_empty()) {
                        prompt
                            .special("<think>")
                            .text("\n")
                            .text(reasoning.trim_matches('\n'))
                            .text("\n")
                            .special("</think>")
                            .text("\n\n")
                            .text(content.trim_start_matches('\n'));
                    } else {
                        prompt.text(&content);
                    }
                    render_tool_calls(&mut prompt, message, &content)?;
                    end(&mut prompt);
                }
                "tool" => {
                    if index == 0 || messages[index - 1]["role"] != "tool" {
                        prompt.special("<|im_start|>").text("user");
                    }
                    prompt.text("\n").special("<tool_response>").text("\n");
                    prompt.text(content);
                    prompt.text("\n").special("</tool_response>");
                    if index + 1 == messages.len() || messages[index + 1]["role"] != "tool" {
                        end(&mut prompt);
                    }
                }
                _ => return Err(Error::new("unexpected Qwen3 message role")),
            }
        }

        if options.add_generation_prompt.unwrap_or(true) {
            start(&mut prompt, "assistant");
            let thinking = options
                .enable_thinking
                .unwrap_or(options.reasoning_effort.as_deref() != Some("none"));
            if !thinking {
                prompt
                    .special("<think>")
                    .text("\n\n")
                    .special("</think>")
                    .text("\n\n");
            }
        }
        Ok(prompt)
    }

    fn create_parser(
        &self,
        encoder: &dyn crate::Encoder,
        decoder: Box<dyn crate::Decoder>,
        request: &Value,
        prompt_ids: &[u32],
    ) -> Result<Box<dyn crate::ResponseParser>, Error> {
        Ok(Box::new(QwenParser::with_format(
            QwenTokens::resolve(encoder)?,
            decoder,
            prompt_ids,
            &request["tools"],
            ToolFormat::Json,
        )?))
    }

    fn stop_token_ids(&self, encoder: &dyn crate::Encoder) -> Result<Vec<u32>, Error> {
        Ok(vec![
            encoder.special_id("<|im_end|>")?,
            encoder.special_id("<|endoftext|>")?,
        ])
    }
}
