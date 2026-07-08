// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

pub mod hf_dataset;
pub mod multi_turn;
pub mod random;
pub mod random_mm;
pub mod sharegpt;
pub mod sonnet;
pub mod speed_bench;

use std::sync::Arc;

/// Represents a single inference request for benchmarking.
/// Matches Python's SampleRequest dataclass from datasets.py:71-82.
///
/// `prompt` uses `Arc<str>` to avoid expensive String clones when distributing
/// requests across tokio tasks. At 100k prompts with 8k tokens each, this saves
/// ~3GB of peak memory vs cloning String per task.
#[derive(Debug, Clone)]
pub struct SampleRequest {
    pub prompt: Arc<str>,
    pub prompt_len: usize,
    pub expected_output_len: usize,
    pub request_id: Option<String>,
    /// Pre-computed token IDs for this prompt.
    /// When set, the completions backend sends these directly via `prompt_token_ids`
    /// instead of the text `prompt`, avoiding server-side re-tokenization.
    pub prompt_token_ids: Option<Arc<[u32]>>,
    /// Multimodal content items as pre-serialized JSON fragments.
    /// Each `Arc<str>` is a complete JSON object string, e.g.
    /// `{"type":"image_url","image_url":{"url":"data:image/jpeg;base64,..."}}`
    ///
    /// Pre-serialized to avoid:
    /// 1. `serde_json::Value` tree overhead (3 Maps + keys per image)
    /// 2. Deep-cloning ~200KB+ base64 data when building request payloads
    ///
    /// Double-`Arc` for zero-cost sharing: outer Arc for the slice, inner Arc for each fragment.
    pub multi_modal_content: Option<Arc<[Arc<str>]>>,
    /// Pre-serialized OpenAI chat `messages` array as a complete JSON string,
    /// e.g. `[{"role":"user","content":[{"type":"text","text":"..."},{"type":"image_url",...}]}]`.
    ///
    /// Set by datasets when `--enable-multimodal-chat` is on (mirrors Python's
    /// `apply_multimodal_chat_transformation`: the dataset builds the chat messages
    /// and the backend sends them verbatim). When set, `multi_modal_content` is None
    /// and the mm items are embedded here instead. `prompt` still holds the text part
    /// for token accounting and /tokenize verification.
    pub chat_messages_json: Option<Arc<str>>,
}

/// A single turn in a multi-turn conversation.
#[derive(Debug, Clone)]
pub struct ConversationTurn {
    pub user_message: Arc<str>,
    pub user_message_len: usize,
    pub expected_output_len: usize,
}

/// A complete multi-turn conversation with all turns pre-generated.
#[derive(Debug, Clone)]
pub struct MultiTurnConversation {
    pub conversation_id: String,
    pub turns: Vec<ConversationTurn>,
}
