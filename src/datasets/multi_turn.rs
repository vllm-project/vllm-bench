// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::sync::Arc;

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use super::{ConversationTurn, MultiTurnConversation};
use crate::error::{BenchError, Result};
use crate::tokenizer::TokenizerKind;

/// Configuration for generating random multi-turn conversations.
#[derive(Debug, Clone)]
pub struct MultiTurnRandomConfig {
    pub num_conversations: usize,
    pub min_turns: usize,
    pub max_turns: usize,
    /// Shared prefix length prepended to the conversation.
    ///
    /// In normal accumulated-history mode this is added to turn 0, so all
    /// later turns inherit it through history. In no-history prefix-sharing
    /// mode it is added to every independent turn.
    pub prefix_len: usize,
    /// Input length for turn 0.
    pub input_len: usize,
    /// Input length for turns 1+. 0 = fallback to input_len.
    pub per_turn_input_len: usize,
    pub output_len: usize,
    pub seed: u64,
    pub request_id_prefix: String,
    pub prefix_sharing_config: Option<PrefixSharingConfig>,
    /// Range ratio for sampling per-turn input lengths (Python semantics:
    /// lengths drawn uniformly from `[len*(1-r), len*(1+r)]`). Default fixed (0.0).
    pub range_ratio: crate::config::RangeRatio,
    /// Bimodal prefix-cache: fraction of conversations that are "warm" and reuse a
    /// shared cached base prefix on turn 0. 0.0 = off. See [`generate_multi_turn_random`].
    pub cache_hit_fraction: f64,
    /// Bimodal prefix-cache: fraction of a warm conversation's turn-0 length that is
    /// the shared cached prefix. Used with `cache_hit_fraction`.
    pub cache_ratio: f64,
}

/// Configuration for 3-tier prefix sharing in multi-turn user messages.
#[derive(Debug, Clone)]
pub struct PrefixSharingConfig {
    /// Fraction of per-turn input tokens shared across ALL conversations.
    pub global_ratio: f64,
    /// Fraction of per-turn input tokens shared within each conversation.
    pub conversation_ratio: f64,
}

/// Generate a deterministic token sequence from allowed tokens using offset+modulo.
fn make_token_seq(allowed_tokens: &[u32], offset: usize, len: usize) -> Vec<u32> {
    let at_len = allowed_tokens.len();
    (0..len)
        .map(|i| allowed_tokens[(offset + i) % at_len])
        .collect()
}

/// Generate synthetic multi-turn conversations with random user messages.
///
/// Each conversation has `num_turns` turns, each with a random user prompt
/// of `input_len` tokens and `output_len` expected output tokens.
pub fn generate_multi_turn_random(
    tokenizer: &TokenizerKind,
    cfg: &MultiTurnRandomConfig,
) -> Result<Vec<MultiTurnConversation>> {
    let num_conversations = cfg.num_conversations;
    let min_turns = cfg.min_turns;
    let max_turns = cfg.max_turns;
    let prefix_len = cfg.prefix_len;
    let input_len = cfg.input_len;
    let output_len = cfg.output_len;
    let seed = cfg.seed;
    let request_id_prefix = &cfg.request_id_prefix;
    let allowed_tokens = tokenizer.get_allowed_tokens();
    if allowed_tokens.is_empty() {
        return Err(BenchError::Tokenizer("No allowed tokens found".into()));
    }

    let vocab_size = tokenizer.vocab_size() as usize;
    let num_special = tokenizer.num_special_tokens_to_add();
    let real_input_len = input_len.saturating_sub(num_special);
    let real_per_turn_len = if cfg.per_turn_input_len > 0 {
        cfg.per_turn_input_len.saturating_sub(num_special)
    } else {
        real_input_len
    };

    if real_input_len < 1 {
        return Err(BenchError::Config(format!(
            "--random-input-len too small: with {num_special} special tokens, \
             effective input length is {real_input_len}"
        )));
    }
    if real_per_turn_len < 1 {
        return Err(BenchError::Config(format!(
            "--per-turn-input-len too small: with {num_special} special tokens, \
             effective per-turn input length is {real_per_turn_len}"
        )));
    }

    // Prefix sharing mode: generate 3-tier prefixed messages
    let mut rng = StdRng::seed_from_u64(seed);
    if let Some(ref ps_cfg) = cfg.prefix_sharing_config {
        return generate_prefix_sharing_conversations(
            tokenizer,
            cfg,
            ps_cfg,
            &allowed_tokens,
            &mut rng,
        );
    }
    // Bimodal prefix-cache mode (mirrors single-turn RandomDataset): a fraction of
    // whole CONVERSATIONS are "warm" and their turn-0 message begins with a shared
    // cached base prefix covering `cache_ratio` of turn 0's length; the rest are
    // "cold" (fully unique). Because every warm conversation's cached slice is a
    // leading slice of the SAME base, they hit each other's server-side prefix
    // cache, and — living in turn 0 — the cached prefix keeps paying off every
    // later turn via history accumulation. When bimodal is on, --random-prefix-len
    // is ignored (the cached base IS the shared prefix), matching single-turn.
    let bimodal = cfg.cache_hit_fraction > 0.0 && cfg.cache_ratio > 0.0;
    if bimodal && (cfg.cache_hit_fraction > 1.0 || cfg.cache_ratio > 1.0) {
        return Err(BenchError::Config(
            "--random-cache-hit-fraction and --random-cache-ratio must be in [0, 1]".into(),
        ));
    }

    let use_range = !cfg.range_ratio.is_fixed();
    let (in_low, in_high) = cfg.range_ratio.input_bounds(real_input_len);
    let (pt_low, pt_high) = cfg.range_ratio.input_bounds(real_per_turn_len);

    // Shared cached base prefix (bimodal only). Built once from turn 0's UPPER bound
    // so any warm conversation's cached slice fits. Uses an independent RNG so the
    // main sampling sequence (and thus non-bimodal output) is unchanged.
    let base_len = if bimodal {
        ((in_high as f64) * cfg.cache_ratio).ceil() as usize
    } else {
        0
    };
    let base_tokens: Vec<u32> = if base_len > 0 {
        let mut br = StdRng::seed_from_u64(seed.wrapping_add(0xCACE));
        let seq: Vec<u32> = (0..base_len)
            .map(|_| allowed_tokens[br.gen_range(0..allowed_tokens.len())])
            .collect();
        // Round-trip through the tokenizer so decode(base_tokens[..c]) is a stable
        // byte-prefix (same assumption the 3-tier prefix-sharing path relies on).
        let (_text, toks) = gen_prompt_to_target_len(tokenizer, &seq, base_len)?;
        toks
    } else {
        Vec::new()
    };
    let base_avail = base_tokens.len();

    // Non-bimodal keeps its original shared prefix from --random-prefix-len.
    let shared_prefix_text = if bimodal {
        Arc::from("")
    } else {
        generate_shared_prefix_text(tokenizer, &allowed_tokens, prefix_len, seed)?
    };

    // Pre-generate per-conversation turn counts and per-turn offsets deterministically.
    // Turn counts are drawn first so the RNG sequence is stable regardless of vocab_size.
    let conv_turn_counts: Vec<usize> = (0..num_conversations)
        .map(|_| {
            if min_turns == max_turns {
                min_turns
            } else {
                rng.gen_range(min_turns..=max_turns)
            }
        })
        .collect();

    let offsets: Vec<Vec<usize>> = conv_turn_counts
        .iter()
        .map(|&n| (0..n).map(|_| rng.gen_range(0..vocab_size)).collect())
        .collect();

    // Per-conversation warm/cold flags (bimodal only), drawn from an independent RNG
    // to keep the main sequence pristine.
    let conv_warm: Vec<bool> = if bimodal {
        let mut wr = StdRng::seed_from_u64(seed.wrapping_add(0xF00D));
        (0..num_conversations)
            .map(|_| wr.gen::<f64>() < cfg.cache_hit_fraction)
            .collect()
    } else {
        Vec::new()
    };

    // Per-(conversation, turn) sampled input lengths (range-ratio only). Independent
    // RNG so the fixed default case consumes no extra randomness and stays identical.
    let turn_lens: Vec<Vec<usize>> = if use_range {
        let mut lr = StdRng::seed_from_u64(seed.wrapping_add(0xBEEF));
        conv_turn_counts
            .iter()
            .map(|&n| {
                (0..n)
                    .map(|t| {
                        let (lo, hi) = if t == 0 {
                            (in_low, in_high)
                        } else {
                            (pt_low, pt_high)
                        };
                        if lo >= hi {
                            lo
                        } else {
                            lr.gen_range(lo..=hi)
                        }
                    })
                    .collect()
            })
            .collect()
    } else {
        Vec::new()
    };

    // Parallel generation across conversations
    offsets
        .par_iter()
        .enumerate()
        .map(|(conv_idx, conv_offsets)| {
            let mut turns = Vec::with_capacity(conv_offsets.len());
            for (turn_idx, &offset) in conv_offsets.iter().enumerate() {
                let target_len = if use_range {
                    turn_lens[conv_idx][turn_idx]
                } else if turn_idx == 0 {
                    real_input_len
                } else {
                    real_per_turn_len
                };

                // Bimodal: warm conversations get a shared cached prefix on turn 0 that
                // counts toward target_len (cached + unique = target_len), so the unique
                // suffix shrinks accordingly.
                let cached = if bimodal && turn_idx == 0 && conv_warm[conv_idx] {
                    (((target_len as f64) * cfg.cache_ratio).round() as usize)
                        .min(base_avail)
                        .min(target_len)
                } else {
                    0
                };
                let suffix_len = target_len - cached;

                // Unique suffix tokens. Use max_turns stride to keep offsets unique
                // across variable-length conversations.
                let (suffix_text, suffix_tokens_len) = if suffix_len > 0 {
                    let inner_seq = make_token_seq(
                        &allowed_tokens,
                        offset + conv_idx * max_turns + turn_idx,
                        suffix_len,
                    );
                    let (text, toks) =
                        gen_prompt_to_target_len(tokenizer, &inner_seq, suffix_len)?;
                    (text, toks.len())
                } else {
                    (String::new(), 0)
                };

                let (user_message, token_len) = if cached > 0 {
                    // decode(base_tokens[..cached]) is a byte-prefix of the full base
                    // text, so all warm conversations share the same leading tokens.
                    let cached_text = tokenizer.decode(&base_tokens[..cached], true)?;
                    let combined = format!("{cached_text}{suffix_text}");
                    let token_len = tokenizer.encode(&combined, false)?.len();
                    (combined, token_len)
                } else if turn_idx == 0 && !shared_prefix_text.is_empty() {
                    let combined = format!("{}{}", &*shared_prefix_text, suffix_text);
                    let token_len = tokenizer.encode(&combined, false)?.len();
                    (combined, token_len)
                } else {
                    (suffix_text, suffix_tokens_len)
                };

                turns.push(ConversationTurn {
                    user_message: Arc::from(user_message),
                    user_message_len: token_len,
                    expected_output_len: output_len,
                });
            }

            Ok(MultiTurnConversation {
                conversation_id: format!("{request_id_prefix}conv-{conv_idx}"),
                turns,
            })
        })
        .collect()
}

/// Generate conversations with 3-tier prefix sharing.
///
/// Each turn's user message = [global_prefix][conversation_prefix][unique_suffix].
/// No history accumulation — each turn sends only its own fixed-length message.
fn generate_prefix_sharing_conversations(
    tokenizer: &TokenizerKind,
    cfg: &MultiTurnRandomConfig,
    ps_cfg: &PrefixSharingConfig,
    allowed_tokens: &[u32],
    rng: &mut StdRng,
) -> Result<Vec<MultiTurnConversation>> {
    let num_conversations = cfg.num_conversations;
    let min_turns = cfg.min_turns;
    let max_turns = cfg.max_turns;
    let prefix_len = cfg.prefix_len;
    let output_len = cfg.output_len;
    let request_id_prefix = &cfg.request_id_prefix;

    let num_special = tokenizer.num_special_tokens_to_add();
    let real_input_len = cfg.input_len.saturating_sub(num_special);
    let real_per_turn_len = if cfg.per_turn_input_len > 0 {
        cfg.per_turn_input_len.saturating_sub(num_special)
    } else {
        real_input_len
    };

    // Compute segment lengths from turn-0 (real_input_len) so the shared prefix
    // bytes stay byte-identical across all turns regardless of per_turn_input_len.
    let global_len = (real_input_len as f64 * ps_cfg.global_ratio).floor() as usize;
    let conv_len = (real_input_len as f64 * ps_cfg.conversation_ratio).floor() as usize;
    let unique_len = real_input_len.saturating_sub(global_len + conv_len);

    // Validate that turns 1+ still have room for a non-empty unique suffix
    if real_per_turn_len <= global_len + conv_len {
        return Err(BenchError::Config(format!(
            "--per-turn-input-len ({real_per_turn_len} after special tokens) is too small: \
             global_len={global_len} + conv_len={conv_len} already fills the budget. \
             Increase --per-turn-input-len or reduce prefix ratios."
        )));
    }

    let at_len = allowed_tokens.len();
    let shared_prefix_text =
        generate_shared_prefix_text(tokenizer, allowed_tokens, prefix_len, cfg.seed)?;

    // Generate global prefix text once
    let global_text: Arc<str> = if global_len > 0 {
        let offset: usize = rng.gen_range(0..at_len);
        let seq = make_token_seq(allowed_tokens, offset, global_len);
        let (text, _) = gen_prompt_to_target_len(tokenizer, &seq, global_len)?;
        Arc::from(text)
    } else {
        Arc::from("")
    };

    // Generate per-conversation prefix texts
    let conv_texts: Vec<Arc<str>> = if conv_len > 0 {
        let mut texts = Vec::with_capacity(num_conversations);
        for conv_idx in 0..num_conversations {
            let offset: usize = rng.gen_range(0..at_len);
            let seq = make_token_seq(allowed_tokens, offset + conv_idx, conv_len);
            let (text, _) = gen_prompt_to_target_len(tokenizer, &seq, conv_len)?;
            texts.push(Arc::from(text));
        }
        texts
    } else {
        vec![Arc::from(""); num_conversations]
    };

    // Pre-generate per-conversation turn counts and unique offsets deterministically.
    let vocab_size = tokenizer.vocab_size() as usize;
    let conv_turn_counts: Vec<usize> = (0..num_conversations)
        .map(|_| {
            if min_turns == max_turns {
                min_turns
            } else {
                rng.gen_range(min_turns..=max_turns)
            }
        })
        .collect();

    let unique_offsets: Vec<Vec<usize>> = conv_turn_counts
        .iter()
        .map(|&n| (0..n).map(|_| rng.gen_range(0..vocab_size)).collect())
        .collect();

    // Parallel generation across conversations
    unique_offsets
        .par_iter()
        .enumerate()
        .map(|(conv_idx, conv_offsets)| {
            let mut turns = Vec::with_capacity(conv_offsets.len());
            for (turn_idx, &offset) in conv_offsets.iter().enumerate() {
                // Turn 0 uses unique_len derived from real_input_len;
                // turns 1+ use per-turn unique_len (prefix bytes stay identical).
                let turn_unique_len = if turn_idx == 0 {
                    unique_len
                } else {
                    real_per_turn_len.saturating_sub(global_len + conv_len)
                };

                // Generate unique suffix
                let unique_text = if turn_unique_len > 0 {
                    let seq = make_token_seq(
                        allowed_tokens,
                        offset + conv_idx * max_turns + turn_idx,
                        turn_unique_len,
                    );
                    let (text, _) = gen_prompt_to_target_len(tokenizer, &seq, turn_unique_len)?;
                    text
                } else {
                    String::new()
                };

                // Concatenate: optional random prefix + global + conversation + unique.
                // Prefix-sharing mode sends each turn independently, so the random
                // prefix must be included on every turn to be present in every request.
                let combined = format!(
                    "{}{}{}{}",
                    &*shared_prefix_text, &*global_text, &*conv_texts[conv_idx], unique_text
                );
                // Re-encode to get actual token count (BPE boundary effects)
                let token_len = tokenizer.encode(&combined, false)?.len();

                turns.push(ConversationTurn {
                    user_message: Arc::from(combined),
                    user_message_len: token_len,
                    expected_output_len: output_len,
                });
            }

            Ok(MultiTurnConversation {
                conversation_id: format!("{request_id_prefix}conv-{conv_idx}"),
                turns,
            })
        })
        .collect()
}

fn generate_shared_prefix_text(
    tokenizer: &TokenizerKind,
    allowed_tokens: &[u32],
    prefix_len: usize,
    seed: u64,
) -> Result<Arc<str>> {
    if prefix_len == 0 {
        return Ok(Arc::from(""));
    }

    let mut rng = StdRng::seed_from_u64(seed.wrapping_add(0xDEAD));
    let tokens: Vec<u32> = (0..prefix_len)
        .map(|_| allowed_tokens[rng.gen_range(0..allowed_tokens.len())])
        .collect();
    let (text, _) = gen_prompt_to_target_len(tokenizer, &tokens, prefix_len)?;
    Ok(Arc::from(text))
}

/// Load multi-turn conversations from a ShareGPT dataset.
///
/// Walks ALL turns in each entry (not just first 2). Filters entries
/// with at least 4 messages (2 user + 2 assistant = 2 real turns).
pub fn load_sharegpt_multi_turn(
    tokenizer: &TokenizerKind,
    dataset_path: &str,
    num_conversations: usize,
    output_len_override: Option<usize>,
    max_turns: Option<usize>,
    seed: u64,
    request_id_prefix: &str,
) -> Result<Vec<MultiTurnConversation>> {
    let content = std::fs::read_to_string(dataset_path).map_err(|e| {
        BenchError::Config(format!(
            "Failed to read ShareGPT file '{dataset_path}': {e}"
        ))
    })?;

    let data: serde_json::Value = serde_json::from_str(&content)
        .map_err(|e| BenchError::Config(format!("Invalid JSON in ShareGPT file: {e}")))?;

    let entries = data
        .as_array()
        .ok_or_else(|| BenchError::Config("ShareGPT file must contain a JSON array".into()))?;

    // Filter entries with at least 4 messages (2 turns: user+assistant+user+assistant)
    let mut filtered: Vec<&serde_json::Value> = entries
        .iter()
        .filter(|entry| {
            entry
                .get("conversations")
                .and_then(|c| c.as_array())
                .map(|a| a.len() >= 4)
                .unwrap_or(false)
        })
        .collect();

    if filtered.is_empty() {
        return Err(BenchError::Config(
            "No valid multi-turn entries in ShareGPT file (need at least 4 messages per entry)"
                .into(),
        ));
    }

    // Shuffle
    let mut rng = StdRng::seed_from_u64(seed);
    filtered.shuffle(&mut rng);

    let mut conversations = Vec::new();

    for entry in &filtered {
        if conversations.len() >= num_conversations {
            break;
        }

        let msgs = entry["conversations"].as_array().unwrap();
        let mut turns = Vec::new();

        // Walk alternating human/gpt pairs, stopping early once max_turns reached
        // to avoid tokenizing turns that would be discarded by truncate().
        let mut i = 0;
        while i + 1 < msgs.len() {
            if let Some(m) = max_turns {
                if turns.len() >= m {
                    break;
                }
            }
            let from = msgs[i].get("from").and_then(|f| f.as_str()).unwrap_or("");
            let user_text = msgs[i].get("value").and_then(|v| v.as_str()).unwrap_or("");
            let assistant_text = msgs[i + 1]
                .get("value")
                .and_then(|v| v.as_str())
                .unwrap_or("");

            // Expect human then gpt
            if from != "human" || user_text.is_empty() {
                i += 1;
                continue;
            }

            let user_ids = tokenizer.encode(user_text, false)?;
            let user_len = user_ids.len();

            let expected_output_len = if let Some(override_len) = output_len_override {
                override_len
            } else {
                let assistant_ids = tokenizer.encode(assistant_text, false)?;
                assistant_ids.len().max(1)
            };

            turns.push(ConversationTurn {
                user_message: Arc::from(user_text),
                user_message_len: user_len,
                expected_output_len,
            });

            i += 2;
        }

        if turns.len() >= 2 {
            let conv_idx = conversations.len();
            conversations.push(MultiTurnConversation {
                conversation_id: format!("{request_id_prefix}conv-{conv_idx}"),
                turns,
            });
        }
    }

    if conversations.is_empty() {
        return Err(BenchError::Config(
            "No valid multi-turn conversations after filtering ShareGPT dataset.".into(),
        ));
    }

    // Oversample if needed
    if conversations.len() < num_conversations {
        let original_len = conversations.len();
        let needed = num_conversations - original_len;
        for i in 0..needed {
            let mut conv = conversations[rng.gen_range(0..original_len)].clone();
            conv.conversation_id = format!("{request_id_prefix}conv-{}", original_len + i);
            conversations.push(conv);
        }
        println!(
            "Oversampled multi-turn conversations from {original_len} to {} total.",
            conversations.len()
        );
    }

    Ok(conversations)
}

/// Ensure decoded-then-encoded prompt length matches the target.
fn gen_prompt_to_target_len(
    tokenizer: &TokenizerKind,
    token_sequence: &[u32],
    target_len: usize,
) -> Result<(String, Vec<u32>)> {
    let max_retry = 20;
    let mut tokens = token_sequence.to_vec();

    for retry in 0..=max_retry {
        let prompt = tokenizer.decode(&tokens, true)?;
        tokens = tokenizer.encode(&prompt, false)?;

        if retry >= max_retry {
            // BPE tokenizers can oscillate by ±1 on certain boundaries.
            // For benchmark random content, accept close-enough and truncate/pad.
            if tokens.len() > target_len {
                tokens.truncate(target_len);
            }
            // If still short by 1-2 tokens, accept as-is — negligible for benchmarks.
            // Re-decode after truncation to ensure prompt string matches token vector.
            let prompt = tokenizer.decode(&tokens, true)?;
            return Ok((prompt, tokens));
        }

        if tokens.len() == target_len {
            return Ok((prompt, tokens));
        } else if tokens.len() < target_len {
            let allowed = tokenizer.get_allowed_tokens();
            let needed = target_len - tokens.len();
            if allowed.is_empty() {
                let vocab_size = tokenizer.vocab_size() as usize;
                for j in 0..needed {
                    tokens.push(((tokens.len() + j) % vocab_size) as u32);
                }
            } else {
                for j in 0..needed {
                    tokens.push(allowed[(tokens.len() + j) % allowed.len()]);
                }
            }
        } else {
            tokens.truncate(target_len);
        }
    }

    unreachable!()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn common_prefix_bytes(strings: &[&str]) -> usize {
        if strings.is_empty() {
            return 0;
        }
        let first = strings[0].as_bytes();
        let mut len = first.len();
        for s in &strings[1..] {
            let b = s.as_bytes();
            len = len.min(b.len());
            for i in 0..len {
                if first[i] != b[i] {
                    len = i;
                    break;
                }
            }
        }
        len
    }

    #[test]
    #[ignore]
    fn test_prefix_sharing_structure() {
        let tok = crate::tokenizer::load_tokenizer("nvidia/Kimi-K2.5-NVFP4", false, None).unwrap();

        let cfg = MultiTurnRandomConfig {
            num_conversations: 5,
            min_turns: 3,
            max_turns: 3,
            prefix_len: 0,
            input_len: 1000,
            per_turn_input_len: 0,
            output_len: 100,
            seed: 42,
            request_id_prefix: "test-".to_string(),
            range_ratio: crate::config::RangeRatio {
                input: 0.0,
                output: 0.0,
            },
            cache_hit_fraction: 0.0,
            cache_ratio: 0.0,
            prefix_sharing_config: Some(PrefixSharingConfig {
                global_ratio: 0.1,
                conversation_ratio: 0.8,
            }),
        };

        let conversations = generate_multi_turn_random(&tok, &cfg).unwrap();
        assert_eq!(conversations.len(), 5);

        let messages: Vec<Vec<&str>> = conversations
            .iter()
            .map(|c| c.turns.iter().map(|t| &*t.user_message).collect())
            .collect();

        // 1. Global prefix: all messages share a common prefix
        let all_msgs: Vec<&str> = messages.iter().flat_map(|v| v.iter().copied()).collect();
        let global_prefix = common_prefix_bytes(&all_msgs);
        println!("Global prefix bytes: {global_prefix}");
        assert!(global_prefix > 0, "Global prefix must be non-empty");

        // 2. Conversation prefix: turns within same conversation share more
        for (i, conv_msgs) in messages.iter().enumerate() {
            let conv_prefix = common_prefix_bytes(conv_msgs);
            println!("Conv {i} prefix bytes: {conv_prefix} (global: {global_prefix})");
            assert!(
                conv_prefix > global_prefix,
                "Conv prefix ({conv_prefix}) must exceed global prefix ({global_prefix})"
            );
        }

        // 3. Different conversations diverge after global prefix
        let cross = common_prefix_bytes(&[messages[0][0], messages[1][0]]);
        let within = common_prefix_bytes(&messages[0]);
        println!("Cross-conv prefix: {cross}, within-conv prefix: {within}");
        assert!(
            cross < within,
            "Cross-conv ({cross}) must be < within-conv ({within})"
        );

        // 4. Turns within same conversation are not identical (unique suffix)
        for (i, conv_msgs) in messages.iter().enumerate() {
            for a in 0..conv_msgs.len() {
                for b in (a + 1)..conv_msgs.len() {
                    assert_ne!(
                        conv_msgs[a], conv_msgs[b],
                        "Conv {i} turn {a} and {b} must differ"
                    );
                }
            }
        }

        // 5. Token lengths approximately match target
        for (i, conv) in conversations.iter().enumerate() {
            for (j, turn) in conv.turns.iter().enumerate() {
                let diff = (turn.user_message_len as i64 - 1000).abs();
                println!(
                    "Conv {i} turn {j}: {} tokens (diff {diff})",
                    turn.user_message_len
                );
                assert!(
                    diff <= 10,
                    "Token len {} too far from 1000",
                    turn.user_message_len
                );
            }
        }

        println!("All prefix sharing checks passed!");
    }

    #[test]
    #[ignore]
    fn test_per_turn_input_len_default_mode() {
        let tok = crate::tokenizer::load_tokenizer("nvidia/Kimi-K2.5-NVFP4", false, None).unwrap();

        let cfg = MultiTurnRandomConfig {
            num_conversations: 4,
            min_turns: 3,
            max_turns: 3,
            prefix_len: 0,
            input_len: 512,
            per_turn_input_len: 128,
            output_len: 64,
            seed: 1,
            request_id_prefix: "test-".to_string(),
            range_ratio: crate::config::RangeRatio {
                input: 0.0,
                output: 0.0,
            },
            cache_hit_fraction: 0.0,
            cache_ratio: 0.0,
            prefix_sharing_config: None,
        };

        let conversations = generate_multi_turn_random(&tok, &cfg).unwrap();
        assert_eq!(conversations.len(), 4);

        for (i, conv) in conversations.iter().enumerate() {
            assert_eq!(conv.turns.len(), 3);
            for (j, turn) in conv.turns.iter().enumerate() {
                let expected = if j == 0 { 512usize } else { 128usize };
                let diff = (turn.user_message_len as i64 - expected as i64).abs();
                println!(
                    "Conv {i} turn {j}: {} tokens (expected ~{expected}, diff {diff})",
                    turn.user_message_len
                );
                assert!(
                    diff <= 5,
                    "Conv {i} turn {j}: token len {} too far from {expected}",
                    turn.user_message_len
                );
            }
        }
        println!("per_turn_input_len default-mode checks passed!");
    }

    #[test]
    #[ignore]
    fn test_variable_turns_range() {
        let tok = crate::tokenizer::load_tokenizer("nvidia/Kimi-K2.5-NVFP4", false, None).unwrap();

        let cfg = MultiTurnRandomConfig {
            num_conversations: 50,
            min_turns: 2,
            max_turns: 5,
            prefix_len: 0,
            input_len: 256,
            per_turn_input_len: 0,
            output_len: 32,
            seed: 7,
            request_id_prefix: "test-".to_string(),
            range_ratio: crate::config::RangeRatio {
                input: 0.0,
                output: 0.0,
            },
            cache_hit_fraction: 0.0,
            cache_ratio: 0.0,
            prefix_sharing_config: None,
        };

        let conversations = generate_multi_turn_random(&tok, &cfg).unwrap();
        assert_eq!(conversations.len(), 50);

        let mut distinct_counts = std::collections::HashSet::new();
        for conv in &conversations {
            let n = conv.turns.len();
            assert!((2..=5).contains(&n), "turn count {n} out of [2,5]");
            distinct_counts.insert(n);
        }
        assert!(
            distinct_counts.len() >= 2,
            "expected at least 2 distinct turn counts, got {distinct_counts:?}"
        );
        println!("variable_turns_range checks passed! counts: {distinct_counts:?}");
    }

    #[test]
    #[ignore]
    fn test_variable_turns_fixed() {
        let tok = crate::tokenizer::load_tokenizer("nvidia/Kimi-K2.5-NVFP4", false, None).unwrap();

        let cfg = MultiTurnRandomConfig {
            num_conversations: 10,
            min_turns: 4,
            max_turns: 4,
            prefix_len: 0,
            input_len: 256,
            per_turn_input_len: 0,
            output_len: 32,
            seed: 42,
            request_id_prefix: "test-".to_string(),
            range_ratio: crate::config::RangeRatio {
                input: 0.0,
                output: 0.0,
            },
            cache_hit_fraction: 0.0,
            cache_ratio: 0.0,
            prefix_sharing_config: None,
        };

        let conversations = generate_multi_turn_random(&tok, &cfg).unwrap();
        for conv in &conversations {
            assert_eq!(conv.turns.len(), 4, "expected exactly 4 turns");
        }
        println!("variable_turns_fixed checks passed!");
    }

    #[test]
    #[ignore]
    fn test_per_turn_input_len_prefix_sharing() {
        let tok = crate::tokenizer::load_tokenizer("nvidia/Kimi-K2.5-NVFP4", false, None).unwrap();

        // Turn 0 input_len=1000, turns 1+ per_turn_input_len=600
        // global_len ≈ 100 (10%), conv_len ≈ 800 (80%), unique ≈ 100
        // per-turn unique ≈ 600 - 900 = negative → would error; use smaller ratios
        // global=0.05 (50), conv=0.5 (500), unique_t0=450, unique_t1=600-550=50
        let cfg = MultiTurnRandomConfig {
            num_conversations: 4,
            min_turns: 3,
            max_turns: 3,
            prefix_len: 0,
            input_len: 1000,
            per_turn_input_len: 600,
            output_len: 64,
            seed: 3,
            request_id_prefix: "test-".to_string(),
            range_ratio: crate::config::RangeRatio {
                input: 0.0,
                output: 0.0,
            },
            cache_hit_fraction: 0.0,
            cache_ratio: 0.0,
            prefix_sharing_config: Some(PrefixSharingConfig {
                global_ratio: 0.05,
                conversation_ratio: 0.50,
            }),
        };

        let conversations = generate_multi_turn_random(&tok, &cfg).unwrap();
        assert_eq!(conversations.len(), 4);

        let messages: Vec<Vec<&str>> = conversations
            .iter()
            .map(|c| c.turns.iter().map(|t| &*t.user_message).collect())
            .collect();

        // Global prefix bytes shared across all turns of all conversations
        let all_msgs: Vec<&str> = messages.iter().flat_map(|v| v.iter().copied()).collect();
        let global_prefix = common_prefix_bytes(&all_msgs);
        assert!(global_prefix > 0, "Global prefix must be non-empty");

        // Within each conversation, prefix grows (conv prefix longer than global)
        for (i, conv_msgs) in messages.iter().enumerate() {
            let conv_prefix = common_prefix_bytes(conv_msgs);
            assert!(
                conv_prefix > global_prefix,
                "Conv {i}: conv_prefix ({conv_prefix}) must exceed global ({global_prefix})"
            );
        }

        // Turn 0 length ≈ 1000, turns 1+ ≈ 600
        for (i, conv) in conversations.iter().enumerate() {
            for (j, turn) in conv.turns.iter().enumerate() {
                let expected = if j == 0 { 1000usize } else { 600usize };
                let diff = (turn.user_message_len as i64 - expected as i64).abs();
                println!(
                    "Conv {i} turn {j}: {} tokens (expected ~{expected})",
                    turn.user_message_len
                );
                assert!(
                    diff <= 10,
                    "Conv {i} turn {j}: token len {} too far from {expected}",
                    turn.user_message_len
                );
            }
        }
        println!("per_turn_input_len prefix-sharing checks passed!");
    }

    #[test]
    #[ignore]
    fn test_bimodal_prefix_cache_multi_turn() {
        let tok = crate::tokenizer::load_tokenizer("nvidia/Kimi-K2.5-NVFP4", false, None).unwrap();

        // 80% warm, 95% cached. Many conversations so the warm fraction is observable.
        let cfg = MultiTurnRandomConfig {
            num_conversations: 200,
            min_turns: 2,
            max_turns: 2,
            prefix_len: 0,
            input_len: 1000,
            per_turn_input_len: 200,
            output_len: 64,
            seed: 7,
            request_id_prefix: "test-".to_string(),
            range_ratio: crate::config::RangeRatio {
                input: 0.0,
                output: 0.0,
            },
            cache_hit_fraction: 0.8,
            cache_ratio: 0.95,
            prefix_sharing_config: None,
        };

        let conversations = generate_multi_turn_random(&tok, &cfg).unwrap();
        assert_eq!(conversations.len(), 200);

        // A conversation is "warm" if its turn-0 message begins with the shared cached
        // base prefix. All warm turn-0 messages must share a large common byte prefix
        // (~95% of 1000 tokens); cold ones do not.
        let turn0: Vec<&str> = conversations
            .iter()
            .map(|c| &*c.turns[0].user_message)
            .collect();

        // Group by whether each shares a long prefix with the most common leading run.
        // Use the observed max pairwise prefix as the warm signal: warm vs warm share
        // hundreds of bytes, warm vs cold share ~0.
        let mut warm_count = 0usize;
        for (i, &a) in turn0.iter().enumerate() {
            // count how many OTHER turn-0 messages share a big prefix with this one
            let shared_big = turn0
                .iter()
                .enumerate()
                .filter(|(j, &b)| *j != i && common_prefix_bytes(&[a, b]) > 100)
                .count();
            if shared_big > 0 {
                warm_count += 1;
            }
        }
        let warm_frac = warm_count as f64 / turn0.len() as f64;
        println!("Observed warm fraction: {warm_frac:.3} (target 0.8)");
        assert!(
            (0.65..=0.95).contains(&warm_frac),
            "warm fraction {warm_frac} not near cache_hit_fraction 0.8"
        );

        // All warm conversations must share a common leading run (they slice the same
        // base). Collect warm turn-0 messages and check their common prefix is large.
        let warm_msgs: Vec<&str> = turn0
            .iter()
            .copied()
            .filter(|&a| {
                turn0
                    .iter()
                    .any(|&b| !std::ptr::eq(a, b) && common_prefix_bytes(&[a, b]) > 100)
            })
            .collect();
        let warm_common = common_prefix_bytes(&warm_msgs);
        println!("Warm-set common prefix: {warm_common} bytes");
        assert!(
            warm_common > 100,
            "warm conversations must share a large cached prefix, got {warm_common} bytes"
        );

        // Turn 0 length ≈ 1000 for every conversation (warm cached + suffix = 1000).
        for conv in &conversations {
            let diff = (conv.turns[0].user_message_len as i64 - 1000).abs();
            assert!(diff <= 15, "turn-0 len {} far from 1000", conv.turns[0].user_message_len);
        }
        println!("bimodal multi-turn prefix-cache checks passed!");
    }

    #[test]
    #[ignore]
    fn test_range_ratio_varies_lengths_multi_turn() {
        let tok = crate::tokenizer::load_tokenizer("nvidia/Kimi-K2.5-NVFP4", false, None).unwrap();

        let cfg = MultiTurnRandomConfig {
            num_conversations: 40,
            min_turns: 2,
            max_turns: 2,
            prefix_len: 0,
            input_len: 1000,
            per_turn_input_len: 0,
            output_len: 64,
            seed: 11,
            request_id_prefix: "test-".to_string(),
            range_ratio: crate::config::RangeRatio {
                input: 0.5,
                output: 0.0,
            },
            cache_hit_fraction: 0.0,
            cache_ratio: 0.0,
            prefix_sharing_config: None,
        };

        let conversations = generate_multi_turn_random(&tok, &cfg).unwrap();
        let turn0_lens: Vec<usize> = conversations
            .iter()
            .map(|c| c.turns[0].user_message_len)
            .collect();

        let min = *turn0_lens.iter().min().unwrap();
        let max = *turn0_lens.iter().max().unwrap();
        println!("range-ratio 0.5 turn-0 lengths: min={min}, max={max}");
        // Expect a spread within [~500, ~1500] and clearly non-constant.
        assert!(max - min > 100, "range_ratio 0.5 should vary lengths, spread was {}", max - min);
        assert!(min >= 400 && max <= 1600, "lengths {min}..{max} out of expected band");
    }
}
