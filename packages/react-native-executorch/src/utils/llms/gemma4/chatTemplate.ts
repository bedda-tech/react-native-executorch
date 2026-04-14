/**
 * Gemma 4 chat template utilities for on-device use in ExecuTorch.
 *
 * The Gemma 4 instruct model uses the following turn structure:
 *
 *   <bos><start_of_turn>user
 *   {message}<end_of_turn>
 *   <start_of_turn>model
 *   {response}<end_of_turn>
 *
 * For function calling, tools are injected into the system turn as a JSON
 * schema block, and the model emits <function_calls> XML blocks.
 *
 * This module provides:
 *  - A hardcoded JavaScript chat template (fallback / reference when
 *    tokenizer_config.json is unavailable).
 *  - Helpers to build properly-formatted prompts from message arrays.
 *  - The `gemma4TokenizerConfig` object that can be passed as
 *    `tokenizerConfigSource` override when using useLLM with a custom loader.
 *
 * @module gemma4/chatTemplate
 */

import { Message, LLMTool } from '../../../types/llm';

// ---------------------------------------------------------------------------
// Special tokens
// ---------------------------------------------------------------------------

/** Gemma 4 special tokens. */
export const GEMMA4_TOKENS = {
  BOS: '<bos>',
  EOS: '<eos>',
  PAD: '<pad>',
  START_OF_TURN: '<start_of_turn>',
  END_OF_TURN: '<end_of_turn>',
  /** Emitted by the model when beginning a function call block. */
  FUNCTION_CALLS_OPEN: '<function_calls>',
  /** Emitted by the model when closing a function call block. */
  FUNCTION_CALLS_CLOSE: '</function_calls>',
} as const;

// ---------------------------------------------------------------------------
// tokenizer_config.json shape (partial — only what we need at runtime)
// ---------------------------------------------------------------------------

/**
 * Minimal shape of Gemma 4's tokenizer_config.json that is needed by the
 * LLMController to apply the chat template and filter special tokens.
 */
// Property names mirror the HuggingFace tokenizer_config.json schema (snake_case is intentional)
/* eslint-disable camelcase */
export const GEMMA4_TOKENIZER_CONFIG = {
  bos_token: GEMMA4_TOKENS.BOS,
  eos_token: GEMMA4_TOKENS.EOS,
  pad_token: GEMMA4_TOKENS.PAD,
  unk_token: '<unk>',
  /**
   * Jinja2 chat template matching the official Gemma 4 instruct template.
   * This mirrors the template shipped in the Gemma 4 HuggingFace checkpoint
   * and is used by LLMController.applyChatTemplate() via @huggingface/jinja.
   *
   * Key behaviours:
   *  - System turns are rendered as `<start_of_turn>user\n{content}<end_of_turn>`
   *    (Gemma 4 merges the system prompt into the first user turn by default,
   *     but the instruct variant accepts an explicit system role).
   *  - If tools are provided, a tool schema block is prepended to the first
   *    user message.
   *  - `add_generation_prompt` appends the opening `<start_of_turn>model\n`
   *    marker so that the model continues with its response.
   */
  // prettier-ignore
  chat_template:
    `{%- if messages[0]['role'] == 'system' -%}` +
    `{%- set system_message = messages[0]['content'] -%}` +
    `{%- set messages = messages[1:] -%}` +
    `{%- else -%}` +
    `{%- set system_message = '' -%}` +
    `{%- endif -%}` +
    `{{ bos_token }}` +
    `{%- if tools -%}` +
    `<start_of_turn>user\n` +
    `{%- if system_message -%}{{ system_message }}\n\n{%- endif -%}` +
    `You have access to the following tools:\n\n` +
    `{{ tools | tojson(indent=2) }}\n\n` +
    `To use a tool, output a <function_calls> block:\n` +
    `<function_calls>\n` +
    `[{"name": "tool_name", "arguments": {"arg": "value"}}]\n` +
    `</function_calls>\n\n` +
    `{%- for message in messages -%}` +
    `{%- if loop.first and message['role'] == 'user' -%}` +
    `{{ message['content'] }}<end_of_turn>\n` +
    `{%- else -%}` +
    `<start_of_turn>{{ message['role'] }}\n{{ message['content'] }}<end_of_turn>\n` +
    `{%- endif -%}` +
    `{%- endfor -%}` +
    `{%- else -%}` +
    `{%- for message in messages -%}` +
    `<start_of_turn>{{ message['role'] }}\n` +
    `{%- if loop.first and system_message -%}{{ system_message }}\n\n{%- endif -%}` +
    `{{ message['content'] }}<end_of_turn>\n` +
    `{%- endfor -%}` +
    `{%- endif -%}` +
    `{%- if add_generation_prompt -%}<start_of_turn>model\n{%- endif -%}`,
} as const;
/* eslint-enable camelcase */

// ---------------------------------------------------------------------------
// Pure-JS chat template formatter (no Jinja dependency)
// ---------------------------------------------------------------------------

/**
 * Options for `formatGemma4Prompt`.
 */
export interface FormatGemma4Options {
  /** Whether to append the opening generation-prompt marker. Default: true. */
  addGenerationPrompt?: boolean;
  /** Tools to make available. When provided they are embedded in the prompt. */
  tools?: LLMTool[];
}

/**
 * Formats an array of `Message` objects into a Gemma 4 instruct prompt string
 * using pure JavaScript (no Jinja runtime required).
 *
 * Use this when you want to manually build a prompt without going through the
 * LLMController / useLLM hook — e.g. in unit tests or in a custom inference loop.
 *
 * @param messages  Conversation turns (system, user, assistant).
 * @param options   Formatting options.
 * @returns         A fully-formatted prompt string ready for the ExecuTorch model.
 *
 * @example
 * ```ts
 * const prompt = formatGemma4Prompt([
 *   { role: 'system', content: 'You are a helpful phone agent.' },
 *   { role: 'user',   content: 'What apps are open?' },
 * ]);
 * ```
 */
export function formatGemma4Prompt(
  messages: Message[],
  options: FormatGemma4Options = {}
): string {
  const { addGenerationPrompt = true, tools } = options;

  // Extract optional system message
  let systemMessage = '';
  let conversationMessages = messages;
  if (messages.length > 0 && messages[0]!.role === 'system') {
    systemMessage = messages[0]!.content;
    conversationMessages = messages.slice(1);
  }

  let prompt = GEMMA4_TOKENS.BOS;

  if (tools && tools.length > 0) {
    // Tool-calling mode: inject tool schema into a synthetic opening turn
    prompt += `${GEMMA4_TOKENS.START_OF_TURN}user\n`;
    if (systemMessage) {
      prompt += `${systemMessage}\n\n`;
    }
    prompt += buildToolInstructions(tools);

    // Append the first user message directly (no extra <start_of_turn>)
    if (
      conversationMessages.length > 0 &&
      conversationMessages[0]!.role === 'user'
    ) {
      prompt += `${conversationMessages[0]!.content}${GEMMA4_TOKENS.END_OF_TURN}\n`;
      conversationMessages = conversationMessages.slice(1);
    }

    // Remaining turns
    for (const msg of conversationMessages) {
      prompt += formatTurn(msg);
    }
  } else {
    // Plain chat mode
    for (let i = 0; i < conversationMessages.length; i++) {
      const msg = conversationMessages[i]!;
      prompt += `${GEMMA4_TOKENS.START_OF_TURN}${msg.role}\n`;
      if (i === 0 && systemMessage) {
        prompt += `${systemMessage}\n\n`;
      }
      prompt += `${msg.content}${GEMMA4_TOKENS.END_OF_TURN}\n`;
    }
  }

  if (addGenerationPrompt) {
    prompt += `${GEMMA4_TOKENS.START_OF_TURN}model\n`;
  }

  return prompt;
}

// ---------------------------------------------------------------------------
// Tool schema builder
// ---------------------------------------------------------------------------

/**
 * Builds the tool-instruction block that is prepended to the user turn when
 * tool calling is enabled.
 *
 * The format matches what the Gemma 4 instruct model was trained on:
 * a JSON array of tool schemas followed by usage instructions.
 *
 * @param tools  Array of tool definition objects.
 * @returns      Formatted instruction string (no leading/trailing newlines).
 */
export function buildToolInstructions(tools: LLMTool[]): string {
  const schema = JSON.stringify(tools, null, 2);
  return (
    `You have access to the following tools:\n\n` +
    `${schema}\n\n` +
    `To use a tool, output a ${GEMMA4_TOKENS.FUNCTION_CALLS_OPEN} block:\n` +
    `${GEMMA4_TOKENS.FUNCTION_CALLS_OPEN}\n` +
    `[{"name": "tool_name", "arguments": {"arg": "value"}}]\n` +
    `${GEMMA4_TOKENS.FUNCTION_CALLS_CLOSE}\n\n`
  );
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

function formatTurn(msg: Message): string {
  return (
    `${GEMMA4_TOKENS.START_OF_TURN}${msg.role}\n` +
    `${msg.content}${GEMMA4_TOKENS.END_OF_TURN}\n`
  );
}
