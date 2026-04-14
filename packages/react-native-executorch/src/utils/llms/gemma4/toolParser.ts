/**
 * Gemma 4 tool-call parser.
 *
 * Gemma 4 emits function calls as an XML-like block:
 *
 *   <function_calls>
 *   [{"name": "tap", "arguments": {"nodeId": "settings-icon-42"}}]
 *   </function_calls>
 *
 * This module parses that output into typed `Gemma4ToolCall` objects.
 *
 * @module gemma4/toolParser
 */

import { ToolCall } from '../../../types/llm';
import { Logger } from '../../../common/Logger';
import { jsonrepair } from 'jsonrepair';
import { GEMMA4_TOKENS } from './chatTemplate';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/**
 * A single parsed tool call emitted by Gemma 4.
 * Extends the base `ToolCall` type with a typed `arguments` field.
 */
export interface Gemma4ToolCall extends ToolCall {
  toolName: string;
  arguments: Record<string, unknown>;
}

/**
 * Result of parsing a model response that may or may not contain tool calls.
 */
export interface Gemma4ParseResult {
  /** Parsed tool calls (empty array if none were found). */
  toolCalls: Gemma4ToolCall[];
  /**
   * Any text that appears outside the function_calls block.
   * May be a reasoning preamble or empty string.
   */
  textContent: string;
  /** True if a <function_calls> block was found (even if parsing failed). */
  hasFunctionCalls: boolean;
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

/**
 * Parses a raw Gemma 4 model response and extracts tool calls.
 *
 * Handles:
 *  - Well-formed JSON arrays inside `<function_calls>` blocks.
 *  - Slightly malformed JSON (via jsonrepair).
 *  - Multiple `<function_calls>` blocks in one response (rare, but defensive).
 *  - Responses with no tool calls (returns empty array + full text).
 *
 * @param response  Raw string output from the model.
 * @returns         Parsed result containing tool calls and plain text.
 *
 * @example
 * ```ts
 * const result = parseGemma4Response(
 *   '<function_calls>\n[{"name":"tap","arguments":{"nodeId":"btn-42"}}]\n</function_calls>'
 * );
 * // result.toolCalls[0] => { toolName: 'tap', arguments: { nodeId: 'btn-42' } }
 * ```
 */
export function parseGemma4Response(response: string): Gemma4ParseResult {
  const openTag = GEMMA4_TOKENS.FUNCTION_CALLS_OPEN;
  const closeTag = GEMMA4_TOKENS.FUNCTION_CALLS_CLOSE;

  const hasFunctionCalls = response.includes(openTag);

  if (!hasFunctionCalls) {
    return {
      toolCalls: [],
      textContent: response.trim(),
      hasFunctionCalls: false,
    };
  }

  const toolCalls: Gemma4ToolCall[] = [];
  let textContent = response;

  // Extract all <function_calls>...</function_calls> blocks
  const blockRegex = new RegExp(
    escapeRegex(openTag) + '([\\s\\S]*?)' + escapeRegex(closeTag),
    'g'
  );

  let match: RegExpExecArray | null;
  // eslint-disable-next-line no-cond-assign
  while ((match = blockRegex.exec(response)) !== null) {
    const rawBlock = match[1]!.trim();

    // Remove the matched block from textContent
    textContent = textContent.replace(match[0], '').trim();

    try {
      const repaired = jsonrepair(rawBlock);
      const parsed: unknown = JSON.parse(repaired);

      if (!Array.isArray(parsed)) {
        Logger.warn(
          '[Gemma4] function_calls block did not parse to an array:',
          rawBlock
        );
        continue;
      }

      for (const item of parsed) {
        const call = extractToolCall(item);
        if (call) {
          toolCalls.push(call);
        }
      }
    } catch (e) {
      Logger.error(
        '[Gemma4] Failed to parse function_calls block:',
        e,
        rawBlock
      );
    }
  }

  return { toolCalls, textContent, hasFunctionCalls };
}

/**
 * Convenience wrapper that only returns the tool calls array.
 * Matches the signature of the existing `parseToolCall` utility so it can be
 * used as a drop-in replacement in the LLMController when model is Gemma 4.
 * @param response - Raw model output string to parse.
 * @returns Array of extracted tool call objects.
 */
export function parseGemma4ToolCalls(response: string): ToolCall[] {
  return parseGemma4Response(response).toolCalls;
}

// ---------------------------------------------------------------------------
// Tool schema helpers
// ---------------------------------------------------------------------------

/**
 * Defines a single tool that can be passed to Gemma 4 for function calling.
 * This is the format Gemma 4 was trained to understand.
 */
export interface Gemma4Tool {
  /** Unique tool name (snake_case recommended). */
  name: string;
  /** Short description telling the model when to use this tool. */
  description: string;
  /** JSON Schema object describing the tool's parameters. */
  parameters: {
    type: 'object';
    properties: Record<
      string,
      {
        type: string;
        description?: string;
        enum?: string[];
      }
    >;
    required?: string[];
  };
}

/**
 * Builds a minimal tool definition object in the format Gemma 4 expects.
 * Use this to construct tool arrays for `formatGemma4Prompt` or `configure()`.
 *
 * @param tool - Tool definition (name, description, parameters).
 * @returns The same tool object, typed as {@link Gemma4Tool}.
 * @example
 * ```ts
 * const tapTool = buildGemma4Tool({
 *   name: 'tap',
 *   description: 'Tap a UI element by accessibility node ID',
 *   parameters: {
 *     type: 'object',
 *     properties: {
 *       nodeId: { type: 'string', description: 'Accessibility node ID' },
 *     },
 *     required: ['nodeId'],
 *   },
 * });
 * ```
 */
export function buildGemma4Tool(tool: Gemma4Tool): Gemma4Tool {
  return tool;
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

function extractToolCall(item: unknown): Gemma4ToolCall | null {
  if (item === null || typeof item !== 'object' || Array.isArray(item)) {
    return null;
  }

  const obj = item as Record<string, unknown>;

  if (typeof obj.name !== 'string') {
    Logger.warn('[Gemma4] Tool call missing "name" field:', item);
    return null;
  }

  const args =
    obj.arguments !== null &&
    typeof obj.arguments === 'object' &&
    !Array.isArray(obj.arguments)
      ? (obj.arguments as Record<string, unknown>)
      : {};

  return {
    toolName: obj.name,
    arguments: args,
  };
}

function escapeRegex(str: string): string {
  return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}
