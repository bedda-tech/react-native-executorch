/**
 * Gemma 4 utilities for react-native-executorch.
 *
 * This module exposes the chat template formatter, tool call parser, and
 * tokenizer config object needed to use Gemma 4 on-device via ExecuTorch.
 *
 * @example Basic usage with useLLM:
 * ```ts
 * import { useLLM, GEMMA4_E4B } from 'react-native-executorch';
 *
 * const { sendMessage, isReady } = useLLM({ model: GEMMA4_E4B });
 * ```
 *
 * @example Manual prompt construction:
 * ```ts
 * import { formatGemma4Prompt } from 'react-native-executorch';
 *
 * const prompt = formatGemma4Prompt([
 *   { role: 'system', content: 'You are a helpful phone agent.' },
 *   { role: 'user',   content: 'Open Settings' },
 * ]);
 * ```
 *
 * @example Tool calling:
 * ```ts
 * import {
 *   formatGemma4Prompt,
 *   parseGemma4Response,
 *   buildGemma4Tool,
 * } from 'react-native-executorch';
 *
 * const tools = [
 *   buildGemma4Tool({
 *     name: 'tap',
 *     description: 'Tap a UI element by node ID',
 *     parameters: {
 *       type: 'object',
 *       properties: { nodeId: { type: 'string' } },
 *       required: ['nodeId'],
 *     },
 *   }),
 * ];
 *
 * const prompt = formatGemma4Prompt(messages, { tools });
 * // ... run inference ...
 * const { toolCalls, textContent } = parseGemma4Response(rawOutput);
 * ```
 *
 * @module gemma4
 */

export {
  // Chat template
  GEMMA4_TOKENS,
  GEMMA4_TOKENIZER_CONFIG,
  formatGemma4Prompt,
  buildToolInstructions,
} from './chatTemplate';

export type { FormatGemma4Options } from './chatTemplate';

export {
  // Tool parser
  parseGemma4Response,
  parseGemma4ToolCalls,
  buildGemma4Tool,
} from './toolParser';

export type {
  Gemma4ToolCall,
  Gemma4ParseResult,
  Gemma4Tool,
} from './toolParser';

export {
  // Phone control tool schemas
  GEMMA4_PHONE_TOOLS,
  TAP_TOOL,
  TYPE_TEXT_TOOL,
  SWIPE_TOOL,
  SCROLL_TOOL,
  OPEN_APP_TOOL,
  READ_SCREEN_TOOL,
  SCREENSHOT_TOOL,
  GLOBAL_ACTION_TOOL,
  WAIT_TOOL,
  TASK_COMPLETE_TOOL,
} from './phoneTools';

export {
  // Screenshot preprocessing for multimodal / vision input
  GEMMA4_VISION_CONFIG,
  prepareScreenshotMessage,
  benchmarkVisionInference,
  buildScreenAnalysisSystemMessage,
  stripDataUriPrefix,
} from './screenshotPreprocessor';

export type {
  WriteBase64Fn,
  VisionBenchmarkResult,
} from './screenshotPreprocessor';
