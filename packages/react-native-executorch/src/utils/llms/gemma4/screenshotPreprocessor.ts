/**
 * Screenshot preprocessing utilities for Gemma 4 multimodal (vision) input.
 *
 * Gemma 4 E4B accepts images via the `mediaPath` field on user `Message`
 * objects.  The accessibility controller (`react-native-accessibility-controller`)
 * returns screenshots as base64-encoded PNG strings.  This module bridges the
 * two: it writes the base64 data to a local file and builds the correctly-shaped
 * `Message` the LLMController expects.
 *
 * ## Usage
 *
 * ```ts
 * import * as FileSystem from 'expo-file-system';
 * import { prepareScreenshotMessage, benchmarkVisionInference } from 'react-native-executorch';
 *
 * // 1. Get a screenshot from the accessibility controller
 * const base64Png = await AccessibilityController.takeScreenshot();
 *
 * // 2. Prepare a Message with the screenshot attached
 * const msg = await prepareScreenshotMessage(
 *   base64Png,
 *   'What UI elements are visible on this screen?',
 *   async (b64) => {
 *     const path = FileSystem.cacheDirectory + 'screenshot_latest.png';
 *     await FileSystem.writeAsStringAsync(path, b64, { encoding: 'base64' });
 *     return path;
 *   },
 * );
 *
 * // 3. Pass to the LLM
 * const { result, durationMs } = await benchmarkVisionInference(() =>
 *   llm.generate([msg])
 * );
 * console.log(`Inference took ${durationMs} ms`);
 * ```
 *
 * @module gemma4/screenshotPreprocessor
 */

import { Message } from '../../../types/llm';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/**
 * Gemma 4 image preprocessing constants.
 *
 * Gemma 4 (PaliGemma vision encoder) natively processes images at 896 × 896.
 * Images of different sizes are rescaled and padded by the native runtime, but
 * keeping screenshots close to this resolution avoids unnecessary token waste.
 *
 * Android accessibility screenshots are typically 1080 × 2400 or similar tall
 * aspect ratios.  The preprocessor guidance below recommends downscaling the
 * long edge to ≤ 896 px before passing to the model.
 */
export const GEMMA4_VISION_CONFIG = {
  /** Native image resolution used by the PaliGemma encoder embedded in Gemma 4. */
  nativeResolution: 896,
  /**
   * Recommended maximum long-edge pixel count for accessibility screenshots.
   * Screenshots larger than this are rescaled on the native side, but staying
   * close to this value reduces the number of image tokens and speeds up TTFT.
   */
  recommendedMaxEdge: 896,
  /**
   * The image token string that Gemma 4's tokenizer uses to mark image
   * positions in the prompt.  This must match the `image_token` field in
   * the model's `tokenizer_config.json`.
   */
  imageToken: '<image>',
  /**
   * Approximate number of visual tokens consumed per image in Gemma 4 E4B.
   * Useful for context-window budget calculations.
   */
  approximateVisualTokens: 256,
} as const;

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/**
 * Callback that the caller provides to write a base64 string to a local file.
 * The function must return the absolute local path of the written file.
 *
 * Example (expo-file-system):
 * ```ts
 * const writeBase64 = async (b64: string) => {
 *   const path = FileSystem.cacheDirectory + 'screenshot.png';
 *   await FileSystem.writeAsStringAsync(path, b64, { encoding: 'base64' });
 *   return path;
 * };
 * ```
 *
 * Example (react-native-fs):
 * ```ts
 * const writeBase64 = async (b64: string) => {
 *   const path = RNFS.CachesDirectoryPath + '/screenshot.png';
 *   await RNFS.writeFile(path, b64, 'base64');
 *   return path;
 * };
 * ```
 */
export type WriteBase64Fn = (base64: string) => Promise<string>;

/**
 * Result returned by `benchmarkVisionInference`.
 */
export interface VisionBenchmarkResult {
  /** The string returned by the inference function. */
  result: string;
  /** Wall-clock time from call to resolve, in milliseconds. */
  durationMs: number;
  /** Tokens-per-second estimate, if `tokenCount` was provided. */
  tokensPerSecond?: number;
}

// ---------------------------------------------------------------------------
// Main API
// ---------------------------------------------------------------------------

/**
 * Prepares a `Message` that attaches a base64 screenshot to a user text query.
 *
 * The function:
 * 1. Calls `writeBase64` to persist the PNG data to a local file.
 * 2. Returns a `Message` with `role: 'user'`, `content: userText`, and
 *    `mediaPath` set to the local file path.
 *
 * The returned `Message` can be passed directly to `llm.generate()` or
 * `llm.generateWithTools()` when the model was loaded with
 * `capabilities: ['vision']`.
 *
 * @param base64Png   - Raw base64-encoded PNG string (no `data:image/...` prefix).
 * @param userText    - The text question to ask about the screenshot.
 * @param writeBase64 - Caller-supplied function to persist the base64 data.
 * @returns           A `Message` with `mediaPath` pointing to the local PNG file.
 *
 * @example
 * ```ts
 * const msg = await prepareScreenshotMessage(
 *   base64Png,
 *   'Which app is currently open?',
 *   async (b64) => {
 *     const path = FileSystem.cacheDirectory + 'screen.png';
 *     await FileSystem.writeAsStringAsync(path, b64, { encoding: 'base64' });
 *     return path;
 *   },
 * );
 * const response = await llm.generate([systemMsg, msg]);
 * ```
 */
export async function prepareScreenshotMessage(
  base64Png: string,
  userText: string,
  writeBase64: WriteBase64Fn
): Promise<Message> {
  const sanitized = stripDataUriPrefix(base64Png);
  const localPath = await writeBase64(sanitized);

  return {
    role: 'user',
    content: userText,
    mediaPath: localPath,
  };
}

/**
 * Wraps a vision inference call to measure its wall-clock duration.
 *
 * @param fn          - Async function that runs inference and returns a string.
 * @param tokenCount  - Optional number of tokens generated (used to compute tok/s).
 * @returns           Object with `result`, `durationMs`, and optional `tokensPerSecond`.
 *
 * @example
 * ```ts
 * const { result, durationMs } = await benchmarkVisionInference(() =>
 *   llm.generate([screenshotMsg])
 * );
 * console.log(`Vision inference: ${durationMs} ms`);
 * ```
 */
export async function benchmarkVisionInference(
  fn: () => Promise<string>,
  tokenCount?: number
): Promise<VisionBenchmarkResult> {
  const start = Date.now();
  const result = await fn();
  const durationMs = Date.now() - start;

  const tokensPerSecond =
    tokenCount !== undefined && durationMs > 0
      ? Math.round((tokenCount / durationMs) * 1000)
      : undefined;

  return { result, durationMs, tokensPerSecond };
}

/**
 * Builds a system prompt that instructs Gemma 4 to interpret accessibility
 * screenshots and suggest UI actions.
 *
 * Pass this as the first `Message` (role: `'system'`) in calls to
 * `llm.generate()` when using Gemma 4 for screen understanding.
 *
 * @param task - The high-level task the agent is trying to accomplish.
 * @returns    A system `Message` with UI understanding instructions.
 */
export function buildScreenAnalysisSystemMessage(task: string): Message {
  return {
    role: 'system',
    content:
      `You are a mobile UI understanding assistant. ` +
      `You are shown a screenshot of an Android phone screen. ` +
      `Your job is to describe what you see and identify the relevant UI elements ` +
      `needed to accomplish the following task: "${task}". ` +
      `Be concise. Focus on actionable observations: visible buttons, text fields, ` +
      `menus, or other interactive elements that are relevant to the task.`,
  };
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Strips any `data:image/...;base64,` prefix from a base64 string.
 * Some sources (e.g. browser APIs) prepend this; ExecuTorch expects raw base64.
 *
 * @param base64 - Raw or prefixed base64 string.
 * @returns      Pure base64 data.
 */
export function stripDataUriPrefix(base64: string): string {
  const comma = base64.indexOf(',');
  if (comma !== -1 && base64.substring(0, comma).includes('base64')) {
    return base64.substring(comma + 1);
  }
  return base64;
}
