/**
 * Phone control tool schemas for Gemma 4 function calling.
 *
 * These are the canonical tool definitions used by the Deft agent loop
 * (react-native-device-agent) when running on-device inference with Gemma 4
 * via react-native-executorch.
 *
 * Each tool is defined in the JSON Schema format that Gemma 4 was trained to
 * understand, and can be passed directly to `formatGemma4Prompt` or the
 * `useLLM` hook's tool-calling API.
 *
 * @module gemma4/phoneTools
 */

import { buildGemma4Tool, Gemma4Tool } from './toolParser';

/**
 * Tap a UI element by its accessibility node ID or screen coordinates.
 * Prefer `nodeId` when the tree is available; fall back to `x`/`y` for
 * elements not exposed by the accessibility tree.
 */
export const TAP_TOOL: Gemma4Tool = buildGemma4Tool({
  name: 'tap',
  description:
    'Tap a UI element. Prefer nodeId from the accessibility tree. Fall back to x/y screen coordinates when the element has no node ID.',
  parameters: {
    type: 'object',
    properties: {
      nodeId: {
        type: 'string',
        description: 'Accessibility node ID (preferred)',
      },
      x: {
        type: 'number',
        description: 'X screen coordinate in pixels (fallback)',
      },
      y: {
        type: 'number',
        description: 'Y screen coordinate in pixels (fallback)',
      },
    },
  },
});

/**
 * Type text into the currently focused input field.
 */
export const TYPE_TEXT_TOOL: Gemma4Tool = buildGemma4Tool({
  name: 'type_text',
  description: 'Type text into the currently focused input field.',
  parameters: {
    type: 'object',
    properties: {
      text: {
        type: 'string',
        description: 'The text to type',
      },
    },
    required: ['text'],
  },
});

/**
 * Swipe between two screen coordinates.
 */
export const SWIPE_TOOL: Gemma4Tool = buildGemma4Tool({
  name: 'swipe',
  description: 'Swipe from one screen position to another.',
  parameters: {
    type: 'object',
    properties: {
      startX: { type: 'number', description: 'Start X coordinate in pixels' },
      startY: { type: 'number', description: 'Start Y coordinate in pixels' },
      endX: { type: 'number', description: 'End X coordinate in pixels' },
      endY: { type: 'number', description: 'End Y coordinate in pixels' },
      durationMs: {
        type: 'number',
        description: 'Swipe duration in milliseconds (default 300)',
      },
    },
    required: ['startX', 'startY', 'endX', 'endY'],
  },
});

/**
 * Scroll a scrollable UI element up, down, left, or right.
 */
export const SCROLL_TOOL: Gemma4Tool = buildGemma4Tool({
  name: 'scroll',
  description: 'Scroll a scrollable UI element in a given direction.',
  parameters: {
    type: 'object',
    properties: {
      nodeId: {
        type: 'string',
        description: 'Accessibility node ID of the scrollable container',
      },
      direction: {
        type: 'string',
        enum: ['up', 'down', 'left', 'right'],
        description: 'Direction to scroll',
      },
    },
    required: ['nodeId', 'direction'],
  },
});

/**
 * Launch an app by its Android package name.
 */
export const OPEN_APP_TOOL: Gemma4Tool = buildGemma4Tool({
  name: 'open_app',
  description: 'Open an installed Android app by its package name.',
  parameters: {
    type: 'object',
    properties: {
      packageName: {
        type: 'string',
        description:
          'Android package name, e.g. "com.android.settings" or "com.google.android.gm"',
      },
    },
    required: ['packageName'],
  },
});

/**
 * Read the current screen state (accessibility tree + serialized text).
 * Use this to observe what is currently on screen before deciding the next action.
 */
export const READ_SCREEN_TOOL: Gemma4Tool = buildGemma4Tool({
  name: 'read_screen',
  description:
    'Capture the current screen state as a structured text representation of the UI tree. Use this to understand what is visible on screen.',
  parameters: {
    type: 'object',
    properties: {},
  },
});

/**
 * Take a screenshot for visual analysis.
 * Returns a base64-encoded PNG image.
 */
export const SCREENSHOT_TOOL: Gemma4Tool = buildGemma4Tool({
  name: 'screenshot',
  description:
    'Take a screenshot of the current screen for visual analysis. Returns a base64-encoded PNG.',
  parameters: {
    type: 'object',
    properties: {},
  },
});

/**
 * Execute a system-level action (home, back, recents, notifications, etc.).
 */
export const GLOBAL_ACTION_TOOL: Gemma4Tool = buildGemma4Tool({
  name: 'global_action',
  description:
    'Execute a system-level action such as pressing Home, Back, or opening the notification shade.',
  parameters: {
    type: 'object',
    properties: {
      action: {
        type: 'string',
        enum: [
          'home',
          'back',
          'recents',
          'notifications',
          'quickSettings',
          'powerDialog',
        ],
        description: 'The system action to perform',
      },
    },
    required: ['action'],
  },
});

/**
 * Wait for the screen to settle before taking the next action.
 */
export const WAIT_TOOL: Gemma4Tool = buildGemma4Tool({
  name: 'wait',
  description:
    'Wait for a specified number of milliseconds before proceeding. Use this to allow animations or loading states to complete.',
  parameters: {
    type: 'object',
    properties: {
      ms: {
        type: 'number',
        description: 'Milliseconds to wait (default 500, max 5000)',
      },
    },
  },
});

/**
 * Signal that the assigned task has been completed successfully.
 * Always call this as the final action when the goal is achieved.
 */
export const TASK_COMPLETE_TOOL: Gemma4Tool = buildGemma4Tool({
  name: 'task_complete',
  description:
    'Signal that the task has been completed successfully. Include a brief summary of what was accomplished.',
  parameters: {
    type: 'object',
    properties: {
      summary: {
        type: 'string',
        description: 'Brief description of what was done',
      },
    },
    required: ['summary'],
  },
});

/**
 * The full set of phone control tools available to the Deft agent.
 * Pass this array to `formatGemma4Prompt` or the `useLLM` hook to enable
 * function calling for phone control tasks.
 *
 * @example
 * ```ts
 * import { formatGemma4Prompt, GEMMA4_PHONE_TOOLS } from 'react-native-executorch';
 *
 * const prompt = formatGemma4Prompt(messages, { tools: GEMMA4_PHONE_TOOLS });
 * ```
 */
export const GEMMA4_PHONE_TOOLS: Gemma4Tool[] = [
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
];
