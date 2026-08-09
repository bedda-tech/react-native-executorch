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
      nodeId: {
        type: 'string',
        description:
          'Accessibility node ID of the editable field (optional — auto-detects focused field if omitted)',
      },
    },
    required: ['text'],
  },
});

/**
 * Long press a UI element by its accessibility node ID or screen coordinates.
 * Opens context menus or selects text.
 */
export const LONG_PRESS_TOOL: Gemma4Tool = buildGemma4Tool({
  name: 'long_press',
  description:
    'Long press a UI element by its node ID or screen coordinates (opens context menus, selects text)',
  parameters: {
    type: 'object',
    properties: {
      nodeId: { type: 'string', description: 'Accessibility node ID' },
      x: {
        type: 'number',
        description: 'X coordinate (fallback if no nodeId)',
      },
      y: {
        type: 'number',
        description: 'Y coordinate (fallback if no nodeId)',
      },
    },
  },
});

/**
 * Clear all text from an input field.
 */
export const CLEAR_TEXT_TOOL: Gemma4Tool = buildGemma4Tool({
  name: 'clear_text',
  description:
    'Clear all text from an input field. If nodeId is omitted, the currently focused editable field is used. Prefer this over type_text with empty string when you want to confirm clearing.',
  parameters: {
    type: 'object',
    properties: {
      nodeId: {
        type: 'string',
        description:
          'Node ID of the editable field (optional, auto-detects focused field if omitted)',
      },
    },
  },
});

/**
 * Press the Enter / IME action key on an input field.
 */
export const PRESS_ENTER_TOOL: Gemma4Tool = buildGemma4Tool({
  name: 'press_enter',
  description:
    'Press the Enter / IME action key on an input field to submit a search, send a message, or confirm input. If nodeId is omitted, the currently focused editable field is used.',
  parameters: {
    type: 'object',
    properties: {
      nodeId: {
        type: 'string',
        description:
          'Node ID of the editable field (optional, auto-detects focused field if omitted)',
      },
    },
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
    required: ['direction'],
  },
});

/**
 * Scroll a container repeatedly in a direction until a matching node appears
 * in the accessibility tree, then return its nodeId.
 */
export const SCROLL_UNTIL_FOUND_TOOL: Gemma4Tool = buildGemma4Tool({
  name: 'scroll_until_found',
  description:
    'Scroll a container repeatedly in a direction until a matching node appears in the accessibility tree, then return its nodeId. Returns null if the node is not found after maxScrolls scrolls. Use for long lists where the target item is not currently visible (e.g. finding a contact, app, or setting buried in a scrollable list).',
  parameters: {
    type: 'object',
    properties: {
      direction: {
        type: 'string',
        description: 'Scroll direction',
        enum: ['up', 'down', 'left', 'right'],
      },
      text: {
        type: 'string',
        description: 'Substring to match against node text (case-sensitive)',
      },
      contentDescription: {
        type: 'string',
        description: 'Substring to match against node content description',
      },
      className: {
        type: 'string',
        description: 'Exact class name to match (e.g. android.widget.Button)',
      },
      isChecked: {
        type: 'boolean',
        description: 'Filter by checked state (true=checked, false=unchecked)',
      },
      isEnabled: {
        type: 'boolean',
        description: 'Filter by enabled state (false to find disabled nodes)',
      },
      scrollNodeId: {
        type: 'string',
        description:
          'Node ID of the scrollable container (optional, auto-detects if omitted)',
      },
      maxScrolls: {
        type: 'number',
        description:
          'Maximum number of scroll steps before giving up (default 20)',
      },
      intervalMs: {
        type: 'number',
        description:
          'Delay in ms between scroll and accessibility-tree check (default 300)',
      },
    },
    required: ['direction'],
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
 * Signal that the assigned task cannot be completed.
 * Prefer this over running until the step limit when the task is impossible
 * or blocked.
 */
export const TASK_FAILED_TOOL: Gemma4Tool = buildGemma4Tool({
  name: 'task_failed',
  description:
    'Signal that the task cannot be completed. Use this when the task is impossible, blocked, or requires unavailable permissions. Prefer this over running until the step limit.',
  parameters: {
    type: 'object',
    properties: {
      reason: {
        type: 'string',
        description: 'Explanation of why the task failed or is impossible',
      },
    },
    required: ['reason'],
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
  LONG_PRESS_TOOL,
  TYPE_TEXT_TOOL,
  CLEAR_TEXT_TOOL,
  PRESS_ENTER_TOOL,
  SWIPE_TOOL,
  SCROLL_TOOL,
  SCROLL_UNTIL_FOUND_TOOL,
  OPEN_APP_TOOL,
  READ_SCREEN_TOOL,
  SCREENSHOT_TOOL,
  GLOBAL_ACTION_TOOL,
  WAIT_TOOL,
  TASK_COMPLETE_TOOL,
  TASK_FAILED_TOOL,
];
