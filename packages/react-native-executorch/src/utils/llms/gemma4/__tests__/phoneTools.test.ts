/**
 * Tests for Gemma 4 phone control tool schemas.
 *
 * Verifies:
 *  - All phone tools are well-formed (name, description, parameters).
 *  - GEMMA4_PHONE_TOOLS exports all expected tools.
 *  - Tool schemas survive a round-trip through the prompt formatter and parser.
 */

import {
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
} from '../phoneTools';
import { formatGemma4Prompt, buildToolInstructions } from '../chatTemplate';
import { parseGemma4Response } from '../toolParser';

// ---------------------------------------------------------------------------
// Schema shape validation
// ---------------------------------------------------------------------------

describe('GEMMA4_PHONE_TOOLS — schema structure', () => {
  it('contains exactly 10 tools', () => {
    expect(GEMMA4_PHONE_TOOLS).toHaveLength(10);
  });

  it('every tool has a non-empty name', () => {
    for (const tool of GEMMA4_PHONE_TOOLS) {
      expect(typeof tool.name).toBe('string');
      expect((tool as any).name.length).toBeGreaterThan(0);
    }
  });

  it('every tool has a non-empty description', () => {
    for (const tool of GEMMA4_PHONE_TOOLS) {
      expect(typeof (tool as any).description).toBe('string');
      expect((tool as any).description.length).toBeGreaterThan(10);
    }
  });

  it('every tool has parameters of type object', () => {
    for (const tool of GEMMA4_PHONE_TOOLS) {
      expect((tool as any).parameters.type).toBe('object');
    }
  });
});

// ---------------------------------------------------------------------------
// Individual tool names
// ---------------------------------------------------------------------------

describe('phone tool names', () => {
  it('tap tool has name "tap"', () => {
    expect((TAP_TOOL as any).name).toBe('tap');
  });

  it('type_text tool has name "type_text"', () => {
    expect((TYPE_TEXT_TOOL as any).name).toBe('type_text');
  });

  it('swipe tool has name "swipe"', () => {
    expect((SWIPE_TOOL as any).name).toBe('swipe');
  });

  it('scroll tool has name "scroll"', () => {
    expect((SCROLL_TOOL as any).name).toBe('scroll');
  });

  it('open_app tool has name "open_app"', () => {
    expect((OPEN_APP_TOOL as any).name).toBe('open_app');
  });

  it('read_screen tool has name "read_screen"', () => {
    expect((READ_SCREEN_TOOL as any).name).toBe('read_screen');
  });

  it('screenshot tool has name "screenshot"', () => {
    expect((SCREENSHOT_TOOL as any).name).toBe('screenshot');
  });

  it('global_action tool has name "global_action"', () => {
    expect((GLOBAL_ACTION_TOOL as any).name).toBe('global_action');
  });

  it('wait tool has name "wait"', () => {
    expect((WAIT_TOOL as any).name).toBe('wait');
  });

  it('task_complete tool has name "task_complete"', () => {
    expect((TASK_COMPLETE_TOOL as any).name).toBe('task_complete');
  });
});

// ---------------------------------------------------------------------------
// Required fields validation
// ---------------------------------------------------------------------------

describe('phone tool required fields', () => {
  it('type_text requires "text"', () => {
    expect((TYPE_TEXT_TOOL as any).parameters.required).toContain('text');
  });

  it('swipe requires startX, startY, endX, endY', () => {
    const req = (SWIPE_TOOL as any).parameters.required as string[];
    expect(req).toContain('startX');
    expect(req).toContain('startY');
    expect(req).toContain('endX');
    expect(req).toContain('endY');
  });

  it('scroll requires nodeId and direction', () => {
    const req = (SCROLL_TOOL as any).parameters.required as string[];
    expect(req).toContain('nodeId');
    expect(req).toContain('direction');
  });

  it('open_app requires packageName', () => {
    expect((OPEN_APP_TOOL as any).parameters.required).toContain('packageName');
  });

  it('global_action requires action', () => {
    expect((GLOBAL_ACTION_TOOL as any).parameters.required).toContain('action');
  });

  it('task_complete requires summary', () => {
    expect((TASK_COMPLETE_TOOL as any).parameters.required).toContain(
      'summary'
    );
  });
});

// ---------------------------------------------------------------------------
// Enum constraints
// ---------------------------------------------------------------------------

describe('phone tool enum constraints', () => {
  it('scroll direction enum covers all four directions', () => {
    const directions = (SCROLL_TOOL as any).parameters.properties.direction
      .enum as string[];
    expect(directions).toEqual(
      expect.arrayContaining(['up', 'down', 'left', 'right'])
    );
  });

  it('global_action enum covers home, back, recents, notifications', () => {
    const actions = (GLOBAL_ACTION_TOOL as any).parameters.properties.action
      .enum as string[];
    expect(actions).toEqual(
      expect.arrayContaining(['home', 'back', 'recents', 'notifications'])
    );
  });
});

// ---------------------------------------------------------------------------
// Integration: tool schemas → prompt → parser round-trip
// ---------------------------------------------------------------------------

describe('phone tools prompt round-trip', () => {
  it('all tool names appear in a prompt built with GEMMA4_PHONE_TOOLS', () => {
    const prompt = formatGemma4Prompt(
      [{ role: 'user', content: 'Open the Settings app' }],
      { tools: GEMMA4_PHONE_TOOLS }
    );

    for (const tool of GEMMA4_PHONE_TOOLS) {
      expect(prompt).toContain(`"${(tool as any).name}"`);
    }
  });

  it('tap tool call parses correctly', () => {
    const rawOutput =
      '<function_calls>\n' +
      '[{"name": "tap", "arguments": {"nodeId": "settings-icon-42"}}]\n' +
      '</function_calls>';

    const result = parseGemma4Response(rawOutput);

    expect(result.toolCalls).toHaveLength(1);
    expect(result.toolCalls[0]!.toolName).toBe('tap');
    expect(result.toolCalls[0]!.arguments.nodeId).toBe('settings-icon-42');
  });

  it('open_app + tap multi-step scenario parses both calls', () => {
    const rawOutput =
      'I will open Settings and then tap Wi-Fi.\n' +
      '<function_calls>\n' +
      '[{"name": "open_app", "arguments": {"packageName": "com.android.settings"}}]\n' +
      '</function_calls>\n' +
      '<function_calls>\n' +
      '[{"name": "tap", "arguments": {"nodeId": "wifi-row-5"}}]\n' +
      '</function_calls>';

    const result = parseGemma4Response(rawOutput);

    expect(result.toolCalls).toHaveLength(2);
    expect(result.toolCalls[0]!.toolName).toBe('open_app');
    expect((result.toolCalls[0]!.arguments as any).packageName).toBe(
      'com.android.settings'
    );
    expect(result.toolCalls[1]!.toolName).toBe('tap');
    expect((result.toolCalls[1]!.arguments as any).nodeId).toBe('wifi-row-5');
  });

  it('task_complete with summary parses correctly', () => {
    const rawOutput =
      '<function_calls>\n' +
      '[{"name": "task_complete", "arguments": {"summary": "Wi-Fi has been enabled successfully."}}]\n' +
      '</function_calls>';

    const result = parseGemma4Response(rawOutput);

    expect(result.toolCalls[0]!.toolName).toBe('task_complete');
    expect((result.toolCalls[0]!.arguments as any).summary).toBe(
      'Wi-Fi has been enabled successfully.'
    );
  });

  it('global_action back parses correctly', () => {
    const rawOutput =
      '<function_calls>\n' +
      '[{"name": "global_action", "arguments": {"action": "back"}}]\n' +
      '</function_calls>';

    const result = parseGemma4Response(rawOutput);

    expect(result.toolCalls[0]!.toolName).toBe('global_action');
    expect((result.toolCalls[0]!.arguments as any).action).toBe('back');
  });

  it('swipe with all required args parses correctly', () => {
    const rawOutput =
      '<function_calls>\n' +
      '[{"name": "swipe", "arguments": {"startX": 540, "startY": 1500, "endX": 540, "endY": 500}}]\n' +
      '</function_calls>';

    const result = parseGemma4Response(rawOutput);
    const args = result.toolCalls[0]!.arguments as any;

    expect(result.toolCalls[0]!.toolName).toBe('swipe');
    expect(args.startX).toBe(540);
    expect(args.startY).toBe(1500);
    expect(args.endX).toBe(540);
    expect(args.endY).toBe(500);
  });

  it('buildToolInstructions includes all phone tool names', () => {
    const instructions = buildToolInstructions(GEMMA4_PHONE_TOOLS);
    const expectedNames = [
      'tap',
      'type_text',
      'swipe',
      'scroll',
      'open_app',
      'read_screen',
      'screenshot',
      'global_action',
      'wait',
      'task_complete',
    ];
    for (const name of expectedNames) {
      expect(instructions).toContain(`"${name}"`);
    }
  });
});
