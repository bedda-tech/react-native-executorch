/**
 * Tests for Gemma 4 chat template formatting.
 *
 * Verifies that formatGemma4Prompt produces the expected token structure
 * for single-turn, multi-turn, system-prompt, and tool-calling scenarios.
 */

import {
  formatGemma4Prompt,
  GEMMA4_TOKENS,
  GEMMA4_TOKENIZER_CONFIG,
  buildToolInstructions,
} from '../chatTemplate';
import type { Gemma4Tool } from '../toolParser';

const { BOS, START_OF_TURN, END_OF_TURN } = GEMMA4_TOKENS;

// ---------------------------------------------------------------------------
// Basic formatting
// ---------------------------------------------------------------------------

describe('formatGemma4Prompt — basic', () => {
  it('starts with <bos> token', () => {
    const prompt = formatGemma4Prompt([{ role: 'user', content: 'Hello' }]);
    expect(prompt.startsWith(BOS)).toBe(true);
  });

  it('formats a single user turn', () => {
    const prompt = formatGemma4Prompt([{ role: 'user', content: 'Hello' }]);
    expect(prompt).toContain(`${START_OF_TURN}user\nHello${END_OF_TURN}`);
  });

  it('appends generation prompt by default', () => {
    const prompt = formatGemma4Prompt([{ role: 'user', content: 'Hello' }]);
    expect(prompt.endsWith(`${START_OF_TURN}model\n`)).toBe(true);
  });

  it('omits generation prompt when addGenerationPrompt=false', () => {
    const prompt = formatGemma4Prompt([{ role: 'user', content: 'Hello' }], {
      addGenerationPrompt: false,
    });
    expect(prompt.endsWith(`${END_OF_TURN}\n`)).toBe(true);
    expect(prompt).not.toContain(`${START_OF_TURN}model\n`);
  });
});

// ---------------------------------------------------------------------------
// System prompt
// ---------------------------------------------------------------------------

describe('formatGemma4Prompt — system prompt', () => {
  it('strips system message and injects into first user turn', () => {
    const prompt = formatGemma4Prompt([
      { role: 'system', content: 'Be brief.' },
      { role: 'user', content: 'Hi' },
    ]);
    // System text should appear before the user content in the first user turn
    const userTurnStart = prompt.indexOf(`${START_OF_TURN}user`);
    const systemIndex = prompt.indexOf('Be brief.');
    const userMsgIndex = prompt.indexOf('Hi');
    expect(systemIndex).toBeGreaterThan(userTurnStart);
    expect(systemIndex).toBeLessThan(userMsgIndex);
  });

  it('does not add a separate system turn', () => {
    const prompt = formatGemma4Prompt([
      { role: 'system', content: 'Be brief.' },
      { role: 'user', content: 'Hi' },
    ]);
    expect(prompt).not.toContain(`${START_OF_TURN}system`);
  });
});

// ---------------------------------------------------------------------------
// Multi-turn
// ---------------------------------------------------------------------------

describe('formatGemma4Prompt — multi-turn', () => {
  it('formats user/assistant alternation', () => {
    const prompt = formatGemma4Prompt([
      { role: 'user', content: 'What time is it?' },
      { role: 'assistant', content: "I don't have a clock." },
      { role: 'user', content: 'That is fine.' },
    ]);

    expect(prompt).toContain(
      `${START_OF_TURN}user\nWhat time is it?${END_OF_TURN}`
    );
    expect(prompt).toContain(
      `${START_OF_TURN}assistant\nI don't have a clock.${END_OF_TURN}`
    );
    expect(prompt).toContain(
      `${START_OF_TURN}user\nThat is fine.${END_OF_TURN}`
    );
    // Generation prompt at end
    expect(prompt.endsWith(`${START_OF_TURN}model\n`)).toBe(true);
  });

  it('appends generation prompt after last user turn', () => {
    const prompt = formatGemma4Prompt([
      { role: 'user', content: 'A' },
      { role: 'assistant', content: 'B' },
      { role: 'user', content: 'C' },
    ]);
    // The last thing before the generation prompt should be the closing turn tag
    expect(prompt).toContain(`C${END_OF_TURN}\n${START_OF_TURN}model\n`);
  });
});

// ---------------------------------------------------------------------------
// Tool calling
// ---------------------------------------------------------------------------

const TAP_TOOL: Gemma4Tool = {
  name: 'tap',
  description: 'Tap a UI element by node ID',
  parameters: {
    type: 'object',
    properties: {
      nodeId: { type: 'string', description: 'Accessibility node ID' },
    },
    required: ['nodeId'],
  },
};

const SCROLL_TOOL: Gemma4Tool = {
  name: 'scroll',
  description: 'Scroll a scrollable element',
  parameters: {
    type: 'object',
    properties: {
      nodeId: { type: 'string' },
      direction: { type: 'string', enum: ['up', 'down', 'left', 'right'] },
    },
    required: ['nodeId', 'direction'],
  },
};

describe('formatGemma4Prompt — tool calling', () => {
  it('includes tool names in the prompt when tools are provided', () => {
    const prompt = formatGemma4Prompt(
      [{ role: 'user', content: 'Open settings' }],
      { tools: [TAP_TOOL, SCROLL_TOOL] }
    );
    expect(prompt).toContain('"tap"');
    expect(prompt).toContain('"scroll"');
  });

  it('includes function_calls format instructions', () => {
    const prompt = formatGemma4Prompt(
      [{ role: 'user', content: 'Tap the icon' }],
      { tools: [TAP_TOOL] }
    );
    expect(prompt).toContain(GEMMA4_TOKENS.FUNCTION_CALLS_OPEN);
    expect(prompt).toContain('tool_name');
  });

  it('still starts with <bos> when tools are provided', () => {
    const prompt = formatGemma4Prompt([{ role: 'user', content: 'Tap' }], {
      tools: [TAP_TOOL],
    });
    expect(prompt.startsWith(BOS)).toBe(true);
  });

  it('embeds user content in tool-mode prompt', () => {
    const prompt = formatGemma4Prompt(
      [{ role: 'user', content: 'Open Settings' }],
      { tools: [TAP_TOOL] }
    );
    expect(prompt).toContain('Open Settings');
  });

  it('includes system message before tool instructions when both present', () => {
    const prompt = formatGemma4Prompt(
      [
        { role: 'system', content: 'You control an Android phone.' },
        { role: 'user', content: 'Tap the back button' },
      ],
      { tools: [TAP_TOOL] }
    );
    const sysIdx = prompt.indexOf('You control an Android phone.');
    const toolIdx = prompt.indexOf('You have access to the following tools');
    expect(sysIdx).toBeGreaterThan(-1);
    expect(toolIdx).toBeGreaterThan(sysIdx);
  });
});

// ---------------------------------------------------------------------------
// buildToolInstructions
// ---------------------------------------------------------------------------

describe('buildToolInstructions', () => {
  it('returns JSON-serialised tool array', () => {
    const instructions = buildToolInstructions([TAP_TOOL]);
    expect(instructions).toContain('"tap"');
    expect(instructions).toContain('"description"');
  });

  it('includes the function_calls usage example', () => {
    const instructions = buildToolInstructions([TAP_TOOL]);
    expect(instructions).toContain(GEMMA4_TOKENS.FUNCTION_CALLS_OPEN);
    expect(instructions).toContain(GEMMA4_TOKENS.FUNCTION_CALLS_CLOSE);
  });
});

// ---------------------------------------------------------------------------
// GEMMA4_TOKENIZER_CONFIG
// ---------------------------------------------------------------------------

describe('GEMMA4_TOKENIZER_CONFIG', () => {
  it('exports expected special tokens', () => {
    expect(GEMMA4_TOKENIZER_CONFIG.bos_token).toBe('<bos>');
    expect(GEMMA4_TOKENIZER_CONFIG.eos_token).toBe('<eos>');
    expect(GEMMA4_TOKENIZER_CONFIG.pad_token).toBe('<pad>');
  });

  it('has a chat_template string', () => {
    expect(typeof GEMMA4_TOKENIZER_CONFIG.chat_template).toBe('string');
    expect(GEMMA4_TOKENIZER_CONFIG.chat_template.length).toBeGreaterThan(50);
  });
});
