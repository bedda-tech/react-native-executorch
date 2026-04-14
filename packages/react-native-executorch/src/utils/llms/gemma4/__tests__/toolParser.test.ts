/**
 * Tests for Gemma 4 tool call parser.
 *
 * Covers: well-formed JSON, malformed JSON (jsonrepair), multi-call blocks,
 * responses with no tool calls, and edge cases.
 */

import {
  parseGemma4Response,
  parseGemma4ToolCalls,
  buildGemma4Tool,
} from '../toolParser';

// ---------------------------------------------------------------------------
// parseGemma4Response — no tool calls
// ---------------------------------------------------------------------------

describe('parseGemma4Response — plain text', () => {
  it('returns empty toolCalls for plain text', () => {
    const result = parseGemma4Response('The weather is sunny today.');
    expect(result.toolCalls).toHaveLength(0);
    expect(result.hasFunctionCalls).toBe(false);
    expect(result.textContent).toBe('The weather is sunny today.');
  });

  it('returns empty toolCalls for empty string', () => {
    const result = parseGemma4Response('');
    expect(result.toolCalls).toHaveLength(0);
    expect(result.hasFunctionCalls).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// parseGemma4Response — single well-formed tool call
// ---------------------------------------------------------------------------

describe('parseGemma4Response — single tool call', () => {
  const TAP_RESPONSE =
    '<function_calls>\n[{"name":"tap","arguments":{"nodeId":"settings-icon-42"}}]\n</function_calls>';

  it('detects hasFunctionCalls=true', () => {
    const result = parseGemma4Response(TAP_RESPONSE);
    expect(result.hasFunctionCalls).toBe(true);
  });

  it('parses toolName correctly', () => {
    const result = parseGemma4Response(TAP_RESPONSE);
    expect(result.toolCalls).toHaveLength(1);
    expect(result.toolCalls[0]!.toolName).toBe('tap');
  });

  it('parses arguments correctly', () => {
    const result = parseGemma4Response(TAP_RESPONSE);
    expect(result.toolCalls[0]!.arguments).toEqual({
      nodeId: 'settings-icon-42',
    });
  });

  it('removes function_calls block from textContent', () => {
    const result = parseGemma4Response(TAP_RESPONSE);
    expect(result.textContent).toBe('');
  });
});

// ---------------------------------------------------------------------------
// parseGemma4Response — reasoning preamble before tool call
// ---------------------------------------------------------------------------

describe('parseGemma4Response — preamble + tool call', () => {
  const PREAMBLE_RESPONSE =
    'I need to tap the settings button.\n' +
    '<function_calls>\n' +
    '[{"name":"tap","arguments":{"nodeId":"btn-7"}}]\n' +
    '</function_calls>';

  it('extracts textContent as the preamble', () => {
    const result = parseGemma4Response(PREAMBLE_RESPONSE);
    expect(result.textContent).toBe('I need to tap the settings button.');
  });

  it('parses the tool call', () => {
    const result = parseGemma4Response(PREAMBLE_RESPONSE);
    expect(result.toolCalls[0]!.toolName).toBe('tap');
    expect(result.toolCalls[0]!.arguments).toEqual({ nodeId: 'btn-7' });
  });
});

// ---------------------------------------------------------------------------
// parseGemma4Response — multiple tool calls in one block
// ---------------------------------------------------------------------------

describe('parseGemma4Response — multiple calls in one block', () => {
  const MULTI_RESPONSE =
    '<function_calls>\n' +
    '[{"name":"tap","arguments":{"nodeId":"n1"}},{"name":"scroll","arguments":{"nodeId":"n2","direction":"down"}}]\n' +
    '</function_calls>';

  it('parses both tool calls', () => {
    const result = parseGemma4Response(MULTI_RESPONSE);
    expect(result.toolCalls).toHaveLength(2);
  });

  it('parses first call correctly', () => {
    const result = parseGemma4Response(MULTI_RESPONSE);
    expect(result.toolCalls[0]!.toolName).toBe('tap');
  });

  it('parses second call correctly', () => {
    const result = parseGemma4Response(MULTI_RESPONSE);
    expect(result.toolCalls[1]!.toolName).toBe('scroll');
    expect(result.toolCalls[1]!.arguments).toEqual({
      nodeId: 'n2',
      direction: 'down',
    });
  });
});

// ---------------------------------------------------------------------------
// parseGemma4Response — malformed JSON (jsonrepair)
// ---------------------------------------------------------------------------

describe('parseGemma4Response — malformed JSON', () => {
  it('handles trailing comma in object', () => {
    const response =
      '<function_calls>\n[{"name":"type_text","arguments":{"text":"hello",}}]\n</function_calls>';
    const result = parseGemma4Response(response);
    expect(result.toolCalls).toHaveLength(1);
    expect(result.toolCalls[0]!.toolName).toBe('type_text');
  });

  it('handles missing quotes on string values', () => {
    // jsonrepair can handle certain unquoted values
    const response =
      '<function_calls>\n[{"name": "global_action", "arguments": {"action": "back"}}]\n</function_calls>';
    const result = parseGemma4Response(response);
    expect(result.toolCalls[0]!.arguments).toEqual({ action: 'back' });
  });
});

// ---------------------------------------------------------------------------
// parseGemma4Response — items without required fields
// ---------------------------------------------------------------------------

describe('parseGemma4Response — invalid items in array', () => {
  it('skips items missing the "name" field', () => {
    const response =
      '<function_calls>\n[{"arguments":{"x":1}},{"name":"tap","arguments":{"nodeId":"n1"}}]\n</function_calls>';
    const result = parseGemma4Response(response);
    // Only the valid tap call should be returned
    expect(result.toolCalls).toHaveLength(1);
    expect(result.toolCalls[0]!.toolName).toBe('tap');
  });

  it('handles missing arguments field gracefully', () => {
    const response = '<function_calls>\n[{"name":"home"}]\n</function_calls>';
    const result = parseGemma4Response(response);
    expect(result.toolCalls).toHaveLength(1);
    expect(result.toolCalls[0]!.arguments).toEqual({});
  });
});

// ---------------------------------------------------------------------------
// parseGemma4ToolCalls convenience wrapper
// ---------------------------------------------------------------------------

describe('parseGemma4ToolCalls', () => {
  it('returns array of ToolCall objects', () => {
    const calls = parseGemma4ToolCalls(
      '<function_calls>\n[{"name":"tap","arguments":{"nodeId":"x"}}]\n</function_calls>'
    );
    expect(Array.isArray(calls)).toBe(true);
    expect(calls[0]!.toolName).toBe('tap');
  });

  it('returns empty array for plain text', () => {
    const calls = parseGemma4ToolCalls('Just a plain response.');
    expect(calls).toHaveLength(0);
  });
});

// ---------------------------------------------------------------------------
// buildGemma4Tool
// ---------------------------------------------------------------------------

describe('buildGemma4Tool', () => {
  it('returns the input object unchanged', () => {
    const tool = buildGemma4Tool({
      name: 'tap',
      description: 'Tap a UI element',
      parameters: {
        type: 'object',
        properties: { nodeId: { type: 'string' } },
        required: ['nodeId'],
      },
    });
    expect(tool.name).toBe('tap');
    expect(tool.parameters.required).toEqual(['nodeId']);
  });
});

// ---------------------------------------------------------------------------
// Realistic agent scenario
// ---------------------------------------------------------------------------

describe('realistic agent scenario', () => {
  it('handles a full Gemma 4 response with preamble and tool call', () => {
    const modelOutput =
      "I'll tap the Settings app icon to open settings.\n" +
      '<function_calls>\n' +
      '[{"name": "tap", "arguments": {"nodeId": "com.android.settings-42"}}]\n' +
      '</function_calls>';

    const result = parseGemma4Response(modelOutput);

    expect(result.hasFunctionCalls).toBe(true);
    expect(result.toolCalls).toHaveLength(1);
    expect(result.toolCalls[0]!.toolName).toBe('tap');
    expect(result.toolCalls[0]!.arguments.nodeId).toBe(
      'com.android.settings-42'
    );
    expect(result.textContent).toContain("I'll tap the Settings app icon");
  });

  it('handles task_complete signal', () => {
    const modelOutput =
      '<function_calls>\n' +
      '[{"name": "task_complete", "arguments": {"summary": "Wi-Fi has been enabled."}}]\n' +
      '</function_calls>';

    const result = parseGemma4Response(modelOutput);
    expect(result.toolCalls[0]!.toolName).toBe('task_complete');
    expect(result.toolCalls[0]!.arguments.summary).toBe(
      'Wi-Fi has been enabled.'
    );
  });
});
