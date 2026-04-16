/**
 * Tests for Gemma 4 screenshot preprocessing utilities.
 *
 * Verifies:
 *  - `stripDataUriPrefix` correctly strips and passes through base64 strings.
 *  - `prepareScreenshotMessage` returns a well-formed Message with mediaPath.
 *  - `benchmarkVisionInference` measures duration and computes tok/s.
 *  - `buildScreenAnalysisSystemMessage` returns a system message with the task.
 *  - `GEMMA4_VISION_CONFIG` has expected shape.
 */

import {
  stripDataUriPrefix,
  prepareScreenshotMessage,
  benchmarkVisionInference,
  buildScreenAnalysisSystemMessage,
  GEMMA4_VISION_CONFIG,
} from '../screenshotPreprocessor';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const FAKE_BASE64 =
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==';
const FAKE_PATH = '/data/user/0/com.deft/cache/screenshot_latest.png';

function makeWriter(
  returnPath = FAKE_PATH
): jest.Mock<Promise<string>, [string]> {
  return jest.fn().mockResolvedValue(returnPath);
}

// ---------------------------------------------------------------------------
// stripDataUriPrefix
// ---------------------------------------------------------------------------

describe('stripDataUriPrefix', () => {
  it('passes through a bare base64 string unchanged', () => {
    expect(stripDataUriPrefix(FAKE_BASE64)).toBe(FAKE_BASE64);
  });

  it('strips a data:image/png;base64, prefix', () => {
    const withPrefix = `data:image/png;base64,${FAKE_BASE64}`;
    expect(stripDataUriPrefix(withPrefix)).toBe(FAKE_BASE64);
  });

  it('strips a data:image/jpeg;base64, prefix', () => {
    const withPrefix = `data:image/jpeg;base64,${FAKE_BASE64}`;
    expect(stripDataUriPrefix(withPrefix)).toBe(FAKE_BASE64);
  });

  it('does not strip a string that contains a comma but no base64 marker', () => {
    const noBase64Marker = 'hello,world';
    expect(stripDataUriPrefix(noBase64Marker)).toBe('hello,world');
  });

  it('handles empty string without throwing', () => {
    expect(stripDataUriPrefix('')).toBe('');
  });
});

// ---------------------------------------------------------------------------
// prepareScreenshotMessage
// ---------------------------------------------------------------------------

describe('prepareScreenshotMessage', () => {
  it('returns a user message with the provided text and mediaPath', async () => {
    const writer = makeWriter();
    const msg = await prepareScreenshotMessage(
      FAKE_BASE64,
      'What is on screen?',
      writer
    );

    expect(msg.role).toBe('user');
    expect(msg.content).toBe('What is on screen?');
    expect(msg.mediaPath).toBe(FAKE_PATH);
  });

  it('calls the writer with the sanitized base64 (no data URI prefix)', async () => {
    const writer = makeWriter();
    const withPrefix = `data:image/png;base64,${FAKE_BASE64}`;
    await prepareScreenshotMessage(withPrefix, 'Describe this screen', writer);

    expect(writer).toHaveBeenCalledWith(FAKE_BASE64);
  });

  it('calls the writer exactly once', async () => {
    const writer = makeWriter();
    await prepareScreenshotMessage(FAKE_BASE64, 'tap', writer);

    expect(writer).toHaveBeenCalledTimes(1);
  });

  it('passes through a custom localPath returned by the writer', async () => {
    const customPath = '/sdcard/Download/screen_123.png';
    const writer = makeWriter(customPath);
    const msg = await prepareScreenshotMessage(FAKE_BASE64, 'test', writer);

    expect(msg.mediaPath).toBe(customPath);
  });
});

// ---------------------------------------------------------------------------
// benchmarkVisionInference
// ---------------------------------------------------------------------------

describe('benchmarkVisionInference', () => {
  it('returns the result from the wrapped function', async () => {
    const { result } = await benchmarkVisionInference(() =>
      Promise.resolve('Tap the Settings button.')
    );
    expect(result).toBe('Tap the Settings button.');
  });

  it('reports a non-negative durationMs', async () => {
    const { durationMs } = await benchmarkVisionInference(() =>
      Promise.resolve('done')
    );
    expect(durationMs).toBeGreaterThanOrEqual(0);
  });

  it('computes tokensPerSecond when tokenCount is provided', async () => {
    // Use a fake slow function so we always get > 0 ms
    const { tokensPerSecond } = await benchmarkVisionInference(
      () => new Promise<string>((r) => setTimeout(() => r('ok'), 10)),
      50
    );
    expect(typeof tokensPerSecond).toBe('number');
    expect(tokensPerSecond!).toBeGreaterThan(0);
  });

  it('omits tokensPerSecond when tokenCount is not provided', async () => {
    const { tokensPerSecond } = await benchmarkVisionInference(() =>
      Promise.resolve('ok')
    );
    expect(tokensPerSecond).toBeUndefined();
  });

  it('re-throws errors from the wrapped function', async () => {
    await expect(
      benchmarkVisionInference(() =>
        Promise.reject(new Error('inference failed'))
      )
    ).rejects.toThrow('inference failed');
  });
});

// ---------------------------------------------------------------------------
// buildScreenAnalysisSystemMessage
// ---------------------------------------------------------------------------

describe('buildScreenAnalysisSystemMessage', () => {
  it('returns a system role message', () => {
    const msg = buildScreenAnalysisSystemMessage('Open Settings');
    expect(msg.role).toBe('system');
  });

  it('includes the task string in the content', () => {
    const msg = buildScreenAnalysisSystemMessage('Turn on Wi-Fi');
    expect(msg.content).toContain('Turn on Wi-Fi');
  });

  it('instructs the model to describe UI elements', () => {
    const msg = buildScreenAnalysisSystemMessage('anything');
    expect(msg.content.toLowerCase()).toContain('ui');
  });

  it('has no mediaPath', () => {
    const msg = buildScreenAnalysisSystemMessage('task');
    expect(msg.mediaPath).toBeUndefined();
  });
});

// ---------------------------------------------------------------------------
// GEMMA4_VISION_CONFIG
// ---------------------------------------------------------------------------

describe('GEMMA4_VISION_CONFIG', () => {
  it('has a nativeResolution of 896', () => {
    expect(GEMMA4_VISION_CONFIG.nativeResolution).toBe(896);
  });

  it('has recommendedMaxEdge as a positive number', () => {
    expect(GEMMA4_VISION_CONFIG.recommendedMaxEdge).toBeGreaterThan(0);
  });

  it('has a non-empty imageToken', () => {
    expect(typeof GEMMA4_VISION_CONFIG.imageToken).toBe('string');
    expect(GEMMA4_VISION_CONFIG.imageToken.length).toBeGreaterThan(0);
  });

  it('has approximateVisualTokens as a positive number', () => {
    expect(GEMMA4_VISION_CONFIG.approximateVisualTokens).toBeGreaterThan(0);
  });
});
