declare module 'pngjs/browser' {
  export class PNG {
    width: number;
    height: number;
    data: Buffer;
    constructor(options?: Record<string, unknown>);
    parse(
      buffer: Buffer | ArrayBufferView,
      callback?: (error: Error | null, data: PNG) => void
    ): PNG;
    pack(): NodeJS.ReadableStream;
  }
}
