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
    static sync: {
      write(png: PNG, options?: Record<string, unknown>): Buffer;
      read(buffer: Buffer, options?: Record<string, unknown>): PNG;
    };
  }
}
