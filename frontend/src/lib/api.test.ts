import { afterEach, describe, expect, it, vi } from 'vitest';
import { investigate, SseDecoder } from './api';
import { demoScript } from './demo';

afterEach(() => vi.unstubAllGlobals());
function mockStream(text: string) {
  const bytes = new TextEncoder().encode(text);
  const stream = new ReadableStream({
    start(controller) {
      for (let i = 0; i < bytes.length; i += 7) controller.enqueue(bytes.slice(i, i + 7));
      controller.close();
    },
  });
  const request = vi
    .fn()
    .mockResolvedValue(new Response(stream, { headers: { 'Content-Type': 'text/event-stream' } }));
  vi.stubGlobal('fetch', request);
  return request;
}
describe('SSE transport', () => {
  it('decodes frames split across CRLF boundaries and ignores heartbeats', () => {
    const decoder = new SseDecoder();
    expect(decoder.push(': ping\r\n\r')).toEqual([]);
    expect(decoder.push('\nevent: progress\r\ndata: {"phase":"started"}\r\n\r')).toEqual([]);
    expect(decoder.push('\n')).toEqual([{ event: 'progress', data: { phase: 'started' } }]);
  });
  it('handles UTF-8 byte boundaries and preserves the final result', async () => {
    const script = demoScript('answered', 'run');
    const records = script.events
      .map((event) => `event: progress\ndata: ${JSON.stringify(event)}\n\n`)
      .join('');
    const request = mockStream(
      records +
        `event: result\ndata: ${JSON.stringify({ run_id: 'run', response: script.result })}\n\n`,
    );
    const observe = vi.fn();
    const result = await investigate('合成问题', true, new AbortController().signal, observe);
    expect(result).toEqual(script.result);
    expect(observe).toHaveBeenCalledTimes(script.events.length);
    expect(request).toHaveBeenCalledTimes(1);
    expect(request.mock.calls[0][0]).toBe('/api/investigate/stream');
  });
  it('rejects a truncated stream without retrying the paid POST', async () => {
    const request = mockStream('event: run_started\ndata: {"run_id":"run"}\n\n');
    await expect(investigate('q', true, new AbortController().signal, vi.fn())).rejects.toThrow(
      '未收到最终结果',
    );
    expect(request).toHaveBeenCalledTimes(1);
  });
  it('rejects missing events and results from another run', async () => {
    const script = demoScript('answered', 'run');
    mockStream(`event: progress\ndata: ${JSON.stringify({ ...script.events[0], seq: 2 })}\n\n`);
    await expect(investigate('q', true, new AbortController().signal, vi.fn())).rejects.toThrow(
      '不连续',
    );
    mockStream(
      `event: progress\ndata: ${JSON.stringify(script.events[0])}\n\nevent: result\ndata: ${JSON.stringify({ run_id: 'other', response: script.result })}\n\n`,
    );
    await expect(investigate('q', true, new AbortController().signal, vi.fn())).rejects.toThrow(
      '编号不一致',
    );
  });
  it('accepts structured failed legacy results, not just HTTP 200', async () => {
    const result = demoScript('generation_timeout', 'run').result;
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(Response.json(result, { status: 503 })));
    expect(await investigate('q', false, new AbortController().signal, vi.fn())).toEqual(result);
  });
});
