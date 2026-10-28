import { afterEach, describe, expect, it, vi } from 'vitest';
import { investigate, SseDecoder, RunEventCursor, commandRun, watchRun } from './api';
import { demoScript } from './demo';
import { resumableDemo, advanceDemo } from './demo';

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
  it.each([true, false])('sends and verifies the selected model (stream=%s)', async (stream) => {
    const model = 'synthetic-b';
    const result = { ...demoScript('answered', 'run').result, model };
    const request = stream
      ? mockStream(
          `event: result\ndata: ${JSON.stringify({ run_id: 'run', response: result })}\n\n`,
        )
      : vi.fn().mockResolvedValue(Response.json(result));
    if (!stream) vi.stubGlobal('fetch', request);
    expect(await investigate('q', stream, new AbortController().signal, vi.fn(), model)).toEqual(
      result,
    );
    expect(JSON.parse(request.mock.calls[0][1].body)).toEqual({ query: 'q', model });
  });
  it.each([true, false])(
    'rejects a result from a different selected model (stream=%s)',
    async (stream) => {
      const result = { ...demoScript('answered', 'run').result, model: 'synthetic-a' };
      if (stream)
        mockStream(
          `event: result\ndata: ${JSON.stringify({ run_id: 'run', response: result })}\n\n`,
        );
      else vi.stubGlobal('fetch', vi.fn().mockResolvedValue(Response.json(result)));
      await expect(
        investigate('q', stream, new AbortController().signal, vi.fn(), 'synthetic-b'),
      ).rejects.toThrow('模型与所选模型不一致');
    },
  );
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

describe('persistent investigations', () => {
  it('deduplicates replayed events and refuses gaps', () => {
    const cursor = new RunEventCursor(12);
    expect(cursor.accept(12)).toBe(false);
    expect(cursor.accept(13)).toBe(true);
    expect(cursor.accept(13)).toBe(false);
    expect(() => cursor.accept(15)).toThrow('事件序号缺失');
    expect(cursor.seq).toBe(13);
  });
  it('retries a command with the same idempotency key and waiting generation', async () => {
    const run = resumableDemo('run', '合成问题').durable!;
    const request = vi
      .fn()
      .mockRejectedValueOnce(new TypeError('network'))
      .mockResolvedValueOnce(Response.json(run));
    vi.stubGlobal('fetch', request);
    await commandRun(run, 'supplement', new AbortController().signal, '甲到乙');
    expect(request.mock.calls[0][1].body).toBe(request.mock.calls[1][1].body);
    expect(JSON.parse(request.mock.calls[0][1].body).wait_id).toBe(run.wait_id);
  });
  it('subscribes with a cursor, ignores duplicate events and never creates a run', async () => {
    const initial = resumableDemo('run', '合成问题').durable!;
    const final = advanceDemo(resumableDemo('run', '合成问题'), 'finish').durable!;
    const event = {
      schema_version: 1,
      run_id: 'run',
      seq: initial.cursor + 1,
      kind: 'state',
      payload: {},
    };
    const wire = `event: run_event\ndata: ${JSON.stringify(event)}\n\n`;
    const request = vi
      .fn()
      .mockResolvedValueOnce(Response.json(initial))
      .mockResolvedValueOnce(
        new Response(wire + wire, { headers: { 'Content-Type': 'text/event-stream' } }),
      )
      .mockResolvedValueOnce(Response.json(final));
    vi.stubGlobal('fetch', request);
    const observe = vi.fn();
    await watchRun('run', new AbortController().signal, observe, vi.fn());
    expect(observe.mock.calls.map((call) => call[0].status)).toEqual([
      'waiting_input',
      'completed',
    ]);
    expect(request.mock.calls[1][0]).toContain(`after=${initial.cursor}`);
    expect(request.mock.calls.every((call) => !call[1].method || call[1].method === 'GET')).toBe(
      true,
    );
  });
});

describe('control intent across lease claim', () => {
  it.each(['queued', 'resuming'])('pauses %s after exactly one internal claim', async (status) => {
    const run = { ...resumableDemo('run', '合成问题').durable!, status, wait_id: null };
    const claimed = { ...run, status: 'running', version: run.version + 1 };
    const paused = { ...claimed, status: 'pausing', version: claimed.version + 1 };
    const request = vi
      .fn()
      .mockResolvedValueOnce(Response.json({ code: 'stale_run_version' }, { status: 409 }))
      .mockResolvedValueOnce(Response.json(claimed))
      .mockResolvedValueOnce(Response.json(paused));
    vi.stubGlobal('fetch', request);
    expect(await commandRun(run, 'pause', new AbortController().signal)).toEqual(paused);
    const first = JSON.parse(request.mock.calls[0][1].body);
    const retry = JSON.parse(request.mock.calls[2][1].body);
    expect(retry.version).toBe(claimed.version);
    expect(retry.request_id).not.toBe(first.request_id);
    expect(request).toHaveBeenCalledTimes(3);
  });
  it.each([
    ['supplement', 'running', 1],
    ['resume', 'running', 1],
    ['pause', 'running', 3],
    ['pause', 'waiting_input', 1],
    ['pause', 'completed', 1],
  ])('does not rebase %s onto %s (+%s)', async (action, status, delta) => {
    const run = { ...resumableDemo('run', '合成问题').durable!, status: 'resuming' };
    const request = vi
      .fn()
      .mockResolvedValueOnce(Response.json({ code: 'stale_run_version' }, { status: 409 }))
      .mockResolvedValueOnce(
        Response.json({ ...run, status, version: run.version + Number(delta) }),
      );
    vi.stubGlobal('fetch', request);
    await expect(
      commandRun(run, String(action), new AbortController().signal, '旧补充'),
    ).rejects.toMatchObject({ code: 'stale_run_version' });
    expect(request.mock.calls.filter((call) => call[1].method === 'POST')).toHaveLength(1);
  });
  it('bounds stale retries and preserves the retry key after response loss', async () => {
    const run = { ...resumableDemo('run', '合成问题').durable!, status: 'queued' };
    const claimed = { ...run, status: 'running', version: run.version + 1 };
    const request = vi
      .fn()
      .mockResolvedValueOnce(Response.json({ code: 'stale_run_version' }, { status: 409 }))
      .mockResolvedValueOnce(Response.json(claimed))
      .mockRejectedValueOnce(new TypeError('response lost'))
      .mockResolvedValueOnce(Response.json({ code: 'stale_run_version' }, { status: 409 }));
    vi.stubGlobal('fetch', request);
    await expect(commandRun(run, 'cancel', new AbortController().signal)).rejects.toMatchObject({
      code: 'stale_run_version',
    });
    expect(request.mock.calls[2][1].body).toBe(request.mock.calls[3][1].body);
    expect(request).toHaveBeenCalledTimes(4);
  });
});

it.each(['wait_id', 'supplements'])('does not rebase control across changed %s', async (field) => {
  const run = { ...resumableDemo('run', '合成问题').durable!, status: 'resuming' };
  const current = {
    ...run,
    status: 'running',
    version: run.version + 1,
    ...(field === 'wait_id'
      ? { wait_id: 'later-wait' }
      : {
          supplements: [{ wait_id: 'later-wait', question: '新问题', text: '新补充' }],
        }),
  };
  const request = vi
    .fn()
    .mockResolvedValueOnce(Response.json({ code: 'stale_run_version' }, { status: 409 }))
    .mockResolvedValueOnce(Response.json(current));
  vi.stubGlobal('fetch', request);
  await expect(commandRun(run, 'pause', new AbortController().signal)).rejects.toMatchObject({
    code: 'stale_run_version',
  });
  expect(request).toHaveBeenCalledTimes(2);
});
