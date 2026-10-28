import { z } from 'zod';
import {
  capabilitiesSchema,
  progressSchema,
  resultSchema,
  durableRunSchema,
  type DurableRun,
  type Capabilities,
  type Progress,
  type Result,
} from './model';

export class ApiError extends Error {
  constructor(
    public code: string,
    message: string,
  ) {
    super(message);
  }
}

async function runRequest(path: string, signal: AbortSignal, body?: unknown): Promise<unknown> {
  const response = await fetch(`/api/investigations${path}`, {
    method: body ? 'POST' : 'GET',
    signal,
    cache: 'no-store',
    headers: { 'Content-Type': 'application/json' },
    ...(body ? { body: JSON.stringify(body) } : {}),
  });
  const value = await response.json();
  if (!response.ok)
    throw new ApiError(value.code ?? 'connection_lost', '调查操作未完成，请查看当前状态后重试。');
  return value;
}
export async function createRun(
  query: string,
  model: string | undefined,
  requestId: string,
  signal: AbortSignal,
) {
  return durableRunSchema.parse(
    await runRequest('', signal, { query, model, request_id: requestId }),
  );
}
export async function getRun(id: string, signal: AbortSignal) {
  const run = durableRunSchema.parse(await runRequest(`/${id}`, signal));
  if (run.run_id !== id) throw new ApiError('connection_lost', '调查编号不一致');
  return run;
}
const runListSchema = z.object({
  runs: z.array(
    z.object({
      id: z.string(),
      query: z.string(),
      model: z.string(),
      status: z.string(),
      created_at: z.number(),
      version: z.number(),
      reason: z.string(),
    }),
  ),
});
export async function listRuns(signal: AbortSignal) {
  return runListSchema.parse(await runRequest('', signal)).runs;
}
export async function commandRun(
  run: DurableRun,
  action: string,
  signal: AbortSignal,
  text?: string,
) {
  async function send(expected: DurableRun): Promise<DurableRun> {
    const body = {
      request_id: crypto.randomUUID(),
      version: expected.version,
      ...(action === 'supplement' ? { wait_id: expected.wait_id, text } : {}),
    };
    const request = async () => {
      const next = durableRunSchema.parse(
        await runRequest(`/${run.run_id}/${action}`, signal, body),
      );
      if (next.run_id !== run.run_id) throw new ApiError('connection_lost', '调查编号不一致');
      return next;
    };
    // Lost command responses may retry only the identical idempotency envelope.
    try {
      return await request();
    } catch (error) {
      if (error instanceof ApiError || signal.aborted) throw error;
      return request();
    }
  }
  try {
    return await send(run);
  } catch (error) {
    if (
      !(error instanceof ApiError) ||
      error.code !== 'stale_run_version' ||
      !['pause', 'cancel'].includes(action)
    )
      throw error;
    const current = await getRun(run.run_id, signal);
    // Only rebase across the single internal lease-claim transition. A newer
    // user command, another waiting round, or terminal result needs fresh intent.
    if (
      !['queued', 'resuming'].includes(run.status) ||
      current.status !== 'running' ||
      current.version !== run.version + 1 ||
      current.wait_id !== run.wait_id ||
      JSON.stringify(current.supplements) !== JSON.stringify(run.supplements)
    )
      throw error;
    return send(current); // At most one rebase; the server still checks version.
  }
}

export class RunEventCursor {
  constructor(public seq = 0) {}
  accept(seq: number): boolean {
    if (seq <= this.seq) return false;
    if (seq !== this.seq + 1) throw new ApiError('connection_lost', '事件序号缺失，正在重新同步');
    this.seq = seq;
    return true;
  }
}

const runEventSchema = z.object({
  schema_version: z.literal(1),
  run_id: z.string(),
  seq: z.number().int().positive(),
  kind: z.string(),
  payload: z.unknown(),
});
const endedRuns = new Set([
  'completed',
  'failed',
  'cancelled',
  'budget_exhausted',
  'unrecoverable',
]);

export async function watchRun(
  id: string,
  signal: AbortSignal,
  receive: (run: DurableRun) => void,
  connection: (reconnecting: boolean) => void,
) {
  while (!signal.aborted) {
    const transport = new AbortController();
    const disconnect = () => transport.abort();
    let stall = globalThis.setTimeout(disconnect, 25_000);
    const alive = () => {
      clearTimeout(stall);
      stall = globalThis.setTimeout(disconnect, 25_000);
    };
    signal.addEventListener('abort', disconnect, { once: true });
    if (typeof window !== 'undefined') window.addEventListener('offline', disconnect);
    try {
      const initial = await getRun(id, transport.signal);
      alive();
      receive(initial);
      connection(false);
      if (endedRuns.has(initial.status)) return;
      const cursor = new RunEventCursor(initial.cursor);
      const response = await fetch(`/api/investigations/${id}/stream?after=${cursor.seq}`, {
        signal: transport.signal,
        cache: 'no-store',
      });
      if (
        !response.ok ||
        !response.body ||
        !response.headers.get('content-type')?.includes('text/event-stream')
      )
        throw new ApiError('connection_lost', '进度连接中断');
      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8', { fatal: true });
      const parser = new SseDecoder();
      try {
        while (!signal.aborted) {
          const { value, done } = await reader.read();
          alive();
          const frames = parser.push(
            done ? decoder.decode() : decoder.decode(value, { stream: true }),
          );
          let changed = false;
          for (const frame of frames) {
            if (frame.event !== 'run_event') continue;
            const event = runEventSchema.parse(frame.data);
            if (event.run_id !== id) throw new ApiError('connection_lost', '调查编号不一致');
            changed = cursor.accept(event.seq) || changed;
          }
          if (changed || done) {
            const current = await getRun(id, transport.signal);
            receive(current);
            cursor.seq = Math.max(cursor.seq, current.cursor);
            if (endedRuns.has(current.status)) return;
          }
          if (done) throw new ApiError('connection_lost', '正在重新连接进度');
        }
      } finally {
        await reader.cancel().catch(() => {});
        reader.releaseLock();
      }
    } catch {
      if (signal.aborted) return;
      connection(true);
      await new Promise<void>((resolve) => {
        const finish = () => {
          clearTimeout(timer);
          signal.removeEventListener('abort', finish);
          resolve();
        };
        const timer = window.setTimeout(finish, 1500);
        signal.addEventListener('abort', finish, { once: true });
      });
    } finally {
      clearTimeout(stall);
      signal.removeEventListener('abort', disconnect);
      if (typeof window !== 'undefined') window.removeEventListener('offline', disconnect);
    }
  }
}
export async function capabilities(signal: AbortSignal): Promise<Capabilities> {
  const response = await fetch('/api/capabilities', { signal, cache: 'no-store' });
  if (!response.ok) throw new ApiError('connection_lost', '无法连接本地服务');
  return capabilitiesSchema.parse(await response.json());
}

// The browser may split a frame, a CRLF, or a UTF-8 codepoint across reads.
export class SseDecoder {
  private buffer = '';
  push(text: string): { event: string; data: unknown }[] {
    this.buffer += text;
    if (this.buffer.length > 2_000_000) throw new ApiError('connection_lost', '事件数据超过上限');
    const frames: { event: string; data: unknown }[] = [];
    let match: RegExpExecArray | null;
    while ((match = /\r?\n\r?\n/.exec(this.buffer))) {
      const block = this.buffer.slice(0, match.index);
      this.buffer = this.buffer.slice(match.index + match[0].length);
      let event = 'message';
      const lines: string[] = [];
      for (const line of block.split(/\r?\n/)) {
        if (line.startsWith('event:')) event = line.slice(6).trim();
        if (line.startsWith('data:')) lines.push(line.slice(5).replace(/^ /, ''));
      }
      if (lines.length) frames.push({ event, data: JSON.parse(lines.join('\n')) });
    }
    return frames;
  }
}

export async function investigate(
  query: string,
  streaming: boolean,
  signal: AbortSignal,
  onProgress: (event: Progress) => void,
  model?: string,
): Promise<Result> {
  const checkedResult = (result: Result) => {
    if (model && result.model !== model)
      throw new ApiError('connection_lost', '服务返回的模型与所选模型不一致');
    return result;
  };
  const response = await fetch(`/api/investigate${streaming ? '/stream' : ''}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: streaming ? 'text/event-stream' : 'application/json',
    },
    body: JSON.stringify({ query, ...(model ? { model } : {}) }),
    signal,
    cache: 'no-store',
  });
  if (!streaming || !response.ok) {
    const body = await response.json();
    const result = resultSchema.safeParse(body);
    if (result.success) return checkedResult(result.data);
    throw new ApiError(
      typeof body.code === 'string' ? body.code : 'connection_lost',
      '服务未能完成请求',
    );
  }
  if (!response.headers.get('content-type')?.includes('text/event-stream') || !response.body)
    throw new ApiError('connection_lost', '服务没有返回有效事件流');
  const reader = response.body.getReader();
  const decoder = new TextDecoder('utf-8', { fatal: true });
  const parser = new SseDecoder();
  let runId: string | undefined;
  let lastSeq = 0;
  try {
    while (true) {
      const { value, done } = await reader.read();
      const frames = parser.push(done ? decoder.decode() : decoder.decode(value, { stream: true }));
      for (const frame of frames) {
        if (frame.event === 'run_started') {
          const start = z.object({ run_id: z.string().min(1) }).parse(frame.data);
          if (runId) throw new ApiError('connection_lost', '运行重复开始');
          runId = start.run_id;
        } else if (frame.event === 'progress') {
          const event = progressSchema.parse(frame.data);
          if (runId && runId !== event.run_id)
            throw new ApiError('connection_lost', '运行编号不一致');
          if (event.seq !== lastSeq + 1) throw new ApiError('connection_lost', '执行事件不连续');
          runId = event.run_id;
          lastSeq = event.seq;
          onProgress(event);
        } else if (frame.event === 'result') {
          const envelope = z
            .object({ run_id: z.string().min(1), response: resultSchema })
            .parse(frame.data);
          if (runId && envelope.run_id !== runId)
            throw new ApiError('connection_lost', '结果编号不一致');
          return checkedResult(envelope.response);
        } else if (frame.event === 'failure') {
          const error = frame.data as { code?: string };
          throw new ApiError(error.code ?? 'connection_lost', '执行连接异常结束');
        }
      }
      if (done) throw new ApiError('connection_lost', '连接结束，但未收到最终结果');
    }
  } finally {
    await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}
