import { z } from 'zod';
import {
  capabilitiesSchema,
  progressSchema,
  resultSchema,
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
): Promise<Result> {
  const response = await fetch(`/api/investigate${streaming ? '/stream' : ''}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: streaming ? 'text/event-stream' : 'application/json',
    },
    body: JSON.stringify({ query }),
    signal,
    cache: 'no-store',
  });
  if (!streaming || !response.ok) {
    const body = await response.json();
    const result = resultSchema.safeParse(body);
    if (result.success) return result.data;
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
          return envelope.response;
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
