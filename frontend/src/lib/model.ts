import { z } from 'zod';

export const sourceSchema = z.object({
  citation_id: z.number().int(),
  rank: z.number(),
  title: z.string(),
  text: z.string(),
  source_url: z.string().nullable(),
});
export const eventSchema = z.object({
  step: z.number(),
  action: z.string(),
  outcome: z.string(),
  elapsed_seconds: z.number().nonnegative(),
  evidence_ids: z.array(z.number()).default([]),
  validation_error: z.string().nullable().optional(),
});
export const resultSchema = z.object({
  status: z.string(),
  message: z.string(),
  blocks: z.array(z.object({ text: z.string(), citations: z.array(z.number()) })),
  sources: z.array(sourceSchema),
  clarification: z.string(),
  events: z.array(eventSchema),
  usage: z.object({
    model_calls: z.number(),
    search_calls: z.number(),
    read_calls: z.number(),
    prompt_estimated_tokens: z.number(),
  }),
  total_seconds: z.number(),
  agent_profile: z.string(),
});
export const progressSchema = z.object({
  run_id: z.string(),
  seq: z.number().int().positive(),
  invocation_id: z.string(),
  step: z.number().int(),
  action: z.string(),
  phase: z.enum(['started', 'completed']),
  elapsed_seconds: z.number().nonnegative(),
  outcome: z.string().nullable(),
  evidence_ids: z.array(z.number()),
});
export const capabilitiesSchema = z.object({
  agent_enabled: z.boolean(),
  agent_profile: z.string().nullable(),
  agent_streaming: z.boolean().default(false),
});
export type Result = z.infer<typeof resultSchema>;
export type Source = z.infer<typeof sourceSchema>;
export type Progress = z.infer<typeof progressSchema>;
export type Capabilities = z.infer<typeof capabilitiesSchema>;
export type NodeKey = 'agent' | 'search' | 'read' | 'finish';
export type Run = {
  id: string;
  query: string;
  demo: boolean;
  started: number;
  state: string;
  progress: Progress[];
  result?: Result;
  error?: string;
  transport: 'demo' | 'stream' | 'batch';
};
export type Invocation = {
  id: string;
  action: string;
  node: NodeKey;
  step: number;
  start?: number;
  end?: number;
  outcome?: string;
  evidenceIds: number[];
};

export function nodeFor(action: string): NodeKey {
  if (action === 'search_docs') return 'search';
  if (action === 'read_passage') return 'read';
  if (['finish', 'answer', 'validate_answer'].includes(action)) return 'finish';
  return 'agent';
}
export const actionNames: Record<string, string> = {
  decide: 'Agent 决策',
  plan: '制定调查计划',
  assess: '检查证据覆盖',
  review_answer: '复核答案',
  search_docs: '搜索文档',
  read_passage: '读取证据',
  validate_answer: '校验答案',
  answer: '准备答案',
  finish: '结束调查',
};
export const statusLabels: Record<string, string> = {
  idle: '尚未开始',
  running: '调查进行中',
  answered: '已生成答案',
  clarification_needed: '需要补充信息',
  insufficient_evidence: '证据不足',
  budget_exhausted: '已达预算上限',
  cancelled: '调查已停止',
  cancel_requested: '已请求停止',
  generation_failed: '模型调用失败',
  generation_timeout: '模型调用超时',
  retrieval_failed: '文档检索失败',
  invalid_action: '动作格式无效',
  invalid_answer: '答案校验失败',
  agent_unavailable: '调查服务未启用',
  service_busy: '服务繁忙',
  connection_lost: '连接已中断',
  stream_overflow: '事件流已中断',
  agent_failed: '调查执行失败',
};
export function toneFor(status: string): 'neutral' | 'brand' | 'success' | 'warning' | 'danger' {
  if (status === 'answered' || status === 'ok' || status === 'completed') return 'success';
  if (status === 'running') return 'brand';
  if (
    [
      'clarification_needed',
      'insufficient_evidence',
      'budget_exhausted',
      'cancel_requested',
      'cancelled',
      'service_busy',
      'agent_unavailable',
    ].includes(status)
  )
    return 'warning';
  if (status === 'idle') return 'neutral';
  return 'danger';
}
export function outcomeTone(outcome?: string): 'neutral' | 'success' | 'warning' | 'danger' {
  if (!outcome) return 'neutral';
  if (/failed|timeout|invalid|rejected/.test(outcome)) return 'danger';
  if (/clarification|insufficient|exhausted|cancel|limit|revise/.test(outcome)) return 'warning';
  if (['ok', 'answered', 'accepted', 'drafted', 'checked', 'created'].includes(outcome))
    return 'success';
  return 'neutral';
}
export function invocations(events: Progress[]): Invocation[] {
  const items = new Map<string, Invocation>();
  for (const e of events) {
    const item = items.get(e.invocation_id) ?? {
      id: e.invocation_id,
      action: e.action,
      node: nodeFor(e.action),
      step: e.step,
      evidenceIds: [],
    };
    if (e.phase === 'started') item.start = e.elapsed_seconds;
    else {
      item.end = e.elapsed_seconds;
      item.outcome = e.outcome ?? undefined;
      item.evidenceIds = e.evidence_ids;
    }
    items.set(e.invocation_id, item);
  }
  return [...items.values()];
}
// Ordinals belong to an action type, independently of model rounds and global sequence.
export function invocationOrdinal(calls: Invocation[], call: Invocation): number {
  const index = calls.findIndex((item) => item.id === call.id);
  return calls.slice(0, index + 1).filter((item) => item.action === call.action).length;
}
export function batchProgress(result: Result, id: string): Progress[] {
  // A legacy completion timestamp is not a start time or an operation duration.
  return result.events.map((e, i) => ({
    ...e,
    run_id: id,
    seq: i + 1,
    invocation_id: `record-${i}`,
    phase: 'completed',
    outcome: e.outcome,
  }));
}
export function safeSourceUrl(value: string | null): string | undefined {
  if (!value) return undefined;
  try {
    const url = new URL(value);
    return ['http:', 'https:'].includes(url.protocol) && !url.username && !url.password
      ? url.href
      : undefined;
  } catch {
    return undefined;
  }
}
