import type { Decision, Progress, Result } from './model';

export const scenarios = [
  {
    id: 'answered',
    name: '完整调查',
    question: '示例集群升级前，需要检查哪些配置与回退条件？',
    description: '搜索 → 读取 → 补充检索 → 带引用回答',
  },
  {
    id: 'clarification_needed',
    name: '需要补充信息',
    question: '可以直接升级我的示例集群吗？',
    description: '缺少版本信息时发起追问',
  },
  {
    id: 'insufficient_evidence',
    name: '证据不足',
    question: '示例文档是否承诺升级过程完全没有停机？',
    description: '检索后仍不足以支撑结论',
  },
  {
    id: 'generation_timeout',
    name: '调用超时',
    question: '请调查示例集群的升级限制。',
    description: '保留已经完成的步骤与证据',
  },
  {
    id: 'budget_exhausted',
    name: '预算耗尽',
    question: '请完整比较示例集群所有版本的升级路径。',
    description: '达到调查边界后安全结束',
  },
] as const;
export type ScenarioId = (typeof scenarios)[number]['id'];
const sources = [
  {
    citation_id: 1,
    rank: 1,
    title: '升级准备 / 配置核对',
    source_url: null,
    text: '【原创合成演示文档】\n\n在此虚构系统中，升级前需要记录当前版本、目标版本与配置快照，并核对兼容性清单。仅当清单确认兼容后，才能进入演练阶段。\n\n本段仅用于展示引用交互，不构成真实产品的操作建议。',
  },
  {
    citation_id: 2,
    rank: 2,
    title: '升级演练 / 回退条件',
    source_url: null,
    text: '【原创合成演示文档】\n\n示例流程要求在隔离环境中验证升级及回退。若配置校验失败、健康检查未通过或回退路径不可用，应暂停执行并补齐验证。\n\n本文不承诺无停机，也未涵盖任何真实版本。',
  },
];
export function demoScript(id: ScenarioId, runId: string): { events: Progress[]; result: Result } {
  const events: Progress[] = [];
  let elapsed = 0;
  let call = 0;
  let step = 0;
  const clarification = '请补充示例集群的当前版本和目标版本，并与原问题一起重新提交。';
  const invoke = (
    action: string,
    outcome = 'ok',
    ids: number[] = [],
    duration = 1.1,
    decision?: Decision,
  ) => {
    if (action === 'decide') step++;
    const invocation_id = `demo-${++call}`;
    events.push({
      run_id: runId,
      seq: events.length + 1,
      invocation_id,
      step,
      action,
      phase: 'started',
      elapsed_seconds: elapsed,
      outcome: null,
      evidence_ids: [],
    });
    elapsed += duration;
    events.push({
      run_id: runId,
      seq: events.length + 1,
      invocation_id,
      step,
      action,
      phase: 'completed',
      elapsed_seconds: elapsed,
      outcome,
      evidence_ids: ids,
      ...(decision ? { decision } : {}),
    });
  };
  const decide = (decision: Decision, duration = 1.1) =>
    invoke('decide', 'ok', [], duration, decision);
  if (id === 'clarification_needed') {
    decide({ action: 'clarify', question: clarification });
  } else {
    decide({
      action: 'search_docs',
      query:
        id === 'insufficient_evidence' ? '示例集群 升级 停机 保证' : '示例集群 升级前 配置 兼容性',
    });
    invoke('search_docs', 'ok', [1], 1.8);
    decide({ action: 'read_passage', evidence_id: 1 });
    invoke('read_passage', 'ok', [1], 0.3);
    if (id === 'answered') {
      decide({ action: 'search_docs', query: '示例集群 升级演练 回退条件' });
      invoke('search_docs', 'ok', [2], 1.5);
      decide({ action: 'read_passage', evidence_id: 2 });
      invoke('read_passage', 'ok', [2], 0.3);
    }
    if (id === 'generation_timeout' || id === 'budget_exhausted') {
      invoke('decide', id, [], id === 'generation_timeout' ? 3 : 1.2);
    } else {
      decide({ action: id === 'answered' ? 'answer' : 'abstain' }, 1.2);
    }
  }
  if (id === 'answered') invoke('validate_answer', 'ok', [1, 2], 0.2);
  invoke('finish', id, [], 0.01);
  return {
    events,
    result: {
      status: id,
      message:
        id === 'answered'
          ? '调查完成，请结合引用核对结论。'
          : id === 'clarification_needed'
            ? '需要补充必要信息。'
            : id === 'insufficient_evidence'
              ? '现有文档不足以支持这一结论。'
              : id === 'budget_exhausted'
                ? '已达到调查预算，未发布未完成的答案。'
                : '模型调用超时，调查已结束。',
      clarification: id === 'clarification_needed' ? clarification : '',
      blocks:
        id === 'answered'
          ? [
              {
                text: '先记录当前版本、目标版本与配置快照，核对兼容性清单，再进入升级演练。',
                citations: [1],
              },
              {
                text: '在隔离环境验证升级与回退。配置校验、健康检查或回退路径任一项未通过，都应暂停执行。',
                citations: [2],
              },
            ]
          : [],
      sources:
        id === 'clarification_needed' ? [] : id === 'answered' ? sources : sources.slice(0, 1),
      events: events
        .filter((e) => e.phase === 'completed')
        .map((e) => ({
          step: e.step,
          action: e.action,
          outcome: e.outcome!,
          elapsed_seconds: e.elapsed_seconds,
          evidence_ids: e.evidence_ids,
          ...(e.decision ? { decision: e.decision } : {}),
        })),
      usage: {
        model_calls: step,
        search_calls: id === 'clarification_needed' ? 0 : id === 'answered' ? 2 : 1,
        read_calls: id === 'clarification_needed' ? 0 : id === 'answered' ? 2 : 1,
        prompt_estimated_tokens: 0,
      },
      total_seconds: elapsed,
      agent_profile: 'synthetic-demo-not-an-evaluation',
    },
  };
}
export async function playDemo(
  id: ScenarioId,
  runId: string,
  signal: AbortSignal,
  onEvent: (event: Progress) => void,
): Promise<Result> {
  const script = demoScript(id, runId);
  for (const event of script.events) {
    if (signal.aborted) throw new DOMException('Aborted', 'AbortError');
    await new Promise<void>((resolve, reject) => {
      const abort = () => {
        clearTimeout(timer);
        reject(new DOMException('Aborted', 'AbortError'));
      };
      const timer = window.setTimeout(
        () => {
          signal.removeEventListener('abort', abort);
          resolve();
        },
        event.phase === 'started' ? 240 : 620,
      );
      signal.addEventListener('abort', abort, { once: true });
    });
    onEvent(event);
  }
  return script.result;
}
