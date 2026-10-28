import { useRef, useState } from 'react';
import { Button } from './ui';
import { commandRun, getRun } from '../lib/api';
import { persistentRun, type Run } from '../lib/model';
import { advanceDemo } from '../lib/demo';

export const recoveryMessages: Record<string, string> = {
  call_outcome_unknown:
    '服务在调用结果保存前中断，无法确认外部调用是否完成。为避免自动重复调用，本调查保留记录并停止恢复；可以新建调查。',
  snapshot_or_model_mismatch:
    '原模型配置或文档快照已不可用，无法安全继续。请恢复原配置和快照后新建调查。',
  executor_failed: '检查点执行失败，记录已保留。请检查本地服务后新建调查。',
  service_interrupted: '服务曾中断。继续时将从已保存边界恢复，沿用证据和剩余预算。',
  service_shutdown: '服务关闭前已保存检查点，可以继续原调查。',
  budget_exhausted: '累计执行预算已耗尽，继续和补充信息不会重置预算。',
};
export function InvestigationControls({ run, onRun }: { run: Run; onRun: (run: Run) => void }) {
  const [text, setText] = useState('');
  const [pending, setPending] = useState(false);
  const guard = useRef(false);
  const [error, setError] = useState('');
  const state = run.durable!;
  async function act(action: string) {
    if (guard.current) return;
    guard.current = true;
    setPending(true);
    setError('');
    const controller = new AbortController();
    try {
      if (run.demo) {
        onRun(advanceDemo(run, action, text));
        setText('');
        return;
      }
      const next = await commandRun(state, action, controller.signal, text);
      onRun(persistentRun(next));
      if (action === 'supplement') setText('');
    } catch {
      setError('操作未完成或页面状态已过期。已重新读取调查，请核对当前问题和状态。');
      try {
        onRun(persistentRun(await getRun(run.id, controller.signal)));
      } catch {
        /* reconnect retains history */
      }
    } finally {
      guard.current = false;
      setPending(false);
    }
  }
  const active = ['queued', 'running', 'resuming'].includes(state.status);
  const canCancel = [
    'queued',
    'running',
    'resuming',
    'pausing',
    'paused',
    'waiting_input',
    'interrupted',
  ].includes(state.status);
  return (
    <div className="investigation-controls" aria-label="调查控制">
      {run.demo && (
        <p>
          原创合成演示：补充后可暂停、继续。此场景只保留在当前页面；跨进程恢复请运行离线合成后端。
        </p>
      )}
      {run.reconnecting && <p role="status">进度连接已断开，正在重连。调查保留在服务端。</p>}
      {recoveryMessages[state.reason] && <p>{recoveryMessages[state.reason]}</p>}
      {state.status === 'pausing' && <p>等待当前调用返回并保存结果后暂停。</p>}
      {state.status === 'cancelling' && <p>等待当前调用结束后取消，不再推进后续操作。</p>}
      {state.status === 'resuming' && <p>正在校验配置与文档快照，并恢复执行。</p>}
      {state.status === 'waiting_input' && (
        <form
          onSubmit={(event) => {
            event.preventDefault();
            void act('supplement');
          }}
        >
          <label htmlFor={`supplement-${run.id}`}>{state.clarification}</label>
          <textarea
            id={`supplement-${run.id}`}
            value={text}
            onChange={(event) => setText(event.target.value)}
            maxLength={2000}
            rows={3}
            required
            disabled={pending}
            placeholder="补充本次调查需要的信息…"
          />
          <Button type="submit" size="sm" disabled={pending || !text.trim()}>
            提交补充并继续
          </Button>
        </form>
      )}
      {!!state.supplements.length && (
        <details className="supplement-history">
          <summary>已提交的补充信息（{state.supplements.length}）</summary>
          {state.supplements.map((item) => (
            <div key={item.wait_id}>
              <p>{item.question}</p>
              <p>你：{item.text}</p>
            </div>
          ))}
        </details>
      )}
      <div className="run-control-buttons">
        {active && (
          <Button size="sm" variant="outline" disabled={pending} onClick={() => void act('pause')}>
            暂停调查
          </Button>
        )}
        {['paused', 'interrupted'].includes(state.status) && (
          <Button size="sm" disabled={pending} onClick={() => void act('resume')}>
            继续调查
          </Button>
        )}
        {canCancel && (
          <Button size="sm" variant="ghost" disabled={pending} onClick={() => void act('cancel')}>
            取消调查
          </Button>
        )}
      </div>
      {error && <p role="alert">{error}</p>}
      <small>
        调查 {run.id.slice(0, 8)} · 已用主动执行时间 {state.active_seconds.toFixed(1)} 秒
      </small>
    </div>
  );
}
