import { useEffect, useRef, useState } from 'react';
import { AnimatePresence, motion, MotionConfig } from 'motion/react';
import {
  ArrowRight,
  ArrowDown,
  AlertCircle,
  ArrowUp,
  BookOpen,
  Check,
  ChevronLeft,
  ChevronRight,
  Circle,
  CircleHelp,
  Clock3,
  Copy,
  ExternalLink,
  FileSearch,
  FileText,
  History,
  Layers2,
  Loader2,
  Moon,
  Network,
  Plus,
  Radio,
  Search,
  ShieldCheck,
  Square,
  Sun,
  Terminal,
  Workflow,
} from 'lucide-react';
import {
  Button,
  Dialog,
  DialogClose,
  Select,
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
  Tip,
  TooltipProvider,
} from './components/ui';
import { Inspector } from './components/Inspector';
import { ExecutionGraph } from './components/ExecutionGraph';
import { ApiError, capabilities, investigate } from './lib/api';
import { playDemo, scenarios, type ScenarioId } from './lib/demo';
import {
  actionNames,
  batchProgress,
  invocations,
  invocationOrdinal,
  outcomeTone,
  safeSourceUrl,
  statusLabels,
  toneFor,
  type Capabilities,
  type Invocation,
  type Run,
  type Source,
} from './lib/model';

function Brand({ small = false }: { small?: boolean }) {
  return (
    <span className={`brand-mark ${small ? 'small' : ''}`} aria-label="RAG">
      <Layers2 size={small ? 15 : 22} strokeWidth={1.7} />
    </span>
  );
}
function Status({ state }: { state: string }) {
  return (
    <span className={`status-label tone-${toneFor(state)}`}>
      {state === 'running' ? (
        <Loader2 size={12} className="spin" />
      ) : (
        <Circle size={7} fill="currentColor" />
      )}
      {statusLabels[state] ?? state}
    </span>
  );
}
function seconds(value: number) {
  return `${value.toFixed(1)}s`;
}
function timeLabel(run: Run) {
  return new Date(run.started).toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' });
}

function Evidence({ source }: { source: Source }) {
  const url = safeSourceUrl(source.source_url);
  return (
    <article className="evidence-detail">
      <div className="section-kicker">证据 {String(source.citation_id).padStart(2, '0')}</div>
      <h3>{source.title || '未命名文档'}</h3>
      <div className="evidence-text">{source.text}</div>
      {url && (
        <a className="text-link" href={url} target="_blank" rel="noopener noreferrer">
          打开原文 <ExternalLink size={13} />
        </a>
      )}
    </article>
  );
}

export default function App() {
  const [theme, setTheme] = useState<'light' | 'dark'>(() =>
    document.documentElement.dataset.theme === 'dark' ? 'dark' : 'light',
  );
  const [mode, setMode] = useState('demo');
  const [model, setModel] = useState(() => {
    try {
      return localStorage.getItem('zhrag-agent-model') ?? '';
    } catch {
      return '';
    }
  });
  const [scenario, setScenario] = useState<ScenarioId>('answered');
  const [query, setQuery] = useState<string>(scenarios[0].question);
  const [runs, setRuns] = useState<Run[]>([]);
  const [selectedRun, setSelectedRun] = useState<string>();
  const [selectedCall, setSelectedCall] = useState<string>();
  const [detailTab, setDetailTab] = useState('step');
  const [inspectorOpen, setInspectorOpen] = useState(false);
  const [sourceId, setSourceId] = useState<number>();
  const [evidenceOpen, setEvidenceOpen] = useState(false);
  const evidenceOrigin = useRef<HTMLElement | null>(null);
  const [caps, setCaps] = useState<Capabilities>();
  const [connection, setConnection] = useState('idle');
  const [connectionAttempt, setConnectionAttempt] = useState(0);
  const [copied, setCopied] = useState(false);
  const [answerEntering, setAnswerEntering] = useState(false);
  const [, setTick] = useState(0);
  const abortRef = useRef<AbortController | null>(null);
  const formRef = useRef<HTMLFormElement>(null);
  const timelineRef = useRef<HTMLDivElement>(null);
  const executionRef = useRef<HTMLElement>(null);
  const current = runs.find((run) => run.id === selectedRun);
  const busy = runs.some((run) => run.state === 'running');
  const selectedModel = caps?.agent_models.find((option) => option.id === model);
  const modelReady = !caps?.agent_models.length || !!selectedModel;
  const calls = invocations(current?.progress ?? []);
  const latest = calls.at(-1);
  const activeCall = current?.state === 'running' && latest?.end === undefined ? latest : undefined;
  const actionSummary =
    current?.state === 'running'
      ? activeCall
        ? `${actionNames[activeCall.action] ?? activeCall.action} · 第 ${calls.filter((call) => call.action === activeCall.action).length} 次 · 执行中`
        : '等待下一步执行'
      : current
        ? (statusLabels[current.state] ?? current.state)
        : '等待发起调查';
  const needsAttention = current && ['warning', 'danger'].includes(toneFor(current.state));
  const selected = calls.find((call) => call.id === selectedCall) ?? latest;
  const sameActionCalls = selected ? calls.filter((call) => call.action === selected.action) : [];
  const selectedActionIndex = sameActionCalls.findIndex((call) => call.id === selected?.id);
  const previousActionCall = sameActionCalls[selectedActionIndex - 1];
  const nextActionCall = sameActionCalls[selectedActionIndex + 1];
  const latestActionCall = sameActionCalls.at(-1);
  const latestForSelectedNode =
    selected && calls.filter((call) => call.node === selected.node).at(-1);
  const sources = current?.result?.sources ?? [];
  const evidence = sources.find((s) => s.citation_id === sourceId);
  const elapsed =
    current?.result?.total_seconds ??
    (current?.demo
      ? (current.progress.at(-1)?.elapsed_seconds ?? 0)
      : current?.state === 'running'
        ? (Date.now() - current.started) / 1000
        : (current?.progress.at(-1)?.elapsed_seconds ?? 0));
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try {
      localStorage.setItem('zhrag-theme', theme);
    } catch {
      /* Theme still works without persistence. */
    }
  }, [theme]);
  useEffect(() => () => abortRef.current?.abort(), []);
  useEffect(() => {
    if (!model) return;
    try {
      localStorage.setItem('zhrag-agent-model', model);
    } catch {
      /* Model selection still works without persistence. */
    }
  }, [model]);
  useEffect(() => {
    if (!selectedCall && timelineRef.current)
      timelineRef.current.scrollTop = timelineRef.current.scrollHeight;
  }, [current?.progress.length, selectedRun, selectedCall]);
  useEffect(() => {
    if (!busy) return;
    const timer = window.setInterval(() => setTick((n) => n + 1), 1000);
    return () => clearInterval(timer);
  }, [busy]);
  useEffect(() => {
    if (mode !== 'live') return;
    const abort = new AbortController();
    setConnection('connecting');
    capabilities(abort.signal)
      .then((value) => {
        if (abort.signal.aborted) return;
        setCaps(value);
        setModel((previous) =>
          value.agent_models.some((option) => option.id === previous)
            ? previous
            : (value.default_agent_model ?? value.agent_models[0]?.id ?? ''),
        );
        setConnection(value.agent_enabled ? 'ready' : 'disabled');
      })
      .catch(() => {
        if (!abort.signal.aborted) {
          setCaps(undefined);
          setConnection('offline');
        }
      });
    return () => abort.abort();
  }, [mode, connectionAttempt]);
  function update(id: string, fn: (run: Run) => Run) {
    setRuns((previous) => previous.map((run) => (run.id === id ? fn(run) : run)));
  }
  async function start() {
    if (busy || !query.trim() || (mode === 'live' && (connection !== 'ready' || !modelReady)))
      return;
    const id = crypto.randomUUID();
    const abort = new AbortController();
    abortRef.current = abort;
    const demo = mode === 'demo';
    const streaming = !!caps?.agent_streaming;
    const run: Run = {
      id,
      query: query.trim(),
      demo,
      model: demo ? undefined : selectedModel?.id,
      modelName: demo ? undefined : selectedModel?.name,
      started: Date.now(),
      state: 'running',
      progress: [],
      transport: demo ? 'demo' : streaming ? 'stream' : 'batch',
    };
    setRuns((previous) => [run, ...previous].slice(0, 10));
    setSelectedRun(id);
    setSelectedCall(undefined);
    setSourceId(undefined);
    setDetailTab('step');
    setAnswerEntering(false);
    try {
      const receive = (event: Run['progress'][number]) => {
        if (!abort.signal.aborted)
          update(id, (row) => ({ ...row, progress: [...row.progress, event] }));
      };
      const result = demo
        ? await playDemo(scenario, id, abort.signal, receive)
        : await investigate(run.query, streaming, abort.signal, receive, run.model);
      if (abort.signal.aborted) return;
      setAnswerEntering(true);
      update(id, (row) => ({
        ...row,
        state: result.status,
        result,
        progress: !demo && !streaming ? batchProgress(result, id) : row.progress,
      }));
    } catch (error) {
      const state = abort.signal.aborted
        ? demo
          ? 'cancelled'
          : 'cancel_requested'
        : error instanceof ApiError
          ? error.code
          : 'connection_lost';
      update(id, (row) => ({
        ...row,
        state,
        error: abort.signal.aborted
          ? demo
            ? '模拟调查已停止。'
            : '已断开接收并请求停止；服务端会在当前依赖调用结束后停止后续操作。'
          : error instanceof ApiError
            ? error.message
            : '未收到完整有效的结果。可以保留当前记录，检查服务后重新发起。',
      }));
    } finally {
      if (abortRef.current === abort) abortRef.current = null;
    }
  }
  function inspect(call: Invocation) {
    setSelectedCall(call.id);
    setDetailTab('step');
    setInspectorOpen(true);
  }
  function cite(id: number) {
    setSourceId(id);
    setDetailTab('evidence');
    if (
      window.matchMedia('(max-width: 1180px)').matches &&
      sources.some((source) => source.citation_id === id)
    ) {
      openEvidence();
    } else {
      setInspectorOpen(true);
    }
  }
  function openEvidence() {
    evidenceOrigin.current =
      document.activeElement instanceof HTMLElement ? document.activeElement : null;
    setEvidenceOpen(true);
  }
  function fresh() {
    if (busy) return;
    setAnswerEntering(false);
    setSelectedRun(undefined);
    setSelectedCall(undefined);
    setSourceId(undefined);
    setQuery(mode === 'demo' ? scenarios.find((s) => s.id === scenario)!.question : '');
  }
  async function copyAnswer() {
    if (!current?.result) return;
    try {
      await navigator.clipboard.writeText(
        current.result.blocks
          .map((block) => `${block.text} ${block.citations.map((id) => `[${id}]`).join('')}`)
          .join('\n\n'),
      );
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1800);
    } catch {
      setCopied(false);
    }
  }
  return (
    <MotionConfig reducedMotion="user">
      <TooltipProvider delayDuration={250}>
        <div className="glacier-wallpaper" aria-hidden="true" />
        <div className="app-shell">
          <header className="topbar">
            <div className="wordmark">
              <a href="/workbench/" className="brand-home" aria-label="RAG 工作台首页">
                <Brand />
                <span>RAG</span>
              </a>
              <span className="wordmark-divider" />
              <span className="breadcrumb">
                工作空间 <ChevronRight size={12} /> 文档调查
              </span>
            </div>
            <div className="topbar-actions">
              <span className="topbar-note">
                <span className="tiny-dot" /> 本地工作空间
              </span>
              <Button variant="outline" aria-label="新调查" disabled={busy} onClick={fresh}>
                <Plus size={15} /> 新调查
              </Button>
            </div>
          </header>
          <aside className="rail" aria-label="工作台导航">
            <div className="rail-top">
              <Tip label="文档调查">
                <Button
                  variant="ghost"
                  size="icon"
                  className="rail-active"
                  aria-label="文档调查"
                  onClick={() => document.getElementById('query')?.focus()}
                >
                  <Network size={20} />
                </Button>
              </Tip>
              <Dialog
                title="本次会话"
                description="保留最近 10 次运行。记录只存在当前页面内，刷新后清空。"
                trigger={
                  <Button variant="ghost" size="icon" aria-label="运行历史">
                    <History size={19} />
                  </Button>
                }
              >
                <div className="history-list">
                  {runs.length ? (
                    runs.map((run) => (
                      <DialogClose key={run.id} asChild>
                        <button
                          className={`history-item ${current?.id === run.id ? 'selected' : ''}`}
                          disabled={busy}
                          onClick={() => {
                            setSelectedRun(run.id);
                            setSelectedCall(undefined);
                            setSourceId(undefined);
                          }}
                        >
                          <span>
                            <span className="history-query">{run.query}</span>
                            <span className="history-meta">
                              {timeLabel(run)} ·{' '}
                              {run.demo ? '模拟演示' : (run.modelName ?? '服务运行')}
                            </span>
                          </span>
                          <Status state={run.state} />
                        </button>
                      </DialogClose>
                    ))
                  ) : (
                    <p className="empty-copy">发起一次调查后，在这里查看执行记录。</p>
                  )}
                </div>
              </Dialog>
            </div>
            <div className="rail-bottom">
              <Tip label={theme === 'light' ? '切换深色' : '切换浅色'}>
                <Button
                  variant="ghost"
                  size="icon"
                  aria-label={theme === 'light' ? '切换深色' : '切换浅色'}
                  onClick={() => setTheme((t) => (t === 'light' ? 'dark' : 'light'))}
                >
                  {theme === 'light' ? <Moon size={18} /> : <Sun size={18} />}
                </Button>
              </Tip>
              <Dialog
                title="让每一步调查，都有据可查"
                description="RAG · 中文文档调查工作台"
                trigger={
                  <Button variant="ghost" size="icon" aria-label="关于工作台">
                    <CircleHelp size={19} />
                  </Button>
                }
              >
                <div className="about-list">
                  <p>
                    <Workflow size={18} />
                    <span>
                      查看 Agent 如何搜索、读取并补充证据。节点代表动作类型，时间线保留每次调用。
                    </span>
                  </p>
                  <p>
                    <ShieldCheck size={18} />
                    <span>
                      页面展示执行记录与引用，不展示模型内部思考。答案格式校验通过，也不等于事实经过独立验证。
                    </span>
                  </p>
                  <p>
                    <Layers2 size={18} />
                    <span>
                      模拟演示使用原创合成资料，按选定脚本播放，不分析输入问题。连接服务后，点击开始才会发起真实调查。
                    </span>
                  </p>
                </div>
              </Dialog>
              <span className="rail-avatar">ZH</span>
            </div>
          </aside>
          <div className="workspace">
            <main>
              <div className="page-heading">
                <div>
                  <div className="section-kicker">文档调查工作台</div>
                  <h1>每一步，都看得见。</h1>
                  <p>从一个问题开始，让答案与证据相连。</p>
                </div>
                <div className="mode-control">
                  <Select
                    label="运行模式"
                    value={mode}
                    onChange={(value) => {
                      setMode(value);
                      if (value === 'live') setQuery('');
                    }}
                    disabled={busy}
                    options={[
                      { value: 'demo', label: '模拟演示' },
                      { value: 'live', label: '连接服务' },
                    ]}
                  />
                  <span
                    className={`connection-note ${connection === 'ready' && mode === 'live' ? 'is-ready' : ''}`}
                  >
                    <span className="tiny-dot" />
                    {mode === 'demo'
                      ? '原创合成数据 · 无 API 调用'
                      : connection === 'ready'
                        ? '服务已连接'
                        : connection === 'connecting'
                          ? '正在检查服务'
                          : connection === 'disabled'
                            ? 'Agent 尚未启用'
                            : '未连接到本地服务'}
                    {mode === 'live' && ['offline', 'disabled'].includes(connection) && (
                      <button
                        type="button"
                        className="reconnect"
                        onClick={() => setConnectionAttempt((n) => n + 1)}
                      >
                        重新检查
                      </button>
                    )}
                  </span>
                </div>
              </div>
              {current && (
                <div
                  className={`mobile-progress ${current.state === 'running' ? 'is-running' : ''}`}
                >
                  <div>
                    <span className="mobile-progress-label">
                      {current.demo ? '模拟演示' : '本次调查'} · 当前进度
                    </span>
                    <span className="mobile-progress-action">{actionSummary}</span>
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    aria-controls="execution-flow"
                    onClick={() => {
                      executionRef.current?.focus({ preventScroll: true });
                      executionRef.current?.scrollIntoView({
                        behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches
                          ? 'instant'
                          : 'smooth',
                        block: 'start',
                      });
                    }}
                  >
                    查看流程 <ArrowDown size={14} />
                  </Button>
                </div>
              )}
              <div
                className={`workbench ${inspectorOpen ? 'inspector-open' : 'inspector-collapsed'}`}
              >
                <section className="conversation-pane" aria-label="问题与答案">
                  <div className="pane-heading">
                    <span>
                      <FileSearch size={16} /> 调查任务
                    </span>
                    <span className="subtle-number">01</span>
                  </div>
                  <div className="conversation-scroll">
                    {current ? (
                      <>
                        <div className="query-meta">
                          <span className="person-dot">你</span>
                          <span>{timeLabel(current)}</span>
                          <span className="data-tag">
                            {current.demo ? '模拟数据' : (current.modelName ?? '本次运行')}
                          </span>
                        </div>
                        <h2 className="submitted-query">{current.query}</h2>
                        <div className="answer-heading">
                          <Brand small />
                          <span>RAG</span>
                          <span className="answer-heading-line" />
                        </div>
                        <div className="answer-status" aria-live="polite">
                          <Status state={current.state} />
                        </div>
                        <AnimatePresence mode="wait">
                          <motion.div
                            key={current.result ? 'result' : current.state}
                            initial={{ opacity: 0, y: 4 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0 }}
                            transition={{ duration: 0.16 }}
                            onAnimationComplete={() => {
                              if (current.result) setAnswerEntering(false);
                            }}
                          >
                            {current.result?.blocks.length ? (
                              <div className="answer-body">
                                {current.result.blocks.map((block, index) => (
                                  <p key={index}>
                                    {block.text}
                                    <span className="inline-citations">
                                      {block.citations.map((id) => (
                                        <button
                                          key={id}
                                          onClick={() => cite(id)}
                                          aria-label={`查看引用 ${id}`}
                                          aria-pressed={sourceId === id}
                                        >
                                          <span>{id}</span>
                                        </button>
                                      ))}
                                    </span>
                                  </p>
                                ))}
                                <div className="answer-tools">
                                  <Button variant="ghost" size="sm" onClick={copyAnswer}>
                                    {copied ? <Check size={13} /> : <Copy size={13} />}{' '}
                                    {copied ? '已复制' : '复制答案'}
                                  </Button>
                                  <span>{sources.length} 份引用证据</span>
                                </div>
                                <p className="answer-caution">
                                  <ShieldCheck size={13} />
                                  引用可追溯，结论仍需结合原文核对。
                                </p>
                              </div>
                            ) : current.result ? (
                              <div className="outcome-copy">
                                <p>{current.result.clarification || current.result.message}</p>
                                {current.result.status === 'clarification_needed' && (
                                  <p className="muted">将补充信息与原问题一起提交，发起新调查。</p>
                                )}
                              </div>
                            ) : current.state === 'running' ? (
                              <div className="working-copy">
                                <p>
                                  {current.transport === 'batch' ? (
                                    '等待服务返回完整结果。此服务暂不支持实时步骤。'
                                  ) : (
                                    <>
                                      调查正在进行，执行过程显示在
                                      <span className="desktop-direction">右侧</span>
                                      <span className="mobile-direction">下方</span>。
                                    </>
                                  )}
                                </p>
                                <div className="working-rule">
                                  <span />
                                </div>
                                <span>答案通过校验后在此发布</span>
                              </div>
                            ) : (
                              <p className="outcome-copy">{current.error}</p>
                            )}
                          </motion.div>
                        </AnimatePresence>
                      </>
                    ) : (
                      <div className="welcome">
                        <div className="welcome-symbol">
                          <FileSearch size={27} />
                          <span />
                        </div>
                        <h2>
                          一个问题，
                          <br />
                          一条清晰的证据路径。
                        </h2>
                        <p>让 Agent 查找相关文档、读取证据，并决定下一步。</p>
                        <div className="intro-steps">
                          <span>
                            <Search size={13} /> 检索
                          </span>
                          <ChevronRight size={12} />
                          <span>
                            <BookOpen size={13} /> 求证
                          </span>
                          <ChevronRight size={12} />
                          <span>
                            <FileText size={13} /> 回答
                          </span>
                        </div>
                        {mode === 'demo' && (
                          <div className="scenario-intro">
                            <div className="section-kicker">试着探索</div>
                            <button
                              onClick={() => {
                                setScenario('answered');
                                setQuery(scenarios[0].question);
                              }}
                            >
                              <span>查看一次完整调查</span>
                              <ArrowUp size={14} />
                            </button>
                            <button
                              onClick={() => {
                                setScenario('clarification_needed');
                                setQuery(scenarios[1].question);
                              }}
                            >
                              <span>信息不足时，Agent 会怎么做？</span>
                              <ArrowUp size={14} />
                            </button>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                  <form
                    ref={formRef}
                    className="composer"
                    onSubmit={(event) => {
                      event.preventDefault();
                      void start();
                    }}
                  >
                    {mode === 'demo' && (
                      <div className="scenario-control">
                        <span>演示场景</span>
                        <Select
                          label="演示场景"
                          value={scenario}
                          disabled={busy}
                          onChange={(value) => {
                            setScenario(value as ScenarioId);
                            setQuery(scenarios.find((s) => s.id === value)!.question);
                          }}
                          options={scenarios.map((s) => ({ value: s.id, label: s.name }))}
                        />
                      </div>
                    )}
                    <div className="input-wrap">
                      <label className="sr-only" htmlFor="query">
                        调查问题
                      </label>
                      <textarea
                        id="query"
                        value={query}
                        onChange={(event) => setQuery(event.target.value)}
                        disabled={busy}
                        maxLength={2000}
                        placeholder="描述你想调查的问题…"
                        rows={3}
                        onKeyDown={(event) => {
                          if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) {
                            event.preventDefault();
                            formRef.current?.requestSubmit();
                          }
                        }}
                      />
                      <div
                        className={`composer-actions ${mode === 'live' && caps?.agent_models.length ? 'has-model' : ''}`}
                      >
                        <span className={`input-hint ${query.length > 1800 ? 'is-count' : ''}`}>
                          {query.length > 1800 ? `${query.length}/2000` : '⌘ / Ctrl + Enter'}
                        </span>
                        {mode === 'live' && !!caps?.agent_models.length && (
                          <div className="model-control">
                            <Select
                              label="调查模型"
                              value={model}
                              displayValue={
                                selectedModel?.name === 'DeepSeek V4.1 Flash'
                                  ? 'DeepSeek'
                                  : selectedModel?.name
                              }
                              side="top"
                              onChange={setModel}
                              disabled={busy || answerEntering || connection !== 'ready'}
                              options={caps.agent_models.map((option) => ({
                                value: option.id,
                                label: option.name,
                              }))}
                            />
                          </div>
                        )}
                        {busy ? (
                          <Button
                            variant="outline"
                            size="sm"
                            type="button"
                            onClick={(event) => {
                              event.preventDefault();
                              abortRef.current?.abort();
                            }}
                          >
                            <Square size={11} /> 停止
                          </Button>
                        ) : (
                          <Button
                            type="submit"
                            size="sm"
                            disabled={
                              !query.trim() ||
                              (mode === 'live' && (connection !== 'ready' || !modelReady))
                            }
                          >
                            {mode === 'demo' ? '运行演示' : '开始调查'}
                            <ArrowUp size={14} />
                          </Button>
                        )}
                      </div>
                    </div>
                    <p className="composer-note">
                      {mode === 'demo'
                        ? '按所选脚本演示交互，不分析输入问题。'
                        : selectedModel
                          ? `使用 ${selectedModel.name} 调查，切换仅对新运行生效。`
                          : '发起后将使用已配置的模型与检索服务。'}
                    </p>
                  </form>
                </section>
                <section
                  ref={executionRef}
                  id="execution-flow"
                  className="execution-pane"
                  aria-label="执行流程"
                  tabIndex={-1}
                >
                  <div className="pane-heading">
                    <span>
                      <Workflow size={16} /> 执行流程
                    </span>
                    <span className="subtle-number">02</span>
                  </div>
                  <div className="graph-toolbar">
                    <span className="graph-view-label">
                      <Network size={13} /> Agent graph
                    </span>
                    <span className="graph-key">
                      <i className="key-active" />
                      执行中
                      <i className="key-done" />
                      已完成
                    </span>
                  </div>
                  <div className={`current-action ${activeCall ? 'is-running' : ''}`} role="status">
                    {activeCall ? <Loader2 size={16} className="spin" /> : <Workflow size={16} />}
                    <span>{actionSummary}</span>
                    <span
                      className="action-elapsed mono"
                      aria-label={`运行耗时 ${seconds(elapsed)}`}
                    >
                      <Clock3 size={12} /> {seconds(elapsed)}
                    </span>
                  </div>
                  <ExecutionGraph
                    run={current}
                    selected={selected?.id}
                    onSelect={inspect}
                    theme={theme}
                  />
                  <p className="graph-selection-note">
                    虚线标记正在查看的调用；节点状态与连线保留实际进度。
                  </p>
                  <div className="timeline-heading">
                    <span>
                      执行时间线 <span className="count">{calls.length}</span>
                    </span>
                    {selectedCall ? (
                      <Button variant="ghost" size="sm" onClick={() => setSelectedCall(undefined)}>
                        <Radio size={12} />
                        跟随执行
                      </Button>
                    ) : (
                      <span className="muted text-xs">
                        {current?.transport === 'batch' ? '完成后返回的记录' : '每次调用，独立记录'}
                      </span>
                    )}
                  </div>
                  <div ref={timelineRef} className="timeline" aria-label="逐次调用记录">
                    {calls.length ? (
                      calls.map((call, index) => (
                        <button
                          key={call.id}
                          className={`timeline-row ${call.id === selected?.id ? 'selected' : ''}`}
                          aria-pressed={call.id === selected?.id}
                          onClick={() => inspect(call)}
                        >
                          <span className="timeline-index">
                            {String(index + 1).padStart(2, '0')}
                          </span>
                          <span
                            className={`timeline-marker tone-${outcomeTone(call.outcome)} ${call.end === undefined && current?.state === 'running' ? 'is-running' : ''}`}
                          >
                            {call.end === undefined && current?.state === 'running' ? (
                              <Loader2 size={12} className="spin" />
                            ) : call.end === undefined ? (
                              <Square size={9} />
                            ) : ['danger', 'warning'].includes(outcomeTone(call.outcome)) ? (
                              <AlertCircle size={12} />
                            ) : (
                              <Check size={11} />
                            )}
                          </span>
                          <span className="timeline-action">
                            <span className="timeline-action-title">
                              {actionNames[call.action] ?? call.action}
                              <span className="invocation-ordinal">
                                {' '}
                                · 第 {invocationOrdinal(calls, call)} 次
                              </span>
                            </span>
                            <span className="timeline-round">
                              {call.step ? `模型轮次 ${call.step}` : ''}
                            </span>
                          </span>
                          <span className="timeline-duration mono">
                            {call.end !== undefined && call.start !== undefined
                              ? seconds(call.end - call.start)
                              : call.end !== undefined
                                ? `+${seconds(call.end)}`
                                : current?.state === 'running'
                                  ? '执行中'
                                  : '未完成'}
                          </span>
                          <ChevronRight size={12} />
                        </button>
                      ))
                    ) : (
                      <div className="timeline-empty">
                        <span className="empty-track" />
                        <p>
                          搜索、读取、补充检索…
                          <br />
                          <span>每一步都会在这里留下记录。</span>
                        </p>
                      </div>
                    )}
                  </div>
                </section>
                <Inspector open={inspectorOpen} onOpenChange={setInspectorOpen}>
                  {needsAttention && (
                    <div className={`inspector-outcome tone-${toneFor(current.state)}`}>
                      <div>
                        <AlertCircle size={18} />
                        <strong>{statusLabels[current.state] ?? current.state}</strong>
                      </div>
                      <p>
                        {current.result?.clarification || current.error || current.result?.message}
                      </p>
                      {sources.length > 0 && (
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => cite(sources[0].citation_id)}
                        >
                          <BookOpen size={14} />
                          查看已读取证据
                        </Button>
                      )}
                    </div>
                  )}
                  <Tabs value={detailTab} onValueChange={setDetailTab} className="inspector-tabs">
                    <TabsList aria-label="详情类型">
                      <TabsTrigger value="step">步骤</TabsTrigger>
                      <TabsTrigger value="evidence">
                        证据<span className="tab-count">{sources.length}</span>
                      </TabsTrigger>
                      <TabsTrigger value="run">运行</TabsTrigger>
                    </TabsList>
                    <TabsContent value="step">
                      <AnimatePresence mode="wait">
                        <motion.div
                          key={selected ? 'step' : 'empty'}
                          initial={{ opacity: 0, x: 4 }}
                          animate={{ opacity: 1, x: 0 }}
                          exit={{ opacity: 0 }}
                          transition={{ duration: 0.13 }}
                        >
                          {selected ? (
                            <div className="step-detail">
                              <div className={`detail-icon tone-${outcomeTone(selected.outcome)}`}>
                                {['warning', 'danger'].includes(outcomeTone(selected.outcome)) ? (
                                  <AlertCircle size={20} />
                                ) : selected.node === 'search' ? (
                                  <Search size={20} />
                                ) : selected.node === 'read' ? (
                                  <BookOpen size={20} />
                                ) : selected.node === 'finish' ? (
                                  <ShieldCheck size={20} />
                                ) : (
                                  <Workflow size={20} />
                                )}
                              </div>
                              <div className="section-kicker">调用 ID · {selected.id}</div>
                              <h2>
                                {actionNames[selected.action] ?? selected.action} · 第{' '}
                                {invocationOrdinal(calls, selected)} 次
                              </h2>
                              <nav className="invocation-navigation" aria-label="同类调用导航">
                                <div className="invocation-navigation-heading">
                                  <span>同类调用</span>
                                  <Button
                                    variant="ghost"
                                    size="sm"
                                    disabled={
                                      !latestActionCall || latestActionCall.id === selected.id
                                    }
                                    onClick={() => latestActionCall && inspect(latestActionCall)}
                                    aria-label="跳到最新同类调用"
                                  >
                                    跳到最新 <ArrowRight size={13} />
                                  </Button>
                                </div>
                                <div className="invocation-navigation-controls">
                                  <Button
                                    variant="outline"
                                    size="sm"
                                    disabled={!previousActionCall}
                                    onClick={() =>
                                      previousActionCall && inspect(previousActionCall)
                                    }
                                    aria-label="上一次同类调用"
                                  >
                                    <ChevronLeft size={14} /> 上一次
                                  </Button>
                                  <span className="invocation-position mono" aria-live="polite">
                                    第 {selectedActionIndex + 1} / {sameActionCalls.length} 次
                                  </span>
                                  <Button
                                    variant="outline"
                                    size="sm"
                                    disabled={!nextActionCall}
                                    onClick={() => nextActionCall && inspect(nextActionCall)}
                                    aria-label="下一次同类调用"
                                  >
                                    下一次 <ChevronRight size={14} />
                                  </Button>
                                </div>
                                <p>按同类动作浏览；完整执行顺序见时间线。</p>
                              </nav>
                              {latestForSelectedNode &&
                                latestForSelectedNode.id !== selected.id && (
                                  <div className="historical-call-note">
                                    <p>
                                      正在查看较早调用；图中“正在查看”与此处对应，“最近”表示最新调用。
                                    </p>
                                  </div>
                                )}
                              <p className="detail-description">
                                {selected.node === 'agent'
                                  ? '根据问题与已有工具结果，选择下一项动作。'
                                  : selected.node === 'search'
                                    ? '从知识库中寻找相关候选段落。候选内容需要读取后才能用于引用。'
                                    : selected.node === 'read'
                                      ? '读取选中段落的完整内容，将它加入本次调查的证据。'
                                      : '检查答案结构与引用，或以明确状态结束调查。'}
                              </p>
                              <dl className="detail-properties">
                                <div>
                                  <dt>调用状态</dt>
                                  <dd>
                                    {selected.end !== undefined
                                      ? '已返回'
                                      : current?.state === 'running'
                                        ? '执行中'
                                        : '未完成'}
                                  </dd>
                                </div>
                                <div>
                                  <dt>模型轮次</dt>
                                  <dd className="mono">{selected.step}</dd>
                                </div>
                                <div>
                                  <dt>耗时</dt>
                                  <dd className="mono">
                                    {selected.start !== undefined && selected.end !== undefined
                                      ? seconds(selected.end - selected.start)
                                      : '—'}
                                  </dd>
                                </div>
                                {selected.end !== undefined && (
                                  <div>
                                    <dt>运行后时间</dt>
                                    <dd className="mono">+{seconds(selected.end)}</dd>
                                  </div>
                                )}
                              </dl>
                              {selected.outcome && (
                                <div className="outcome-code">
                                  <span>返回码</span>
                                  <code>{selected.outcome}</code>
                                </div>
                              )}
                              {selected.evidenceIds.length > 0 && (
                                <div className="step-evidence">
                                  <h3>{selected.node === 'search' ? '候选编号' : '关联证据'}</h3>
                                  {selected.evidenceIds.map((id) => (
                                    <button
                                      key={id}
                                      disabled={!sources.some((s) => s.citation_id === id)}
                                      onClick={() => cite(id)}
                                    >
                                      <FileText size={13} />
                                      <span>段落 {id}</span>
                                      <ArrowRight size={12} />
                                    </button>
                                  ))}
                                  {!current?.result && (
                                    <p className="muted text-xs">完整证据随最终结果返回。</p>
                                  )}
                                </div>
                              )}
                            </div>
                          ) : (
                            <div className="inspector-empty">
                              <div className="inspector-empty-icon">
                                <Layers2 size={23} />
                              </div>
                              <h3>把过程展开来看</h3>
                              <p>
                                选择流程节点或时间线中的一次调用，查看状态、返回结果与关联证据。
                              </p>
                              <span className="empty-micro">
                                <span />
                                从全局，到细节
                              </span>
                            </div>
                          )}
                        </motion.div>
                      </AnimatePresence>
                    </TabsContent>
                    <TabsContent value="evidence">
                      {sources.length ? (
                        <div className="evidence-panel">
                          <div className="evidence-list">
                            {sources.map((source) => (
                              <button
                                key={source.citation_id}
                                className={sourceId === source.citation_id ? 'selected' : ''}
                                aria-pressed={sourceId === source.citation_id}
                                onClick={() => cite(source.citation_id)}
                              >
                                <span className="citation-square">{source.citation_id}</span>
                                <span>{source.title || '未命名文档'}</span>
                                <ChevronRight size={13} />
                              </button>
                            ))}
                          </div>
                          {evidence ? (
                            <div>
                              <Button
                                className="expand-evidence"
                                variant="ghost"
                                size="sm"
                                onClick={openEvidence}
                              >
                                展开阅读 <ExternalLink size={13} />
                              </Button>
                              <Evidence source={evidence} />
                            </div>
                          ) : (
                            <p className="evidence-prompt">选择一份证据，阅读完整段落。</p>
                          )}
                        </div>
                      ) : (
                        <div className="inspector-empty">
                          <BookOpen size={24} />
                          <h3>证据会在这里汇合</h3>
                          <p>仅展示本次实际读取的段落。搜索候选不等于已引用证据。</p>
                        </div>
                      )}
                    </TabsContent>
                    <TabsContent value="run">
                      {current ? (
                        <div className="run-detail">
                          <div className="section-kicker">RUN DETAILS</div>
                          <h2>本次运行</h2>
                          <dl className="detail-properties">
                            {!current.demo && current.modelName && (
                              <div>
                                <dt>调查模型</dt>
                                <dd>{current.modelName}</dd>
                              </div>
                            )}
                            <div>
                              <dt>数据来源</dt>
                              <dd>{current.demo ? '原创合成演示' : '已连接服务'}</dd>
                            </div>
                            <div>
                              <dt>记录方式</dt>
                              <dd>
                                {current.transport === 'demo'
                                  ? '模拟事件'
                                  : current.transport === 'stream'
                                    ? '实时事件流'
                                    : '完成后返回'}
                              </dd>
                            </div>
                            <div>
                              <dt>总耗时</dt>
                              <dd className="mono">{seconds(elapsed)}</dd>
                            </div>
                            {current.result && (
                              <>
                                <div>
                                  <dt>模型调用</dt>
                                  <dd>{current.result.usage.model_calls}</dd>
                                </div>
                                <div>
                                  <dt>搜索 / 读取</dt>
                                  <dd>
                                    {current.result.usage.search_calls} /{' '}
                                    {current.result.usage.read_calls}
                                  </dd>
                                </div>
                                {!current.demo && (
                                  <div>
                                    <dt>估算输入 token</dt>
                                    <dd>{current.result.usage.prompt_estimated_tokens}</dd>
                                  </div>
                                )}
                              </>
                            )}
                          </dl>
                          <div className="profile-detail">
                            <span>运行编号</span>
                            <code>{current.progress[0]?.run_id ?? current.id}</code>
                            {current.result && (
                              <>
                                <span>Agent profile</span>
                                <code>{current.result.agent_profile}</code>
                              </>
                            )}
                          </div>
                          <p className="muted text-xs">
                            {current.demo
                              ? '演示耗时与事件来自脚本，不代表模型性能。'
                              : '计数与 token 为运行诊断信息，不等于供应商账单。'}
                          </p>
                        </div>
                      ) : (
                        <div className="inspector-empty">
                          <Terminal size={24} />
                          <h3>等待创建运行</h3>
                          <p>发起调查后，可查看数据来源、调用次数与配置标识。</p>
                        </div>
                      )}
                    </TabsContent>
                  </Tabs>
                  <div className="inspector-footer">
                    <ShieldCheck size={13} />
                    <span>执行记录与证据 · 不含内部思考</span>
                  </div>
                </Inspector>
              </div>
              <footer className="workspace-footer">
                <span>
                  <Brand small /> 让答案有依据，让过程可解释。
                </span>
                <span>
                  {mode === 'demo' ? 'DEMO / SYNTHETIC DATA' : 'LOCAL / CONNECTED SERVICE'}
                  <span className="footer-divider" />
                  RAG Workbench
                </span>
              </footer>
            </main>
          </div>
        </div>
        <Dialog
          title="引用证据"
          description="本次已读取的完整段落；关闭后返回原阅读位置。"
          open={evidenceOpen}
          onOpenChange={setEvidenceOpen}
          className="evidence-dialog"
          onCloseAutoFocus={(event) => {
            event.preventDefault();
            evidenceOrigin.current?.focus({ preventScroll: true });
          }}
        >
          {evidence && <Evidence source={evidence} />}
        </Dialog>
      </TooltipProvider>
    </MotionConfig>
  );
}
