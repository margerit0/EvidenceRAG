import { memo, useEffect, useMemo, useRef, useState } from 'react';
import {
  Background,
  BackgroundVariant,
  Controls,
  Handle,
  MarkerType,
  Position,
  ReactFlow,
  type Node,
  type NodeProps,
} from '@xyflow/react';
import {
  ArrowUpRight,
  AlertCircle,
  Check,
  CircleDashed,
  FileCheck2,
  FileText,
  Loader2,
  Search,
  Workflow,
} from 'lucide-react';
import {
  actionNames,
  invocationOrdinal,
  invocations,
  outcomeTone,
  type Invocation,
  type NodeKey,
  type Run,
} from '../lib/model';

type GraphData = {
  label: string;
  code: string;
  kind: NodeKey;
  status: string;
  visits: number;
  active: boolean;
  inspection?: string;
};
type FlowNode = Node<GraphData>;
const icons = { agent: Workflow, search: Search, read: FileText, finish: FileCheck2 };
const AgentNode = memo(function AgentNode({ data, selected }: NodeProps<FlowNode>) {
  const Icon = icons[data.kind];
  return (
    <div className={`flow-node ${data.status} ${selected ? 'is-selected' : ''}`}>
      <Handle type="target" position={Position.Top} id="in" />
      <Handle type="target" position={Position.Left} id="return-left" />
      <Handle type="target" position={Position.Right} id="return-right" />
      <div className="node-heading">
        <span className="node-icon">
          <Icon size={17} />
        </span>
        <span>{data.label}</span>
        {data.active ? (
          <Loader2 size={13} className="spin node-status" />
        ) : data.status === 'done' ? (
          <Check size={13} className="node-status" />
        ) : ['failed', 'warning'].includes(data.status) ? (
          <AlertCircle size={13} className="node-status" />
        ) : (
          <CircleDashed size={13} className="node-status" />
        )}
      </div>
      <div className="node-meta">
        <span>{data.code}</span>
        <span>
          {data.active
            ? '执行中'
            : data.visits
              ? ['search', 'read'].includes(data.kind)
                ? `最近第 ${data.visits} 次`
                : `${data.visits} 次调用`
              : '等待'}
        </span>
      </div>
      {data.inspection && <div className="node-inspection">正在查看{data.inspection}</div>}
      <Handle type="source" position={Position.Bottom} id="out" />
      <Handle type="source" position={Position.Left} id="loop-left" />
      <Handle type="source" position={Position.Right} id="loop-right" />
    </div>
  );
});
const nodeTypes = { stage: AgentNode };
const stages: { id: NodeKey; label: string; code: string; x: number; y: number }[] = [
  { id: 'agent', label: 'Agent 决策', code: 'DECIDE', x: 154, y: 20 },
  { id: 'search', label: '搜索文档', code: 'SEARCH', x: 0, y: 163 },
  { id: 'read', label: '读取证据', code: 'READ', x: 308, y: 163 },
  { id: 'finish', label: '校验与输出', code: 'FINISH', x: 154, y: 309 },
];
const definitions = [
  {
    id: 'agent-search',
    source: 'agent',
    target: 'search',
    sourceHandle: 'out',
    targetHandle: 'in',
  },
  { id: 'agent-read', source: 'agent', target: 'read', sourceHandle: 'out', targetHandle: 'in' },
  {
    id: 'search-agent',
    source: 'search',
    target: 'agent',
    sourceHandle: 'loop-left',
    targetHandle: 'return-left',
  },
  {
    id: 'read-agent',
    source: 'read',
    target: 'agent',
    sourceHandle: 'loop-right',
    targetHandle: 'return-right',
  },
  {
    id: 'agent-finish',
    source: 'agent',
    target: 'finish',
    sourceHandle: 'out',
    targetHandle: 'in',
  },
];
export function ExecutionGraph({
  run,
  selected,
  onSelect,
  theme,
}: {
  run?: Run;
  selected?: string;
  onSelect: (invocation: Invocation) => void;
  theme: 'light' | 'dark';
}) {
  const container = useRef<HTMLDivElement>(null);
  const [compact, setCompact] = useState(false);
  useEffect(() => {
    if (!container.current) return;
    const observer = new ResizeObserver(([entry]) => setCompact(entry.contentRect.width < 400));
    observer.observe(container.current);
    return () => observer.disconnect();
  }, []);
  const calls = useMemo(() => invocations(run?.progress ?? []), [run?.progress]);
  const latest = calls.at(-1);
  const active = run?.state === 'running' && latest?.end === undefined ? latest?.node : undefined;
  const nodes: FlowNode[] = stages.map((stage, index) => {
    const visits = calls.filter((c) => c.node === stage.id);
    const last = visits.at(-1);
    const inspected = visits.find((call) => call.id === selected);
    const inspection = inspected
      ? `${['search', 'read'].includes(stage.id) ? '' : `${actionNames[inspected.action] ?? inspected.action} · `}第 ${invocationOrdinal(calls, inspected)} 次`
      : undefined;
    const tone = outcomeTone(last?.outcome);
    return {
      id: stage.id,
      type: 'stage',
      position: compact ? { x: 55, y: 20 + index * 130 } : { x: stage.x, y: stage.y },
      selected: !!inspected,
      data: {
        label: stage.label,
        code: stage.code,
        kind: stage.id,
        visits: visits.length,
        inspection,
        active: active === stage.id,
        status:
          active === stage.id
            ? 'running'
            : tone === 'danger'
              ? 'failed'
              : tone === 'warning'
                ? 'warning'
                : last?.end !== undefined
                  ? 'done'
                  : 'waiting',
      },
      ariaLabel: `${stage.label}，${active === stage.id ? '执行中' : visits.length ? `已调用 ${visits.length} 次` : '等待执行'}${inspection ? `，正在查看${inspection}` : ''}。点击查看最近调用。`,
    };
  });
  const edges = definitions.map((edge) => {
    const traversed = calls.some(
      (call, index) =>
        index > 0 && calls[index - 1].node === edge.source && call.node === edge.target,
    );
    const flowing =
      !!active &&
      edge.target === active &&
      calls.length > 1 &&
      calls[calls.length - 2].node === edge.source;
    return {
      ...edge,
      ...(compact && ['agent-read', 'agent-finish'].includes(edge.id)
        ? { sourceHandle: 'loop-right', targetHandle: 'return-right' }
        : {}),
      ...(compact && edge.id === 'read-agent'
        ? { sourceHandle: 'loop-left', targetHandle: 'return-left' }
        : {}),
      type: 'default',
      animated: flowing,
      markerEnd: {
        type: MarkerType.ArrowClosed,
        width: 14,
        height: 14,
        color: flowing ? 'var(--brand)' : 'var(--graph-line)',
      },
      style: {
        stroke: flowing ? 'var(--brand)' : traversed ? 'var(--graph-visited)' : 'var(--graph-line)',
        strokeWidth: flowing ? 1.7 : 1.2,
        opacity: traversed || flowing ? 1 : 0.72,
      },
    };
  });
  return (
    <div
      ref={container}
      className={`graph-canvas ${compact ? 'is-compact' : ''}`}
      data-testid="execution-graph"
    >
      <ReactFlow
        key={compact ? 'compact' : 'wide'}
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        fitView
        fitViewOptions={{ padding: 0.23 }}
        minZoom={0.35}
        maxZoom={1.4}
        nodesDraggable={false}
        nodesConnectable={false}
        elementsSelectable
        zoomOnScroll={false}
        preventScrolling={false}
        colorMode={theme}
        onNodeClick={(_, node) => {
          const call = calls.filter((c) => c.node === node.id).at(-1);
          if (call) onSelect(call);
        }}
        ariaLabelConfig={{
          'controls.zoomIn.ariaLabel': '放大流程图',
          'controls.zoomOut.ariaLabel': '缩小流程图',
          'controls.fitView.ariaLabel': '适应画布',
        }}
      >
        <Background variant={BackgroundVariant.Dots} gap={20} size={1} color="var(--graph-dot)" />
        <Controls showInteractive={false} position="bottom-left" />
      </ReactFlow>
      <span className="graph-caption">
        <ArrowUpRight size={12} />
        工具结果返回 Agent，继续决策
      </span>
    </div>
  );
}
