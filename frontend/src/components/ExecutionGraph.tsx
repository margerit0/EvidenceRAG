import { memo, useLayoutEffect, useMemo, useRef, useState, type RefObject } from 'react';
import {
  Background,
  BackgroundVariant,
  BaseEdge,
  ControlButton,
  Controls,
  Handle,
  MarkerType,
  Position,
  ReactFlow,
  getBezierPath,
  getNodesBounds,
  getViewportForBounds,
  useReactFlow,
  type Node,
  type NodeProps,
  type Edge,
  type EdgeProps,
  type Rect,
} from '@xyflow/react';
import {
  ArrowUpRight,
  AlertCircle,
  Check,
  CircleDashed,
  FileCheck2,
  FileText,
  Loader2,
  Maximize,
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
        <span className="node-status" key={data.status}>
          {data.active ? (
            <Loader2 size={13} className="spin" />
          ) : data.status === 'done' ? (
            <Check size={13} />
          ) : ['failed', 'warning'].includes(data.status) ? (
            <AlertCircle size={13} />
          ) : (
            <CircleDashed size={13} />
          )}
        </span>
      </div>
      <div className="node-meta">
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
      <div className="node-inspection" aria-hidden={!data.inspection}>
        {data.inspection ? `正在查看${data.inspection}` : '\u00a0'}
      </div>
      <Handle type="source" position={Position.Bottom} id="out" />
      <Handle type="source" position={Position.Left} id="loop-left" />
      <Handle type="source" position={Position.Right} id="loop-right" />
    </div>
  );
});
const nodeTypes = { stage: AgentNode };
type SignalEdge = Edge<{ flowing: boolean }>;
const ExecutionEdge = memo(function ExecutionEdge(props: EdgeProps<SignalEdge>) {
  let [path] = getBezierPath(props);
  if (
    props.sourceX === props.targetX &&
    props.sourcePosition === props.targetPosition &&
    [Position.Left, Position.Right].includes(props.sourcePosition)
  ) {
    // Give compact tool/return routes separate lanes instead of drawing several
    // vertical paths on the node borders. These are still the same event edges.
    const direction = props.sourcePosition === Position.Right ? 1 : -1;
    const lane = ['agent-finish', 'read-agent'].includes(props.id) ? 68 : 42;
    const { sourceX: sx, sourceY: sy, targetX: tx, targetY: ty } = props;
    path = `M ${sx},${sy} C ${sx + lane * direction},${sy} ${tx + lane * direction},${ty} ${tx},${ty}`;
  }
  return (
    <>
      <BaseEdge id={props.id} path={path} style={props.style} markerEnd={props.markerEnd} />
      {props.data?.flowing && (
        <path d={path} className="graph-edge-signal" pathLength={100} fill="none" />
      )}
    </>
  );
});
const edgeTypes = { execution: ExecutionEdge };
const nodeWidth = 188;
const nodeHeight = 92;
const stages: { id: NodeKey; label: string; x: number; y: number }[] = [
  { id: 'agent', label: 'Agent 决策', x: 112, y: 0 },
  { id: 'search', label: '搜索文档', x: 0, y: 112 },
  { id: 'read', label: '读取证据', x: 224, y: 112 },
  { id: 'finish', label: '校验与输出', x: 112, y: 224 },
];
const fitOptions = { padding: 0.025, maxZoom: 1 };

function FitGraph({
  container,
  bounds,
}: {
  container: RefObject<HTMLDivElement | null>;
  bounds: Rect;
}) {
  const { viewportInitialized, setViewport } = useReactFlow();
  useLayoutEffect(() => {
    const canvas = container.current;
    if (!viewportInitialized || !canvas) return;
    const fit = (width: number, height: number) => {
      if (width && height) {
        // The layout transition is the only animation clock. Apply each observed
        // size before paint; a second viewport tween would trail or fight it.
        void setViewport(getViewportForBounds(bounds, width, height, 0.35, 1, fitOptions.padding));
      }
    };
    fit(canvas.clientWidth, canvas.clientHeight);
    const observer = new ResizeObserver(([entry]) =>
      fit(entry.contentRect.width, entry.contentRect.height),
    );
    observer.observe(canvas);
    return () => observer.disconnect();
  }, [viewportInitialized, container, bounds, setViewport]);
  return (
    <Controls showInteractive={false} showFitView={false} position="bottom-left">
      <ControlButton
        aria-label="适应画布"
        title="适应画布"
        onClick={() => {
          const canvas = container.current;
          if (canvas)
            void setViewport(
              getViewportForBounds(
                bounds,
                canvas.clientWidth,
                canvas.clientHeight,
                0.35,
                1,
                fitOptions.padding,
              ),
            );
        }}
      >
        <Maximize size={12} />
      </ControlButton>
    </Controls>
  );
}
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
  useLayoutEffect(() => {
    if (!container.current) return;
    const observer = new ResizeObserver(([entry]) => setCompact(entry.contentRect.width < 380));
    observer.observe(container.current);
    return () => observer.disconnect();
  }, []);
  const layout = useMemo(
    () =>
      stages.map((stage, index) => ({
        id: stage.id,
        position: compact ? { x: 55, y: index * 112 } : { x: stage.x, y: stage.y },
        width: nodeWidth,
        height: nodeHeight,
        data: {},
      })),
    [compact],
  );
  const bounds = useMemo(() => {
    const rect = getNodesBounds(layout);
    // Include return curves and inspection outlines, not just node rectangles.
    const side = compact ? 56 : 32;
    return {
      x: rect.x - side,
      y: rect.y - 6,
      // Leave room for the fixed inspector handle at the phone's right edge.
      width: rect.width + side * 2 + (compact ? 28 : 0),
      height: rect.height + 12,
    };
  }, [layout, compact]);
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
      // Fixed dimensions match the reserved title/status/selection rows.
      // Supplying them also lets viewport fitting run without a measurement round trip.
      width: nodeWidth,
      height: nodeHeight,
      position: layout[index].position,
      selected: !!inspected,
      data: {
        label: stage.label,
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
      type: 'execution',
      className: `graph-edge ${flowing ? 'is-flowing' : traversed ? 'is-traversed' : ''}`,
      data: { flowing },
      markerEnd: {
        type: MarkerType.ArrowClosed,
        width: 14,
        height: 14,
        color: flowing ? 'var(--brand)' : traversed ? 'var(--graph-visited)' : 'var(--graph-line)',
      },
      style: {
        stroke: flowing ? 'var(--brand)' : traversed ? 'var(--graph-visited)' : 'var(--graph-line)',
        strokeWidth: flowing ? 1.6 : 1.15,
        opacity: flowing ? 0.55 : traversed ? 0.85 : 0.7,
      },
    };
  });
  return (
    <>
      <div
        ref={container}
        className={`graph-canvas ${compact ? 'is-compact' : ''}`}
        data-testid="execution-graph"
      >
        <ReactFlow
          nodes={nodes}
          edges={edges}
          nodeTypes={nodeTypes}
          edgeTypes={edgeTypes}
          fitViewOptions={fitOptions}
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
          <FitGraph container={container} bounds={bounds} />
        </ReactFlow>
      </div>
      <p className="graph-caption">
        <ArrowUpRight size={12} />
        工具结果返回 Agent，继续决策
      </p>
    </>
  );
}
