// Replay the frontend's actual SVG/CSS effects against the integer movie clock.
// CSS wall-clock animations remain disabled; paused Web Animations carry the
// native keyframes, durations and easing, so every frame can be sought directly.
const GRAPH_BEATS = [
  [0, 'question'], [45, 'agentStart'], [60, 'search1'], [75, 'read1'],
  [90, 'return1'], [105, 'investigation'], [120, 'read2'],
  [135, 'validate'], [150, 'answer'],
];
const NATIVE_GRAPH_TIMING = {
  signal: 1400, spinner: 1300, node: 220, edge: 200, status: 180, selection: 180,
};
let nativeGraphStyles;
let pausedGraphAnimations = [];

const graphStyle = (element, properties) => {
  const computed = getComputedStyle(element);
  return Object.fromEntries(properties.map(property => [property, computed[property]]));
};

function prepareNativeGraphMotion(doc) {
  nativeGraphStyles = Object.fromEntries(GRAPH_BEATS.map(([, key]) => {
    restoreGraph(doc, key);
    const nodes = Object.fromEntries([...doc.querySelectorAll('.react-flow__node-stage')].map(wrapper => {
      const node = wrapper.querySelector('.flow-node');
      return [wrapper.dataset.id, {
        box: graphStyle(node, ['backgroundColor', 'borderColor', 'boxShadow']),
        icon: graphStyle(node.querySelector('.node-icon'), ['color', 'backgroundColor', 'borderColor']),
      }];
    }));
    const edges = Object.fromEntries([...doc.querySelectorAll('.react-flow__edge')].map(edge => [
      edge.dataset.id, graphStyle(edge.querySelector('.react-flow__edge-path'), ['stroke', 'strokeWidth', 'opacity']),
    ]));
    return [key, {nodes, edges}];
  }));
  doc.querySelectorAll('.react-flow__edge').forEach(edge => {
    // This is the second path in ExecutionEdge, absent from a completed-run DOM snapshot.
    const signal = doc.createElementNS('http://www.w3.org/2000/svg', 'path');
    signal.setAttribute('d', edge.querySelector('.react-flow__edge-path').getAttribute('d'));
    signal.setAttribute('class', 'graph-edge-signal');
    signal.setAttribute('pathLength', '100');
    signal.setAttribute('fill', 'none');
    signal.style.display = 'none';
    edge.append(signal);
    const path = edge.querySelector('.react-flow__edge-path');
    const markerId = path.getAttribute('marker-end')?.match(/^url\(["']?#(.+?)["']?\)$/)?.[1];
    const marker = doc.getElementById(markerId);
    for (const [kind, color] of [['flowing', 'var(--brand)'], ['traversed', 'var(--graph-visited)'], ['waiting', 'var(--graph-line)']]) {
      const copy = marker.cloneNode(true);
      copy.id = `movie-arrow-${edge.dataset.id}-${kind}`;
      copy.querySelectorAll('path,polyline,polygon').forEach(shape => {
        shape.style.stroke = color; shape.style.fill = color;
      });
      marker.parentElement.append(copy);
    }
  });
  const css = doc.createElement('style');
  css.textContent = '.flow-node::after{opacity:var(--movie-selection-opacity,0)!important}';
  doc.head.append(css);
}

function seekNativeAnimation(element, keyframes, duration, age, {loop = false, easing = 'ease-out'} = {}) {
  const animation = element.animate(keyframes, {duration, easing, iterations: loop ? Infinity : 1, fill: 'both'});
  animation.pause();
  animation.currentTime = age;
  pausedGraphAnimations.push(animation);
}

function nativeEaseOut(t) {
  // CSS ease-out = cubic-bezier(0, 0, .58, 1).
  t = Math.min(1, Math.max(0, t));
  let lo = 0, hi = 1;
  for (let i = 0; i < 18; i++) {
    const u = (lo + hi) / 2, x = 3 * .58 * (1 - u) * u * u + u * u * u;
    if (x < t) lo = u; else hi = u;
  }
  const u = (lo + hi) / 2;
  return 3 * (1 - u) * u * u + u * u * u;
}

function animateNativeGraph(doc, frame) {
  for (const animation of pausedGraphAnimations) animation.cancel();
  pausedGraphAnimations = [];
  const index = GRAPH_BEATS.findLastIndex(([start]) => frame >= start);
  const [start, key] = GRAPH_BEATS[index], priorKey = GRAPH_BEATS[Math.max(0, index - 1)][1];
  const state = D.demo.graphStates[key], previous = D.demo.graphStates[priorKey];
  const age = (frame - start) * 1000 / 30;
  const styles = nativeGraphStyles[key], oldStyles = nativeGraphStyles[priorKey];
  doc.querySelectorAll('.react-flow__node-stage').forEach(wrapper => {
    const id = wrapper.dataset.id, node = wrapper.querySelector('.flow-node');
    if (index && age < NATIVE_GRAPH_TIMING.node && state.nodes[id].status !== previous.nodes[id].status) {
      seekNativeAnimation(node, [oldStyles.nodes[id].box, styles.nodes[id].box], NATIVE_GRAPH_TIMING.node, age);
      seekNativeAnimation(node.querySelector('.node-icon'), [oldStyles.nodes[id].icon, styles.nodes[id].icon], NATIVE_GRAPH_TIMING.node, age);
    }
    let changed = 0;
    for (let i = 1; i <= index; i++) {
      if (D.demo.graphStates[GRAPH_BEATS[i][1]].nodes[id].status !== D.demo.graphStates[GRAPH_BEATS[i - 1][1]].nodes[id].status) changed = GRAPH_BEATS[i][0];
    }
    const statusAge = (frame - changed) * 1000 / 30;
    if (changed && statusAge < NATIVE_GRAPH_TIMING.status) {
      seekNativeAnimation(node.querySelector('.node-status'), [{opacity: .4, transform: 'translateY(2px)'}, {opacity: 1, transform: 'translateY(0px)'}], NATIVE_GRAPH_TIMING.status, statusAge);
    }
    if (state.nodes[id].status === 'running') {
      const spinner = node.querySelector('.node-status svg');
      spinner.classList.add('spin');
      seekNativeAnimation(spinner, [{transform: 'rotate(0deg)'}, {transform: 'rotate(360deg)'}], NATIVE_GRAPH_TIMING.spinner, statusAge, {loop: true, easing: 'linear'});
    }
    const selection = state.nodes[id].inspection ? .8 * nativeEaseOut((frame - 135) * 1000 / 30 / NATIVE_GRAPH_TIMING.selection) : 0;
    node.style.setProperty('--movie-selection-opacity', String(selection));
  });
  doc.querySelectorAll('.react-flow__edge').forEach(edge => {
    const id = edge.dataset.id;
    const markerKind = state.flowing === id ? 'flowing' : state.traversed.includes(id) ? 'traversed' : 'waiting';
    edge.querySelector('.react-flow__edge-path').setAttribute('marker-end', `url(#movie-arrow-${id}-${markerKind})`);
    if (index && age < NATIVE_GRAPH_TIMING.edge) {
      seekNativeAnimation(edge.querySelector('.react-flow__edge-path'), [oldStyles.edges[id], styles.edges[id]], NATIVE_GRAPH_TIMING.edge, age);
    }
    const signal = edge.querySelector('.graph-edge-signal');
    signal.style.display = state.flowing === id ? '' : 'none';
    if (state.flowing === id) {
      seekNativeAnimation(signal, [{strokeDashoffset: '100'}, {strokeDashoffset: '0'}], NATIVE_GRAPH_TIMING.signal, age, {loop: true, easing: 'linear'});
    }
  });
  window.graphMotionState = {state: key, startFrame: start, ageMs: age, activeNode: state.active, flowingEdge: state.flowing};
}
