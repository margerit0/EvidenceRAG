// Project adapter for the resumable-investigation promo: product selectors,
// camera regions, overlay design and how recorded backend snapshots are paced.
// Generic frame stepping and capture live in render.mjs.

export const view = { width: 1776, height: 999 };
export const output = { width: 1920, height: 1080 };

// Click targets resolve against the real DOM at render time.
export const targets = {
  'mode-trigger': { selector: 'button.select-trigger[aria-label="运行模式"]' },
  'mode-live': { selector: '.select-item', text: '连接服务' },
  query: { selector: '#query', at: [0.28, 0.42] },
  start: { selector: 'form.composer button[type="submit"]' },
  supplement: { selector: '.investigation-controls textarea', at: [0.3, 0.45] },
  'submit-supplement': { selector: '.investigation-controls button[type="submit"]' },
  pause: { selector: '.run-control-buttons button', text: '暂停调查' },
  resume: { selector: '.run-control-buttons button', text: '继续调查' },
  'citation-2': { selector: '.inline-citations button[aria-label="查看引用 2"]' },
};

// Frames the cursor travels before a click, then rests on the target. After
// some clicks it steps aside to an empty spot (resolved against the live
// layout) so it does not cover what the viewer reads next. After the second
// resume it waits at the empty end of the status row: the answer appears
// below it, never under it.
const belowControls = { selector: '.investigation-controls', at: [0.62, 1], offset: [0, 34] };
export const motion = {
  travel: 16,
  dwell: 5,
  rest: { x: 1560, y: 640 },
  park: {
    'pick-live': { selector: '.page-heading', at: [0.55, 0.55] },
    'start-run': { selector: '.composer', at: [0.6, 0.3] },
    'submit-supplement': belowControls,
    pause: belowControls,
    'resume-1': belowControls,
    'resume-2': { selector: '.answer-status', at: [0.8, 0.5] },
    'open-citation': { selector: '.answer-tools', at: [0.5, 0.5] },
  },
  parkDelay: 6,
  parkFrames: 14,
};
// Bounding box of the cursor arrow from its tip, in overlay design pixels
// (the 28-unit path drawn 30 px wide); renders log any text it covers.
export const cursorArrow = { width: 17, height: 24 };

// Expected visible results, checked after each click (frame offset, selector, text).
export const expectations = {
  'open-mode': [6, '[role="option"]', '连接服务'],
  'pick-live': [12, '.connection-note', '服务已连接'],
  'start-run': [6, '.submitted-query', '星桥集群是否可以升级？'],
  'submit-supplement': [6, '.supplement-history summary', '已提交的补充信息（1）'],
  pause: [6, '.answer-status', '正在暂停'],
  'resume-1': [4, '.answer-status', '正在恢复'],
  'resume-2': [4, '.answer-status', '正在恢复'],
  'open-citation': [20, '.evidence-detail h3', '回退演练'],
};

// Copy a viewer reads at representative frames, logged with each element's box
// and visibility for an exact comparison with the recorded synthetic backend
// (see verify_final.py).
const answer = { selector: '.answer-body > p:not(.answer-caution)', exclude: '.inline-citations' };
const citations = { selector: '.inline-citations button', attr: 'aria-label' };
const status = { selector: '.answer-status' };
const controls = { selector: '.investigation-controls p' };
const history = { selector: '.supplement-history summary' };
const film = { caption: { selector: '#film-overlay .film-caption' }, label: { selector: '#film-overlay .film-label' } };
export const probes = {
  171: { ...film, mode: { selector: 'button.select-trigger[aria-label="运行模式"]' }, connection: { selector: '.connection-note' } },
  240: { ...film, query: { selector: '#query' } },
  405: {
    ...film,
    status,
    question: { selector: '.investigation-controls label' },
    placeholder: { selector: '.investigation-controls textarea', attr: 'placeholder' },
  },
  480: { ...film, supplement: { selector: '.investigation-controls textarea' } },
  585: { ...film, status, controls, history },
  660: { ...film, status, history, buttons: { selector: '.run-control-buttons button' } },
  819: { ...film, status, controls, history },
  903: { ...film, status, controls, history },
  1059: { ...film, status, answer, citations },
  1170: {
    ...film,
    query: { selector: '.submitted-query' },
    answer,
    citations,
    pressed: { selector: '.inline-citations button[aria-pressed="true"]', attr: 'aria-label' },
    caution: { selector: '.answer-caution' },
    evidenceKicker: { selector: '.evidence-detail .section-kicker' },
    evidenceTitle: { selector: '.evidence-detail h3' },
    evidenceText: { selector: '.evidence-detail .evidence-text' },
  },
  1272: {
    ...film,
    brandName: { selector: '#film-overlay .film-brand-name' },
    brandSub: { selector: '#film-overlay .film-brand-sub' },
    tagline: { selector: '#film-overlay .film-brand-tagline' },
  },
};

// Named camera regions in product CSS pixels, fitted to 16:9 and clamped.
export const regions = {
  full: () => [{ x: 0, y: 0, width: view.width, height: view.height }],
  mode: () => [{ selector: '.mode-control', pad: [36, 300, 140, 300] }],
  conversation: () => [{ selector: '.conversation-pane', pad: [22, 22, 22, 22] }],
  execution: () => [{ selector: '.execution-pane', pad: [22, 22, 22, 22] }],
  work: () => [
    { selector: '.conversation-pane', pad: [18, 18, 18, 18] },
    { selector: '.execution-pane', pad: [18, 18, 18, 18] },
  ],
  controls: () => [
    { selector: '.submitted-query', pad: [40, 30, 30, 30] },
    { selector: '.investigation-controls', pad: [30, 30, 46, 30] },
  ],
};
export const minCameraWidth = 960;

// A local framing may crop the periphery, but a top or bottom frame edge that
// slices a line of text or an icon reads as a mistake. Each region's camera is
// nudged vertically (up to maxShift CSS px) or widened by one of the `grow`
// factors so both edges fall in gaps at least `margin` px from any ink, while
// the region's elements (unpadded) stay `coreMargin` px inside. Every cut
// costs `cutCost`; otherwise the smallest move wins, 1% wider ≈ growCost / 100 px.
export const cameraClearance = {
  margin: 2,
  coreMargin: 4,
  maxShift: 48,
  step: 0.5,
  grow: [1, 1.02, 1.04, 1.06, 1.08],
  cutCost: 1000,
  growCost: 600,
};

// Recorded snapshots are grouped by the command that preceded them. Each
// phase ends at a timeline event; snapshots in between keep their recorded
// order and roughly their recorded proportions, with a readable minimum gap.
export const pacing = {
  minGapMs: 250,
  latency: { get: 0, post: 100, refused: 0 },
  phases: {
    create: { endEvent: 'waiting-input', endStatus: 'waiting_input' },
    supplement: { endBeforeEvent: 'pause', leadMs: 500 },
    pause: { endEvent: 'paused', endStatus: 'paused' },
    resume1: { endBeforeEvent: 'service-stop', leadMs: 400 },
    resume2: { endEvent: 'answered', endStatus: 'completed' },
  },
  // The stream closes on this event; the service is back before the client's
  // second retry (1.5 s cadence), so the third request lands on `reconnected`.
  outage: { phase: 'resume1', startEvent: 'service-stop', endBeforeEvent: 'reconnected', endLeadMs: 500 },
};

// Lucide `layers-2`, the brand mark the product draws with <Layers2>
// (ISC; docs/resumable-promo/licenses/Lucide-ISC.txt).
const layersIcon =
  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="m16.02 12 5.48 3.13a1 1 0 0 1 0 1.74L13 21.74a2 2 0 0 1-2 0l-8.5-4.87a1 1 0 0 1 0-1.74L7.98 12"/><path d="M13 13.74a2 2 0 0 1-2 0L2.5 8.87a1 1 0 0 1 0-1.74L11 2.26a2 2 0 0 1 2 0l8.5 4.87a1 1 0 0 1 0 1.74Z"/></svg>';

export const overlayHtml = `
<div data-part="caption" class="film-caption"></div>
<div data-part="ripple" class="film-ripple"></div>
<svg data-part="cursor" class="film-cursor" viewBox="0 0 28 28" aria-hidden="true">
  <path d="M6 3.5v19.2l5.1-4.6 3.5 7.6 3.6-1.6-3.5-7.5h6.9z" fill="#fff" stroke="#17242e" stroke-width="1.5" stroke-linejoin="round"/>
</svg>
<div data-part="brand" class="film-brand">
  <div class="film-brand-mark">${layersIcon}</div>
  <div class="film-brand-name">EvidenceRAG</div>
  <div class="film-brand-sub">可暂停、可恢复的中文文档调查</div>
  <div data-part="tagline" class="film-brand-tagline"></div>
</div>
<div data-part="label" class="film-label">合成演示 · 流程已压缩</div>`;

export const overlayCss = `
#film-overlay { position: fixed; left: 0; top: 0; width: ${view.width}px; height: ${view.height}px;
  transform-origin: 0 0; pointer-events: none; z-index: 2147483647; contain: layout style;
  font-family: 'Inter Variable', 'Segoe UI', 'Microsoft YaHei', sans-serif; color: #263845; }
#film-overlay *, #film-overlay *::before, #film-overlay *::after { box-sizing: border-box;
  transition: none !important; animation: none !important; }
.film-label { position: absolute; right: 20px; top: 6px; font-size: 13px; line-height: 20px;
  padding: 1px 11px; border-radius: 999px; color: #34505f; background: #ffffffc4;
  border: 1px solid #ffffffe0; box-shadow: 0 2px 10px #28425b14; letter-spacing: 0.02em; }
.film-caption { position: absolute; left: 50%; top: 46px; transform: translateX(-50%); opacity: 0;
  font-size: 20px; line-height: 28px; font-weight: 600; letter-spacing: 0.01em; white-space: nowrap;
  padding: 9px 22px; border-radius: 999px; color: #22394a; background: #f6fafdeb;
  border: 1px solid #ffffff; box-shadow: inset 0 1px 0 #fff, 0 10px 34px #12304a24; }
.film-cursor { position: absolute; left: 0; top: 0; width: 30px; height: 30px; transform-origin: 6px 3.5px;
  filter: drop-shadow(0 2px 3px #0b1a2440); opacity: 0; }
.film-ripple { position: absolute; left: 0; top: 0; width: 0; height: 0; border-radius: 50%;
  border: 2px solid #527b96; opacity: 0; }
.film-brand { position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center;
  justify-content: center; opacity: 0; color: #263845; }
.film-brand-mark { width: 78px; height: 78px; display: grid; place-items: center; color: #527b96;
  border-radius: 24px; border: 1px solid #ffffffd1; background: #ffffff52;
  box-shadow: inset 0 1px 0 #fffffff0, 0 10px 30px #355c761c; }
.film-brand-mark svg { width: 44px; height: 44px; }
.film-brand-name { margin-top: 26px; font-size: 60px; line-height: 1.1; font-weight: 650; letter-spacing: -1.5px; }
.film-brand-sub { margin-top: 12px; font-size: 20px; color: #415a6c; letter-spacing: 0.06em; }
.film-brand-tagline { margin-top: 34px; font-size: 27px; font-weight: 550; color: #2f4a5c; letter-spacing: 0.02em; }
`;

// Candidates offered with the storyboard; the first is the working default.
export const taglines = ['停下来，也能接着查。', '暂停或重启之后，从保存点继续。', '断线不丢进度，重启接着调查。'];
export const tagline = taglines[0];

const ease = (x) => (x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2);
const clamp01 = (x) => Math.min(1, Math.max(0, x));

// Per-frame overlay values in the overlay's 1600 x 900 design space.
export function overlayState({ frame, fps, shot, camera, cursor, click, plan }) {
  const local = frame - shot.startFrame;
  const left = shot.endFrameExclusive - frame;
  const caption = shot.data.caption || '';
  const captionOpacity = caption ? Math.min(clamp01((local - 3) / 9), clamp01(left / 7)) : 0;
  const scale = view.width / camera.width;
  const parts = {
    caption: {
      text: caption,
      opacity: String(captionOpacity),
      transform: `translateX(-50%) translateY(${(1 - ease(clamp01((local - 3) / 9))) * -6}px)`,
    },
    cursor: { opacity: '0' },
    ripple: { opacity: '0' },
    brand: { opacity: '0' },
    tagline: { text: tagline },
  };
  if (cursor.visible) {
    const x = (cursor.x - camera.x) * scale;
    const y = (cursor.y - camera.y) * scale;
    const press = click ? clamp01(1 - Math.abs(frame - click.frame) / 3) : 0;
    parts.cursor = {
      opacity: String(cursor.opacity),
      transform: `translate(${x - 6}px, ${y - 3.5}px) scale(${1 - 0.12 * press})`,
    };
    if (click && frame >= click.frame && frame - click.frame < 14) {
      const p = (frame - click.frame) / 13;
      const radius = 7 + 24 * ease(p);
      const cx = (click.x - camera.x) * scale;
      const cy = (click.y - camera.y) * scale;
      parts.ripple = {
        opacity: String(0.55 * (1 - p)),
        width: `${radius * 2}px`,
        height: `${radius * 2}px`,
        transform: `translate(${cx - radius}px, ${cy - radius}px)`,
      };
    }
  }
  if (shot.id === 'brand') {
    const p = ease(clamp01((local - 8) / 22));
    parts.brand = { opacity: String(p), transform: `translateY(${(1 - p) * 10}px)` };
  }
  void fps;
  void plan;
  return parts;
}

// Film-only treatment of the product window during the brand shot.
export function shellStyle({ frame, shot }) {
  if (shot.id !== 'brand') return { opacity: '', transform: '', filter: '' };
  const p = ease(clamp01((frame - shot.startFrame) / 18));
  return {
    opacity: String(1 - p),
    transform: `scale(${1 - 0.02 * p})`,
    filter: `blur(${(4 * p).toFixed(2)}px)`,
  };
}
