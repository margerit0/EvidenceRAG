// Deterministic renderer for the resumable-investigation promo.
//
// Loads the real production bundle of frontend/ in Chromium, injects
// film-runtime.js (virtual clock, recorded backend, animation seeking,
// overlay), drives real mouse/keyboard input at timeline frames and captures
// each frame with a 16:9 camera clip at exactly 1920 x 1080.
//
//   node render.mjs --plan ../plan.json --recording api-recording.json \
//     --out <frames-dir> [--keyframes | --clip <id> | --frames a-b] [--build]
import { createRequire } from 'node:module';
import { spawnSync } from 'node:child_process';
import crypto from 'node:crypto';
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import * as scene from './scene.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const repo = path.resolve(here, '../../../../..');
const frontend = path.join(repo, 'frontend');
const require = createRequire(path.join(frontend, 'package.json'));
const { chromium } = require('playwright-core');

function args() {
  const result = { build: false, keyframes: false, frames: null, clip: null, chrome: null };
  const list = process.argv.slice(2);
  for (let index = 0; index < list.length; index++) {
    const key = list[index];
    const next = () => list[++index];
    if (key === '--plan') result.plan = next();
    else if (key === '--recording') result.recording = next();
    else if (key === '--out') result.out = next();
    else if (key === '--app') result.app = next();
    else if (key === '--clip') result.clip = next();
    else if (key === '--frames') result.frames = next();
    else if (key === '--keyframes') result.keyframes = true;
    else if (key === '--build') result.build = true;
    else if (key === '--chrome') result.chrome = next();
    else if (key === '--list') result.list = next();
    else throw new Error(`unknown argument ${key}`);
  }
  for (const key of ['plan', 'recording', 'out', 'app']) {
    if (!result[key]) throw new Error(`--${key} is required`);
  }
  return result;
}

const readJson = (file) => JSON.parse(fs.readFileSync(file, 'utf8').replace(/^﻿/, ''));
const writeJson = (file, value) => {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, JSON.stringify(value, null, 2) + '\n', 'utf8');
};
const sha256 = (data) => crypto.createHash('sha256').update(data).digest('hex');

function buildApp(outDir) {
  // Production bundle of the committed frontend; the product tree is untouched.
  const vite = path.join(frontend, 'node_modules', 'vite', 'bin', 'vite.js');
  const result = spawnSync(process.execPath, [vite, 'build', '--outDir', outDir, '--emptyOutDir'], {
    cwd: frontend,
    encoding: 'utf8',
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  if (result.status !== 0) throw new Error(`vite build failed\n${result.stdout}\n${result.stderr}`);
  const head = spawnSync('git', ['rev-parse', 'HEAD'], { cwd: repo, encoding: 'utf8' }).stdout.trim();
  const dirty = spawnSync('git', ['status', '--porcelain', '--', 'frontend'], { cwd: repo, encoding: 'utf8' }).stdout.trim();
  writeJson(path.join(outDir, '..', 'app-build.json'), {
    gitHead: head,
    frontendDirty: dirty,
    builtAt: new Date().toISOString(),
    indexSha256: sha256(fs.readFileSync(path.join(outDir, 'index.html'))),
  });
}

function serve(root) {
  const types = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.css': 'text/css',
    '.svg': 'image/svg+xml', '.woff2': 'font/woff2', '.png': 'image/png', '.json': 'application/json' };
  const server = http.createServer((request, response) => {
    const url = new URL(request.url, 'http://local');
    let file = decodeURIComponent(url.pathname);
    if (!file.startsWith('/workbench/')) {
      response.writeHead(404).end();
      return;
    }
    file = file.slice('/workbench/'.length) || 'index.html';
    const target = path.resolve(root, file);
    if (!target.startsWith(path.resolve(root)) || !fs.existsSync(target) || fs.statSync(target).isDirectory()) {
      response.writeHead(404).end();
      return;
    }
    response.writeHead(200, { 'Content-Type': types[path.extname(target)] || 'application/octet-stream', 'Cache-Control': 'no-store' });
    fs.createReadStream(target).pipe(response);
  });
  return new Promise((resolve) => server.listen(0, '127.0.0.1', () => resolve(server)));
}

// ------------------------------------------------------------ backend pacing
function phases(recording, plan) {
  const fps = Number(plan.video.fps);
  const ms = (frame) => (frame * 1000) / fps;
  const eventFrame = (id) => {
    const event = plan.events.find((item) => item.id === id);
    if (!event) throw new Error(`timeline event ${id} missing`);
    return event.frame;
  };
  const clickFrame = { create: eventFrame('start-run'), supplement: eventFrame('submit-supplement'),
    pause: eventFrame('pause'), resume1: eventFrame('resume-1'), resume2: eventFrame('resume-2') };
  const snaps = recording.snapshots;
  const responses = [];
  snaps.forEach((item, index) => {
    if (item.label.endsWith('-response')) responses.push(index);
  });
  const names = [];
  let resumes = 0;
  for (const index of responses) {
    const action = snaps[index].label.replace(/-response$/, '');
    names.push(action === 'resume' ? `resume${++resumes}` : action);
  }
  const restartIndex = snaps.findIndex((item) => item.label === 'after-restart');
  const result = {};
  const latency = scene.pacing.latency;
  responses.forEach((responseIndex, order) => {
    const name = names[order];
    const config = scene.pacing.phases[name];
    let last = order + 1 < responses.length ? responses[order + 1] - 1 : snaps.length - 1;
    if (name === scene.pacing.outage.phase) last = restartIndex - 1;
    const members = [];
    for (let index = responseIndex + 1; index <= last; index++) {
      if (snaps[index].label !== 'after-restart') members.push(index);
    }
    const start = ms(clickFrame[name]) + latency.post;
    let end;
    if (config.endEvent) {
      end = ms(eventFrame(config.endEvent));
      const final = snaps[members.at(-1)].value.status;
      if (final !== config.endStatus) throw new Error(`${name} ends with ${final}, expected ${config.endStatus}`);
    } else {
      end = ms(eventFrame(config.endBeforeEvent)) - config.leadMs;
    }
    // Weight each gap by recorded duration, then enforce a readable minimum.
    const times = members.map((index) => snaps[index].t);
    const origin = snaps[responseIndex].t;
    const gaps = times.map((t, i) => Math.max(0.001, t - (i ? times[i - 1] : origin)));
    const total = gaps.reduce((a, b) => a + b, 0);
    const span = end - start;
    const min = scene.pacing.minGapMs;
    if (span < min * gaps.length) throw new Error(`${name}: ${gaps.length} steps do not fit in ${span} ms`);
    const free = span - min * gaps.length;
    const steps = [];
    let at = 0;
    gaps.forEach((gap, i) => {
      at += min + (free * gap) / total;
      steps.push([Math.round(at * 1000) / 1000, members[i]]);
    });
    // Land the closing state exactly on its timeline event.
    steps[steps.length - 1][0] = Math.round((end - start) * 1000) / 1000;
    result[name] = { response: responseIndex, steps };
  });
  const outage = scene.pacing.outage;
  const outagePhaseStart = ms(clickFrame[outage.phase]) + latency.post;
  result[outage.phase].outage = {
    start: ms(eventFrame(outage.startEvent)) - outagePhaseStart,
    end: ms(eventFrame(outage.endBeforeEvent)) - outage.endLeadMs - outagePhaseStart,
    restored: restartIndex,
  };
  return { latency, phases: result };
}

// ------------------------------------------------------------ camera
const ease = (x) => (x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2);
function snapCamera(camera) {
  const { width: W, height: H } = scene.view;
  const { width: OW, height: OH } = scene.output;
  let best = null;
  for (let distance = 0; distance <= 12 && !best; distance++) {
    for (const width of [Math.round(camera.width) - distance, Math.round(camera.width) + distance]) {
      if (width < 64 || width > W) continue;
      const height = Math.round((width * OH) / OW);
      const scale = OW / width;
      if (height <= H && Math.round(width * scale) === OW && Math.round(height * scale) === OH) {
        best = { width, height, scale };
        break;
      }
    }
  }
  if (!best) throw new Error(`no exact clip near width ${camera.width}`);
  const cx = camera.x + camera.width / 2;
  const cy = camera.y + camera.height / 2;
  const place = (center, size, limit) => {
    const value = Math.min(Math.max(0, center - size / 2), limit - size);
    const snapped = Math.round(value * best.scale) / best.scale;
    return snapped > limit - size ? Math.floor((limit - size) * best.scale) / best.scale : snapped;
  };
  return { x: place(cx, best.width, W), y: place(cy, best.height, H), ...best };
}
function fit(box) {
  const { width: W, height: H } = scene.view;
  let width = Math.max(box.width, (box.height * 16) / 9, scene.minCameraWidth);
  width = Math.min(width, W);
  const height = (width * 9) / 16;
  let x = box.x + box.width / 2 - width / 2;
  let y = box.y + box.height / 2 - height / 2;
  x = Math.min(Math.max(0, x), W - width);
  y = Math.min(Math.max(0, y), H - height);
  return { x, y, width, height };
}
async function region(page, name) {
  const parts = scene.regions[name]();
  let box = null;
  let core = null;
  const union = (a, b) => {
    if (!a) return b;
    const x = Math.min(a.x, b.x);
    const y = Math.min(a.y, b.y);
    return { x, y, width: Math.max(a.x + a.width, b.x + b.width) - x, height: Math.max(a.y + a.height, b.y + b.height) - y };
  };
  for (const part of parts) {
    let rect = part;
    if (part.selector) {
      rect = await page.evaluate((selector) => window.__film.rect(selector), part.selector);
      if (!rect) continue;
      core = union(core, rect);
      const [top, right, bottom, left] = part.pad;
      rect = { x: rect.x - left, y: rect.y - top, width: rect.width + left + right, height: rect.height + top + bottom };
    }
    box = union(box, rect);
  }
  if (!box) throw new Error(`camera region ${name} not found`);
  return { camera: fit(box), core };
}

// Ink boxes that a clip's top or bottom edge slices. Edges on the viewport
// boundary are the product's own and are not checked. With margin > 0 an edge
// must also keep that distance from ink it does not cut.
function edgeCuts(boxes, clip, margin) {
  const edges = [];
  if (clip.y > 0.5) edges.push(['top', clip.y]);
  if (clip.y + clip.height < scene.view.height - 0.5) edges.push(['bottom', clip.y + clip.height]);
  const cuts = [];
  for (const box of boxes) {
    if (box.x + box.width <= clip.x || box.x >= clip.x + clip.width) continue;
    for (const [edge, at] of edges) {
      if (box.y - margin < at && box.y + box.height + margin > at) {
        cuts.push({ edge, depth: Math.round(Math.min(at - box.y, box.y + box.height - at) * 100) / 100, ...box });
      }
    }
  }
  return cuts;
}

// Choose the camera for a region: the fitted 16:9 box, moved as little as
// possible so its top and bottom edges clear the current ink.
async function framing(page, name) {
  const { camera, core } = await region(page, name);
  const rule = scene.cameraClearance;
  const { width: W, height: H } = scene.view;
  const fitted = snapCamera(camera);
  if (!core) return { camera, report: { name, fitted, chosen: fitted, cuts: [] } };
  const boxes = await page.evaluate(() => window.__film.ink());
  const inside = (clip) =>
    core.x >= clip.x + rule.coreMargin &&
    core.y >= clip.y + rule.coreMargin &&
    core.x + core.width <= clip.x + clip.width - rule.coreMargin &&
    core.y + core.height <= clip.y + clip.height - rule.coreMargin;
  const centerX = camera.x + camera.width / 2;
  const centerY = camera.y + camera.height / 2;
  let best = null;
  for (const grow of rule.grow) {
    const width = Math.min(W, camera.width * grow);
    const height = (width * 9) / 16;
    for (let step = 0; step * rule.step <= rule.maxShift; step++) {
      for (const sign of step ? [1, -1] : [1]) {
        const candidate = {
          x: Math.min(Math.max(0, centerX - width / 2), W - width),
          y: Math.min(Math.max(0, centerY - height / 2 + sign * step * rule.step), H - height),
          width,
          height,
        };
        const clip = snapCamera(candidate);
        // The fitted framing is the fallback even if its padding is tight.
        if (best && !inside(clip)) continue;
        const cuts = edgeCuts(boxes, clip, rule.margin);
        const cost = cuts.length * rule.cutCost + Math.abs(clip.y - fitted.y) + (grow - 1) * rule.growCost;
        if (!best || cost < best.cost) best = { cost, candidate, clip, cuts, grow };
      }
    }
  }
  const label = (cut) => `${cut.edge}:${cut.kind}:${cut.label}`;
  return {
    camera: best.candidate,
    report: {
      name,
      fitted,
      chosen: best.clip,
      grow: best.grow,
      fittedCuts: edgeCuts(boxes, fitted, rule.margin).map(label),
      cuts: best.cuts.map(label),
    },
  };
}

// ------------------------------------------------------------ main
async function main() {
  const options = args();
  const plan = readJson(options.plan);
  const recording = readJson(options.recording);
  const fps = Number(plan.video.fps);
  if (plan.video.width !== scene.output.width || plan.video.height !== scene.output.height) {
    throw new Error('plan size differs from scene output');
  }
  const appDist = path.resolve(options.app);
  if (options.build || !fs.existsSync(path.join(appDist, 'index.html'))) buildApp(appDist);

  let wanted;
  if (options.keyframes) wanted = [...new Set(plan.keyframes.map((item) => item.frame))];
  else if (options.list) wanted = options.list.split(',').map(Number);
  else if (options.clip) {
    const clip = plan.previewClips.find((item) => item.id === options.clip);
    if (!clip) throw new Error(`unknown clip ${options.clip}`);
    wanted = Array.from({ length: clip.frameCount }, (_, i) => clip.startFrame + i);
  } else if (options.frames) {
    const [a, b] = options.frames.split('-').map(Number);
    wanted = Array.from({ length: b - a }, (_, i) => a + i);
  } else wanted = Array.from({ length: plan.video.totalFrames }, (_, i) => i);
  wanted.sort((a, b) => a - b);
  const capture = new Set(wanted);
  const lastFrame = wanted.at(-1);

  const out = path.resolve(options.out);
  if (fs.existsSync(out) && fs.readdirSync(out).length) throw new Error(`output exists and is not empty: ${out}`);
  fs.mkdirSync(out, { recursive: true });

  const backend = phases(recording, plan);
  const firstCreated = recording.snapshots[0].value.created_at;
  const config = {
    // Virtual wall clock starts a little before the recorded creation time.
    epochMs: Math.round(firstCreated * 1000) - Math.round(ms(plan.events.find((e) => e.id === 'start-run').frame)),
    recording,
    backend,
    view: scene.view,
    overlayHtml: scene.overlayHtml,
    overlayCss: scene.overlayCss,
  };
  function ms(frame) {
    return (frame * 1000) / fps;
  }

  const server = await serve(appDist);
  const port = server.address().port;
  const browser = await chromium.launch({
    headless: true,
    executablePath: options.chrome || undefined,
    args: ['--force-color-profile=srgb', '--hide-scrollbars', '--disable-gpu-vsync', '--font-render-hinting=none'],
  });
  const context = await browser.newContext({
    viewport: scene.view,
    deviceScaleFactor: 1,
    locale: 'zh-CN',
    timezoneId: 'Asia/Shanghai',
    colorScheme: 'light',
    reducedMotion: 'no-preference',
  });
  await context.addInitScript(() => {
    try {
      localStorage.clear();
      localStorage.setItem('zhrag-theme', 'light');
    } catch {
      /* storage unavailable */
    }
  });
  await context.addInitScript((value) => {
    window.__FILM_CONFIG__ = value;
  }, config);
  await context.addInitScript({ path: path.join(here, 'film-runtime.js') });
  const page = await context.newPage();
  const consoleErrors = [];
  page.on('console', (message) => {
    if (['error', 'warning'].includes(message.type())) consoleErrors.push(`${message.type()}: ${message.text()}`);
  });
  page.on('pageerror', (error) => consoleErrors.push(`pageerror: ${error.message}`));
  await page.goto(`http://127.0.0.1:${port}/workbench/`, { waitUntil: 'load' });
  await page.evaluate(async () => {
    await document.fonts.ready;
  });
  await page.evaluate(() => window.__film.advanceTo(0));
  const cdp = await context.newCDPSession(page);

  // Input actions from the timeline.
  const shots = plan.shots;
  const shotAt = (frame) => shots.find((shot) => frame >= shot.startFrame && frame < shot.endFrameExclusive);
  const clicks = plan.events.filter((event) => event.action === 'click');
  const typing = [];
  for (const event of plan.events.filter((item) => item.action === 'type')) {
    [...event.text].forEach((char, index) => typing.push({ frame: event.frame + 1 + index * event.everyFrames, char, id: event.id }));
  }
  const cursor = { x: scene.motion.rest.x, y: scene.motion.rest.y, visible: true, opacity: 1 };
  let move = null;
  let parked = null;
  let lastClick = null;
  const clickLog = [];
  const checks = [];
  const pendingChecks = [];
  let camera = { x: 0, y: 0, width: scene.view.width, height: scene.view.height };
  let cameraMove = null;
  const cameraCues = [];
  for (const shot of shots) {
    for (const [at, name, frames] of shot.data.camera || []) {
      cameraCues.push({ frame: shot.startFrame + Math.round(at * fps), name, frames: frames || 0 });
    }
  }
  const timings = [];
  const layout = {};
  const framingLog = [];
  const edgeLog = {};
  const caretLog = {};
  const overlayLog = {};
  const probeLog = {};
  const cursorInkLog = {};
  const measured = ['.workbench', '.conversation-pane', '.execution-pane', '.inspector-slot', '.graph-canvas',
    '.timeline', '.investigation-controls', '.answer-body', '.evidence-detail'];
  const manifest = { planSha256: plan.planSha256, complete: false, frames: [], camera: {}, size: scene.output };
  let mouseAt = null;

  for (let frame = 0; frame <= lastFrame; frame++) {
    const started = Date.now();
    await page.evaluate((t) => window.__film.advanceTo(t), ms(frame));
    const shot = shotAt(frame);

    // Camera cues resolve named regions against the live layout.
    for (const cue of cameraCues.filter((item) => item.frame === frame)) {
      const { camera: to, report } = await framing(page, cue.name);
      framingLog.push({ frame, ...report });
      cameraMove = cue.frames ? { from: { ...camera }, to, start: frame, frames: cue.frames } : null;
      if (!cue.frames) camera = to;
    }
    if (cameraMove) {
      const p = ease(Math.min(1, (frame - cameraMove.start) / cameraMove.frames));
      const lerp = (a, b) => a + (b - a) * p;
      camera = {
        x: lerp(cameraMove.from.x, cameraMove.to.x),
        y: lerp(cameraMove.from.y, cameraMove.to.y),
        width: lerp(cameraMove.from.width, cameraMove.to.width),
      };
      camera.height = (camera.width * 9) / 16;
      if (p >= 1) cameraMove = null;
    }

    // Cursor travel toward the next click target.
    if (parked && frame >= parked.start) {
      move = { park: true, id: null, start: frame, from: { x: cursor.x, y: cursor.y }, to: { x: cursor.x, y: cursor.y }, spec: parked.spec };
      parked = null;
    }
    const upcoming = clicks.find((event) => event.frame >= frame && event.frame - frame <= scene.motion.travel + scene.motion.dwell);
    if (upcoming && (!move || move.id !== upcoming.id)) {
      move = { id: upcoming.id, from: { x: cursor.x, y: cursor.y }, start: frame, end: upcoming.frame - scene.motion.dwell, event: upcoming };
    }
    if (move && move.park) {
      // Track the parking spot during the move (layout may still be settling).
      // Until the spot exists the glide waits rather than jumping later.
      const spot = await page.evaluate((value) => window.__film.target(value), move.spec);
      if (spot) {
        const [fx, fy] = move.spec.at || [0.5, 0.5];
        const [ox, oy] = move.spec.offset || [0, 0];
        move.to = { x: spot.x + spot.width * fx + ox, y: spot.y + spot.height * fy + oy };
      } else move.start = frame + 1;
      const p = ease(Math.min(1, Math.max(0, (frame - move.start) / scene.motion.parkFrames)));
      cursor.x = move.from.x + (move.to.x - move.from.x) * p;
      cursor.y = move.from.y + (move.to.y - move.from.y) * p;
      if (p >= 1) move = null;
    } else if (move) {
      const spec = scene.targets[move.event.target];
      const box = await page.evaluate((value) => window.__film.target(value), spec);
      if (box) {
        const [fx, fy] = spec.at || [0.5, 0.5];
        const goal = { x: box.x + box.width * fx, y: box.y + box.height * fy };
        const span = Math.max(1, move.end - move.start);
        const p = ease(Math.min(1, Math.max(0, (frame - move.start) / span)));
        // A gentle arc keeps long moves from reading as mechanical.
        const lift = Math.sin(Math.PI * p) * Math.min(40, Math.hypot(goal.x - move.from.x, goal.y - move.from.y) * 0.08);
        cursor.x = move.from.x + (goal.x - move.from.x) * p;
        cursor.y = move.from.y + (goal.y - move.from.y) * p - lift;
        if (frame >= move.end) {
          cursor.x = goal.x;
          cursor.y = goal.y;
        }
      }
    }
    if (shot.id === 'brand') cursor.opacity = Math.max(0, 1 - (frame - shot.startFrame) / 9);
    const point = { x: Math.round(cursor.x * 100) / 100, y: Math.round(cursor.y * 100) / 100 };
    if (!mouseAt || mouseAt.x !== point.x || mouseAt.y !== point.y) {
      await page.mouse.move(point.x, point.y);
      mouseAt = point;
    }

    // Real input at its frame: hit-test first, then the native event.
    for (const event of clicks.filter((item) => item.frame === frame)) {
      const spec = scene.targets[event.target];
      const hit = await page.evaluate(([value, x, y]) => window.__film.hit(value, x, y), [spec, point.x, point.y]);
      clickLog.push({ id: event.id, frame, x: point.x, y: point.y, ...hit });
      if (!hit.hit || hit.disabled) throw new Error(`click ${event.id} at frame ${frame} misses: ${JSON.stringify(hit)}`);
      await page.mouse.down();
      await page.mouse.up();
      lastClick = { frame, x: point.x, y: point.y };
      move = null;
      const park = scene.motion.park[event.id];
      if (park) parked = { start: frame + scene.motion.parkDelay, spec: park };
      const expectation = scene.expectations[event.id];
      if (expectation) pendingChecks.push({ id: event.id, frame: frame + expectation[0], selector: expectation[1], text: expectation[2] });
    }
    for (const key of typing.filter((item) => item.frame === frame)) await page.keyboard.insertText(key.char);
    await page.evaluate(() => window.__film.flush());

    for (const check of pendingChecks.filter((item) => item.frame === frame)) {
      const texts = await page.evaluate((selector) => window.__film.text(selector), check.selector);
      const ok = texts.some((value) => value.includes(check.text));
      checks.push({ ...check, ok, texts: texts.slice(0, 4) });
      if (!ok) throw new Error(`expectation after ${check.id} failed at ${frame}: ${JSON.stringify(texts)}`);
    }

    const shell = scene.shellStyle({ frame, shot });
    await page.evaluate((style) => {
      const element = document.querySelector('.app-shell');
      if (element) Object.assign(element.style, style);
    }, shell);
    // Chromium truncates a clip to whole CSS pixels and rounds width x scale.
    // Pick the nearest integer width whose 16:9 clip lands on exactly
    // 1920 x 1080; offsets stay fractional (snapped to output pixels).
    const clip = snapCamera(camera);
    const scale = clip.scale;
    delete clip.scale;
    const overlay = scene.overlayState({ frame, fps, shot, camera: clip, cursor, click: lastClick, plan });
    await page.evaluate((state) => window.__film.overlay(state), { camera: clip, parts: overlay });

    if (capture.has(frame)) {
      const { data } = await cdp.send('Page.captureScreenshot', {
        format: 'png',
        clip: { ...clip, scale },
        captureBeyondViewport: false,
        optimizeForSpeed: true,
      });
      const buffer = Buffer.from(data, 'base64');
      const width = buffer.readUInt32BE(16);
      const height = buffer.readUInt32BE(20);
      if (width !== scene.output.width || height !== scene.output.height) {
        throw new Error(`frame ${frame} captured at ${width}x${height} clip ${JSON.stringify(clip)} scale ${scale}`);
      }
      fs.writeFileSync(path.join(out, `frame-${String(frame).padStart(6, '0')}.png`), buffer);
      manifest.frames.push(frame);
      manifest.camera[frame] = clip;
      // Cursor tip and click ripple in output pixels, for masking in checks.
      const toOutput = (x, y) => [Math.round((x - clip.x) * scale * 10) / 10, Math.round((y - clip.y) * scale * 10) / 10];
      overlayLog[frame] = {
        cursor: cursor.visible && cursor.opacity > 0 ? [...toOutput(cursor.x, cursor.y), cursor.opacity] : null,
        ripple: lastClick && frame >= lastClick.frame && frame - lastClick.frame < 14 ? toOutput(lastClick.x, lastClick.y) : null,
      };
      layout[frame] = await page.evaluate((selectors) => {
        const result = {};
        for (const selector of selectors) result[selector] = window.__film.rect(selector);
        return result;
      }, measured);
      // Audit what this frame's top and bottom edges actually slice.
      const ink = await page.evaluate((span) => window.__film.ink(span), { x: clip.x, width: clip.width });
      const cuts = edgeCuts(ink, clip, 0);
      if (cuts.length) edgeLog[frame] = cuts.map((cut) => ({ edge: cut.edge, kind: cut.kind, label: cut.label, depth: cut.depth }));
      // Text under the cursor arrow (its bounding box in product pixels).
      if (cursor.visible && cursor.opacity > 0) {
        const size = clip.width / scene.view.width;
        const arrow = { x: cursor.x, y: cursor.y, width: scene.cursorArrow.width * size, height: scene.cursorArrow.height * size };
        const covered = ink.filter((box) => box.x < arrow.x + arrow.width && box.x + box.width > arrow.x &&
          box.y < arrow.y + arrow.height && box.y + box.height > arrow.y);
        if (covered.length) cursorInkLog[frame] = covered.map((box) => `${box.kind}:${box.label}`);
      }
      const caret = await page.evaluate(() => window.__film.caret());
      if (caret.origin !== undefined) caretLog[frame] = caret;
      if (scene.probes[frame]) {
        probeLog[frame] = await page.evaluate((specs) => {
          const result = {};
          for (const [name, spec] of Object.entries(specs)) result[name] = window.__film.read(spec);
          return result;
        }, scene.probes[frame]);
      }
    }
    timings.push(Date.now() - started);
    if (frame % 60 === 0) process.stdout.write(`frame ${frame}/${lastFrame}\n`);
  }

  const errors = await page.evaluate(() => window.__film.errors());
  const backendLog = await page.evaluate(() => window.__film.backendLog());
  manifest.complete = wanted.every((frame) => manifest.frames.includes(frame));
  writeJson(path.join(out, 'render-manifest.json'), manifest);
  writeJson(path.join(out, 'render-log.json'), {
    planSha256: plan.planSha256,
    requested: { keyframes: options.keyframes, clip: options.clip, frames: options.frames, list: options.list },
    clicks: clickLog,
    checks,
    backend: backendLog,
    backendPlan: backend,
    pageErrors: errors,
    console: consoleErrors,
    framing: framingLog,
    edgeCuts: edgeLog,
    caret: caretLog,
    overlay: overlayLog,
    cursorInk: cursorInkLog,
    probes: probeLog,
    layout,
    meanFrameMs: Math.round(timings.reduce((a, b) => a + b, 0) / timings.length),
  });
  await browser.close();
  server.close();
  if (errors.length) {
    console.error(errors.join('\n'));
    process.exitCode = 1;
  }
  console.log(JSON.stringify({ frames: manifest.frames.length, complete: manifest.complete, errors: errors.length }));
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
