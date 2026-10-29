// Film runtime for the resumable-investigation promo. Injected before the real
// production bundle of frontend/ via Playwright addInitScript; it never edits the
// product. It provides:
//   1. a virtual clock (timers, animation frames, performance.now, Date),
//   2. a replay of the recorded synthetic backend (REST + SSE),
//   3. frame-exact seeking of CSS animations/transitions,
//   4. a screen-space overlay (cursor, captions, synthetic-demo label, brand card).
// Every visible state is a function of the integer frame the driver advances to.
(() => {
  'use strict';
  const config = window.__FILM_CONFIG__;
  if (!config) throw new Error('film config missing');
  const real = {
    Date: window.Date,
    fetch: window.fetch.bind(window),
    requestAnimationFrame: window.requestAnimationFrame.bind(window),
  };
  const errors = [];
  const report = (error) => errors.push(String(error && error.stack ? error.stack : error));
  window.addEventListener('error', (event) => report(event.error || event.message));
  window.addEventListener('unhandledrejection', (event) => {
    const reason = event.reason;
    // Aborted fetches are part of the product's normal reconnect path.
    if (reason && reason.name === 'AbortError') return;
    report(reason);
  });

  // ---------------------------------------------------------------- clock
  let now = 0;
  let nextId = 1;
  const timers = new Map();
  const frameCallbacks = new Map();
  function addTimer(fn, delay, args, repeat) {
    const id = nextId++;
    const ms = Math.max(0, Number(delay) || 0);
    timers.set(id, { id, at: now + ms, fn, args, every: repeat ? Math.max(1, ms) : 0 });
    return id;
  }
  window.setTimeout = (fn, delay, ...args) => addTimer(fn, delay, args, false);
  window.setInterval = (fn, delay, ...args) => addTimer(fn, delay, args, true);
  window.clearTimeout = (id) => void timers.delete(id);
  window.clearInterval = (id) => void timers.delete(id);
  window.requestAnimationFrame = (fn) => {
    const id = nextId++;
    frameCallbacks.set(id, fn);
    return id;
  };
  window.cancelAnimationFrame = (id) => void frameCallbacks.delete(id);
  window.requestIdleCallback = (fn) =>
    addTimer(() => fn({ didTimeout: false, timeRemaining: () => 0 }), 1, [], false);
  window.cancelIdleCallback = (id) => void timers.delete(id);
  performance.now = () => now;
  function VirtualDate(...args) {
    if (!new.target) return new real.Date(config.epochMs + now).toString();
    return args.length ? new real.Date(...args) : new real.Date(config.epochMs + now);
  }
  VirtualDate.prototype = real.Date.prototype;
  VirtualDate.now = () => config.epochMs + now;
  VirtualDate.parse = real.Date.parse;
  VirtualDate.UTC = real.Date.UTC;
  window.Date = VirtualDate;

  // Motion uses WAAPI only when Element.prototype owns animate(). Keep the
  // method reachable for every element but route Motion through its frame loop,
  // which this clock drives exactly once per video frame.
  const nativeAnimate = Element.prototype.animate;
  delete Element.prototype.animate;
  HTMLElement.prototype.animate = nativeAnimate;
  SVGElement.prototype.animate = nativeAnimate;

  const channel = new MessageChannel();
  const waiting = [];
  channel.port1.onmessage = () => {
    const resolve = waiting.shift();
    if (resolve) resolve();
  };
  const macrotask = () =>
    new Promise((resolve) => {
      waiting.push(resolve);
      channel.port2.postMessage(0);
    });
  async function settle(rounds = 8) {
    for (let index = 0; index < rounds; index++) await macrotask();
  }
  // Resize/intersection observers run in the browser's own rendering steps.
  // Waiting for two real frames makes their layout-driven updates land in the
  // same video frame on every render, sequential or not.
  const realFrame = () => new Promise((resolve) => real.requestAnimationFrame(() => resolve()));
  async function render() {
    await realFrame();
    await realFrame();
    await settle();
  }

  // ---------------------------------------------------- animation seeking
  const animationStart = new WeakMap();
  function syncAnimations() {
    for (const animation of document.getAnimations()) {
      if (!animationStart.has(animation)) {
        animationStart.set(animation, now);
        animation.pause();
      }
      const local = (now - animationStart.get(animation)) * (animation.playbackRate || 1);
      const timing = animation.effect ? animation.effect.getComputedTiming() : null;
      const end = timing ? timing.endTime : Infinity;
      if (Number.isFinite(end) && local >= end) {
        try {
          animation.finish();
        } catch (error) {
          report(error);
        }
      } else {
        animation.currentTime = Math.max(0, local);
      }
    }
  }

  function nextDue(limit) {
    let best = null;
    for (const timer of timers.values()) {
      if (timer.at > limit) continue;
      if (!best || timer.at < best.at || (timer.at === best.at && timer.id < best.id)) best = timer;
    }
    return best;
  }
  async function runTimersUntil(target) {
    let guard = 0;
    for (;;) {
      const timer = nextDue(target);
      if (!timer) break;
      if (++guard > 50000) throw new Error('timer storm');
      now = Math.max(now, timer.at);
      if (timer.every) timer.at += timer.every;
      else timers.delete(timer.id);
      try {
        if (typeof timer.fn === 'function') timer.fn(...timer.args);
      } catch (error) {
        report(error);
      }
      await settle();
      syncAnimations();
    }
  }
  async function advanceTo(target) {
    if (target < now) throw new Error(`clock cannot go back: ${target} < ${now}`);
    await runTimersUntil(target);
    now = target;
    const callbacks = [...frameCallbacks.values()];
    frameCallbacks.clear();
    for (const callback of callbacks) {
      try {
        callback(now);
      } catch (error) {
        report(error);
      }
    }
    await settle();
    // Work scheduled at exactly this instant by the frame callbacks.
    await runTimersUntil(target);
    await render();
    await runTimersUntil(target);
    syncAnimations();
  }
  async function flush() {
    // After driver input at the current instant: let React commit, then seek.
    await settle();
    await runTimersUntil(now);
    await render();
    await runTimersUntil(now);
    syncAnimations();
  }

  // ---------------------------------------------------- replayed backend
  const recording = config.recording;
  const plan = config.backend;
  const runId = recording.run_id;
  const terminal = new Set(['completed', 'failed', 'cancelled', 'budget_exhausted', 'unrecoverable']);
  const eventBySeq = new Map(recording.events.map((event) => [event.seq, event]));
  const encoder = new TextEncoder();
  const backend = {
    up: true,
    current: null,
    streams: new Set(),
    commands: [],
    requests: [],
    resumes: 0,
  };
  const snapshot = () => recording.snapshots[backend.current].value;
  function pump(stream) {
    if (stream.closed || backend.current === null) return;
    const value = snapshot();
    while (stream.cursor < value.cursor) {
      const event = eventBySeq.get(stream.cursor + 1);
      if (!event) {
        report(new Error(`missing recorded event ${stream.cursor + 1}`));
        break;
      }
      stream.cursor = event.seq;
      const data = JSON.stringify(event);
      stream.controller.enqueue(encoder.encode(`id: ${event.seq}\nevent: run_event\ndata: ${data}\n\n`));
    }
    if (terminal.has(value.status) && stream.cursor >= value.cursor) closeStream(stream);
  }
  function closeStream(stream) {
    if (stream.closed) return;
    stream.closed = true;
    backend.streams.delete(stream);
    try {
      stream.controller.close();
    } catch (error) {
      report(error);
    }
  }
  function show(index, why) {
    backend.current = index;
    backend.requests.push({ t: now, kind: 'serve', index, why, status: snapshot().status });
    for (const stream of [...backend.streams]) pump(stream);
  }
  function schedule(steps) {
    for (const [offset, index] of steps) window.setTimeout(() => show(index, 'step'), offset);
  }
  function startPhase(name) {
    const phase = plan.phases[name];
    if (!phase) throw new Error(`no backend phase ${name}`);
    show(phase.response, `${name}-response`);
    schedule(phase.steps);
    if (phase.outage) {
      window.setTimeout(() => {
        backend.up = false;
        backend.requests.push({ t: now, kind: 'outage-start' });
        for (const stream of [...backend.streams]) closeStream(stream);
      }, phase.outage.start);
      window.setTimeout(() => {
        backend.up = true;
        backend.requests.push({ t: now, kind: 'outage-end' });
        show(phase.outage.restored, 'restart');
      }, phase.outage.end);
    }
    return recording.snapshots[phase.response].value;
  }
  const json = (body, status = 200) =>
    new Response(JSON.stringify(body), {
      status,
      headers: { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' },
    });
  function listing() {
    if (backend.current === null) return { schema_version: 1, runs: [] };
    const value = snapshot();
    return {
      schema_version: 1,
      runs: [
        {
          id: value.run_id,
          query: value.query,
          model: value.model,
          status: value.status,
          reason: value.reason,
          created_at: value.created_at,
          version: value.version,
        },
      ],
    };
  }
  function route(method, url, body) {
    const path = url.pathname;
    if (method === 'GET' && path === '/api/capabilities') return json(recording.capabilities);
    if (path === '/api/investigations') {
      if (method === 'GET') return json(listing());
      if (method === 'POST') {
        if (body.query !== recording.question) report(new Error(`unexpected query ${body.query}`));
        backend.commands.push({ t: now, action: 'create', body });
        return json(startPhase('create'), 201);
      }
    }
    const prefix = `/api/investigations/${runId}`;
    if (method === 'GET' && path === prefix) return json(snapshot());
    if (method === 'GET' && path === `${prefix}/stream`) {
      const after = Number(url.searchParams.get('after') || 0);
      const stream = { cursor: after, closed: false, controller: null };
      const body = new ReadableStream({
        start(controller) {
          stream.controller = controller;
        },
        cancel() {
          stream.closed = true;
          backend.streams.delete(stream);
        },
      });
      backend.streams.add(stream);
      window.setTimeout(() => pump(stream), 0);
      return new Response(body, {
        status: 200,
        headers: { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-store' },
      });
    }
    const command = path.startsWith(`${prefix}/`) ? path.slice(prefix.length + 1) : '';
    if (method === 'POST' && ['pause', 'resume', 'supplement', 'cancel'].includes(command)) {
      const current = snapshot();
      if (body.version !== current.version) {
        report(new Error(`${command} sent version ${body.version}, backend at ${current.version}`));
        return json({ code: 'stale_run_version', message: 'stale_run_version' }, 409);
      }
      if (command === 'supplement' && body.wait_id !== current.wait_id) {
        report(new Error('supplement wait_id mismatch'));
      }
      backend.commands.push({ t: now, action: command, body });
      const name = command === 'resume' ? `resume${++backend.resumes}` : command;
      return json(startPhase(name));
    }
    report(new Error(`unmocked request ${method} ${path}`));
    return json({ code: 'not_found' }, 404);
  }
  window.fetch = function filmFetch(input, init = {}) {
    const url = new URL(typeof input === 'string' ? input : input.url, location.href);
    if (!url.pathname.startsWith('/api/')) return real.fetch(input, init);
    const method = String(init.method || 'GET').toUpperCase();
    const signal = init.signal;
    const body = init.body ? JSON.parse(init.body) : undefined;
    backend.requests.push({ t: now, kind: 'request', method, path: url.pathname + url.search });
    return new Promise((resolve, reject) => {
      const aborted = () => reject(new DOMException('The operation was aborted.', 'AbortError'));
      if (signal && signal.aborted) return aborted();
      const latency = backend.up
        ? method === 'POST'
          ? plan.latency.post
          : plan.latency.get
        : plan.latency.refused;
      const timer = window.setTimeout(() => {
        if (signal) signal.removeEventListener('abort', onAbort);
        if (!backend.up) {
          backend.requests.push({ t: now, kind: 'refused', path: url.pathname });
          reject(new TypeError('Failed to fetch'));
          return;
        }
        try {
          resolve(route(method, url, body));
        } catch (error) {
          report(error);
          reject(error);
        }
      }, latency);
      function onAbort() {
        window.clearTimeout(timer);
        aborted();
      }
      if (signal) signal.addEventListener('abort', onAbort, { once: true });
    });
  };

  // ------------------------------------------------------------- caret
  // Chromium blinks the text caret on its own wall clock, which frame-stepped
  // capture samples at random phases (a 7 Hz flicker that differs per render).
  // The native caret is made transparent and redrawn on the virtual clock with
  // Chromium's default rules: solid when a field gains focus or its text or
  // selection changes, then toggling every 500 ms. Position and height come
  // from a hidden mirror of the focused field laid out with its computed style.
  const caretBlinkMs = 500;
  const caretIds = new WeakMap();
  let caretSerial = 0;
  let caretKey = null;
  let caretOrigin = 0;
  let caretElement;
  let caretMirror;
  let caretState = { visible: false };
  function caretTarget() {
    const element = document.activeElement;
    if (!element || !document.hasFocus()) return null;
    const textual =
      element.tagName === 'TEXTAREA' ||
      (element.tagName === 'INPUT' && ['text', 'search', 'url', 'email', 'tel', 'password'].includes(element.type));
    if (!textual || element.disabled || element.readOnly || !element.getClientRects().length) return null;
    return element.selectionStart === element.selectionEnd ? element : null;
  }
  function caretBox(element) {
    if (!caretMirror) {
      caretMirror = document.createElement('div');
      caretMirror.id = 'film-caret-mirror';
      caretMirror.setAttribute('aria-hidden', 'true');
      document.documentElement.append(caretMirror);
    }
    const style = getComputedStyle(element);
    const mirror = caretMirror.style;
    mirror.cssText = '';
    for (const name of style) mirror.setProperty(name, style.getPropertyValue(name));
    Object.assign(mirror, {
      position: 'absolute',
      left: '0px',
      top: '0px',
      margin: '0px',
      display: 'block',
      visibility: 'hidden',
      overflow: 'hidden',
      boxSizing: 'border-box',
      width: `${element.offsetWidth}px`,
      height: 'auto',
      minHeight: '0px',
      maxHeight: 'none',
      transform: 'none',
      transition: 'none',
      animation: 'none',
      pointerEvents: 'none',
    });
    const at = element.selectionEnd;
    caretMirror.textContent = element.value.slice(0, at);
    const marker = document.createElement('span');
    marker.textContent = element.value.slice(at) || '.';
    caretMirror.append(marker);
    const outer = caretMirror.getBoundingClientRect();
    const line = marker.getClientRects()[0] || marker.getBoundingClientRect();
    const field = element.getBoundingClientRect();
    return {
      x: field.left + line.left - outer.left - element.scrollLeft,
      y: field.top + line.top - outer.top - element.scrollTop,
      height: line.height,
      color: style.color,
    };
  }
  function updateCaret() {
    if (!caretElement) {
      const style = document.createElement('style');
      style.id = 'film-caret-style';
      style.textContent = 'textarea, input { caret-color: transparent !important; }';
      document.documentElement.append(style);
      caretElement = document.createElement('div');
      caretElement.id = 'film-caret';
      caretElement.setAttribute('aria-hidden', 'true');
      caretElement.style.cssText =
        'position: fixed; width: 1px; display: none; pointer-events: none; z-index: 2147483646;';
      document.documentElement.append(caretElement);
    }
    const element = caretTarget();
    if (!element) {
      caretKey = null;
      caretElement.style.display = 'none';
      caretState = { visible: false };
      return;
    }
    if (!caretIds.has(element)) caretIds.set(element, ++caretSerial);
    const key = `${caretIds.get(element)}|${element.selectionEnd}|${element.value}`;
    if (key !== caretKey) {
      caretKey = key;
      caretOrigin = now;
    }
    // Frame times are thirds of a millisecond; the epsilon keeps a phase that
    // lands exactly on a toggle from rounding down a frame late.
    const visible = Math.floor((now - caretOrigin) / caretBlinkMs + 1e-6) % 2 === 0;
    const box = caretBox(element);
    caretState = { visible, origin: caretOrigin, x: box.x, y: box.y, height: box.height };
    if (!visible) {
      caretElement.style.display = 'none';
      return;
    }
    Object.assign(caretElement.style, {
      display: 'block',
      left: `${box.x}px`,
      top: `${box.y}px`,
      height: `${box.height}px`,
      background: box.color,
    });
  }

  // ------------------------------------------------------------- overlay
  const view = config.view; // CSS viewport of the product, e.g. 1600 x 900
  let overlay;
  const parts = {};
  function buildOverlay() {
    overlay = document.createElement('div');
    overlay.id = 'film-overlay';
    overlay.setAttribute('aria-hidden', 'true');
    overlay.innerHTML = config.overlayHtml;
    const style = document.createElement('style');
    style.textContent = config.overlayCss;
    overlay.prepend(style);
    document.documentElement.append(overlay);
    for (const element of overlay.querySelectorAll('[data-part]')) parts[element.dataset.part] = element;
  }
  function applyOverlay(state) {
    if (!overlay) buildOverlay();
    updateCaret();
    const camera = state.camera;
    const scale = camera.width / view.width;
    overlay.style.transform = `translate(${camera.x}px, ${camera.y}px) scale(${scale})`;
    for (const [name, values] of Object.entries(state.parts || {})) {
      const element = parts[name];
      if (!element) continue;
      for (const [key, value] of Object.entries(values)) {
        if (key === 'text') element.textContent = value;
        else if (key === 'html') element.innerHTML = value;
        else element.style.setProperty(key, value);
      }
    }
  }

  // --------------------------------------------------------------- probes
  function rect(selector, index = 0) {
    const element = typeof selector === 'string' ? document.querySelectorAll(selector)[index] : selector;
    if (!element) return null;
    const box = element.getBoundingClientRect();
    return { x: box.x, y: box.y, width: box.width, height: box.height };
  }
  function byText(selector, text) {
    return [...document.querySelectorAll(selector)].find(
      (element) => element.textContent.trim() === text || element.getAttribute('aria-label') === text,
    );
  }
  function target(spec) {
    const element = spec.text ? byText(spec.selector, spec.text) : document.querySelectorAll(spec.selector)[spec.index || 0];
    if (!element) return null;
    const box = element.getBoundingClientRect();
    return { x: box.x, y: box.y, width: box.width, height: box.height, disabled: !!element.disabled };
  }
  function hit(spec, x, y) {
    const element = spec.text ? byText(spec.selector, spec.text) : document.querySelectorAll(spec.selector)[spec.index || 0];
    const top = document.elementFromPoint(x, y);
    return {
      found: !!element,
      hit: !!(element && top && (element === top || element.contains(top))),
      top: top ? `${top.tagName.toLowerCase()}.${String(top.className && top.className.baseVal !== undefined ? top.className.baseVal : top.className).slice(0, 60)}` : null,
      disabled: !!(element && element.disabled),
    };
  }
  function text(selector) {
    return [...document.querySelectorAll(selector)].map((element) => element.textContent.trim());
  }
  // What a viewer reads in matching elements: a field's value, an attribute,
  // or the text without `exclude` descendants (e.g. inline citation badges),
  // with the element's box and whether the browser paints it at all. Content
  // of a closed <details> keeps a layout box but is not shown.
  function read(spec) {
    return [...document.querySelectorAll(spec.selector)].map((element) => {
      let text;
      if (spec.attr) text = element.getAttribute(spec.attr);
      else if (['TEXTAREA', 'INPUT'].includes(element.tagName)) text = element.value;
      else {
        const copy = element.cloneNode(true);
        if (spec.exclude) for (const child of copy.querySelectorAll(spec.exclude)) child.remove();
        text = copy.textContent.trim();
      }
      const box = element.getBoundingClientRect();
      return { text, box: { x: box.x, y: box.y, width: box.width, height: box.height }, visible: painted(element) };
    });
  }
  function painted(element) {
    return element.checkVisibility({ visibilityProperty: true, opacityProperty: true });
  }

  // Ink a frame edge must not slice: one box per rendered text line, the first
  // line of each text field (value or placeholder) and small icons. Boxes are
  // clipped by overflow ancestors; the film overlay and anything the browser
  // does not paint (closed <details> content, hidden or transparent) are
  // skipped. `within` limits the result to boxes overlapping a horizontal span
  // { x, width }.
  function ink(within) {
    const shownCache = new Map();
    const shown = (element) => {
      if (!shownCache.has(element)) shownCache.set(element, painted(element));
      return shownCache.get(element);
    };
    const innerCache = new Map();
    const inner = (element) => {
      // Where this element's content can paint, after every clipping ancestor.
      if (!element) return { x0: 0, y0: 0, x1: window.innerWidth, y1: window.innerHeight };
      if (innerCache.has(element)) return innerCache.get(element);
      const result = { ...inner(element.parentElement) };
      const style = getComputedStyle(element);
      if (style.overflowX !== 'visible' || style.overflowY !== 'visible') {
        const box = element.getBoundingClientRect();
        if (style.overflowX !== 'visible') {
          result.x0 = Math.max(result.x0, box.left);
          result.x1 = Math.min(result.x1, box.right);
        }
        if (style.overflowY !== 'visible') {
          result.y0 = Math.max(result.y0, box.top);
          result.y1 = Math.min(result.y1, box.bottom);
        }
      }
      innerCache.set(element, result);
      return result;
    };
    const boxes = [];
    const add = (box, clip, kind, label) => {
      const x0 = Math.max(box.left, clip.x0);
      const x1 = Math.min(box.right, clip.x1);
      const y0 = Math.max(box.top, clip.y0);
      const y1 = Math.min(box.bottom, clip.y1);
      if (x1 - x0 < 0.5 || y1 - y0 < 0.5) return;
      if (within && (x1 <= within.x || x0 >= within.x + within.width)) return;
      boxes.push({ x: x0, y: y0, width: x1 - x0, height: y1 - y0, kind, label });
    };
    const skip = '#film-overlay, #film-caret-mirror, script, style, noscript, textarea, title';
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
    const range = document.createRange();
    for (let node = walker.nextNode(); node; node = walker.nextNode()) {
      const value = node.nodeValue;
      const parent = node.parentElement;
      if (!value || !value.trim() || !parent || parent.closest(skip) || !shown(parent)) continue;
      range.selectNodeContents(node);
      const clip = inner(parent);
      for (const line of range.getClientRects()) add(line, clip, 'text', value.trim().slice(0, 18));
    }
    for (const field of document.querySelectorAll('textarea, input')) {
      if (['hidden', 'checkbox', 'radio'].includes(field.type) || field.closest('#film-overlay') || !shown(field)) continue;
      const label = field.value || field.placeholder;
      if (!label) continue;
      const style = getComputedStyle(field);
      const box = field.getBoundingClientRect();
      const px = (value) => parseFloat(value) || 0;
      const top = box.top + px(style.borderTopWidth) + px(style.paddingTop);
      const lineHeight = px(style.lineHeight) || px(style.fontSize) * 1.4;
      add(
        {
          left: box.left + px(style.borderLeftWidth) + px(style.paddingLeft),
          right: box.right - px(style.borderRightWidth) - px(style.paddingRight),
          top,
          bottom: top + lineHeight,
        },
        inner(field),
        'field',
        label.slice(0, 18),
      );
    }
    for (const icon of document.querySelectorAll('svg')) {
      if (icon.ownerSVGElement || icon.closest('#film-overlay') || !shown(icon)) continue;
      const box = icon.getBoundingClientRect();
      if (box.width < 4 || box.height < 4 || box.width > 48 || box.height > 48) continue;
      add(box, inner(icon.parentElement), 'icon', String(icon.getAttribute('class') || 'svg').slice(0, 32));
    }
    return boxes;
  }

  window.__film = {
    advanceTo,
    flush,
    now: () => now,
    overlay: applyOverlay,
    rect,
    target,
    hit,
    text,
    read,
    ink,
    caret: () => ({ ...caretState }),
    errors: () => errors.slice(),
    backendLog: () => ({ commands: backend.commands.slice(), requests: backend.requests.slice() }),
    pendingTimers: () => timers.size,
  };
})();
