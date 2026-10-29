// Plays the final MP4 once at 1x in system Chrome (H.264/AAC need the branded
// build) and records what the browser itself reports: presented frames,
// dropped frames, wall time and the audio level read through Web Audio, whose
// envelope is aligned with the file's own decoded audio track. It then seeks to
// a few frames and compares Chrome's decoded pixels with the rendered PNGs.
// A zero gain after the analyser keeps the output silent: this is not a
// listening test.
//
//   node playback.mjs --video <final.mp4> --frames <frames-dir> \
//     --chrome <chrome.exe> --at 171,585,1059,1170,1272 --out <playback.json> \
//     [--captures <dir>] [--flags "<extra Chrome switches>"]
import { createRequire } from 'node:module';
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const repo = path.resolve(here, '../../../../..');
const require = createRequire(path.join(repo, 'frontend', 'package.json'));
const { chromium } = require('playwright-core');

function args() {
  const result = {};
  const list = process.argv.slice(2);
  for (let index = 0; index < list.length; index += 2) {
    const key = list[index].replace(/^--/, '');
    if (!['video', 'frames', 'chrome', 'at', 'out', 'fps', 'flags', 'captures'].includes(key)) {
      throw new Error(`unknown argument ${list[index]}`);
    }
    result[key] = list[index + 1];
  }
  for (const key of ['video', 'frames', 'chrome', 'at', 'out']) {
    if (!result[key]) throw new Error(`--${key} is required`);
  }
  return result;
}

const page = `<!doctype html><meta charset="utf-8">
<style>body{margin:0;background:#000}video{display:block;width:1920px;height:1080px}</style>
<video src="/video.mp4" preload="auto" playsinline></video>`;

// Byte ranges, as the media pipeline asks for them.
function sendFile(request, response, file, type) {
  const size = fs.statSync(file).size;
  const range = /^bytes=(\d*)-(\d*)$/.exec(request.headers.range || '');
  if (!range) {
    response.writeHead(200, { 'Content-Type': type, 'Content-Length': size, 'Accept-Ranges': 'bytes' });
    fs.createReadStream(file).pipe(response);
    return;
  }
  const start = range[1] ? Number(range[1]) : size - Number(range[2]);
  const end = range[1] && range[2] ? Math.min(Number(range[2]), size - 1) : size - 1;
  response.writeHead(206, {
    'Content-Type': type,
    'Content-Length': end - start + 1,
    'Content-Range': `bytes ${start}-${end}/${size}`,
    'Accept-Ranges': 'bytes',
  });
  fs.createReadStream(file, { start, end }).pipe(response);
}

function serve(video, frames) {
  const server = http.createServer((request, response) => {
    const url = new URL(request.url, 'http://local');
    const frame = /^\/frame\/(\d+)\.png$/.exec(url.pathname);
    if (url.pathname === '/') {
      response.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
      response.end(page);
    } else if (url.pathname === '/video.mp4') sendFile(request, response, video, 'video/mp4');
    else if (frame) sendFile(request, response, path.join(frames, `frame-${frame[1].padStart(6, '0')}.png`), 'image/png');
    else {
      response.writeHead(404);
      response.end();
    }
  });
  return new Promise((resolve) => server.listen(0, '127.0.0.1', () => resolve(server)));
}

async function play() {
  const video = document.querySelector('video');
  await new Promise((resolve, reject) => {
    if (video.readyState >= 2) resolve();
    video.addEventListener('loadeddata', resolve, { once: true });
    video.addEventListener('error', () => reject(new Error(`media error ${video.error?.code}`)), { once: true });
  });
  const context = new AudioContext();
  const analyser = context.createAnalyser();
  analyser.fftSize = 2048;
  const silence = context.createGain();
  silence.gain.value = 0;
  context.createMediaElementSource(video).connect(analyser);
  analyser.connect(silence);
  silence.connect(context.destination);
  await context.resume();
  const samples = new Float32Array(analyser.fftSize);
  const callbacks = [];
  let watching = true;
  const tick = (now, meta) => {
    analyser.getFloatTimeDomainData(samples);
    let power = 0;
    for (const x of samples) power += x * x;
    callbacks.push({
      now,
      mediaTime: meta.mediaTime,
      presentedFrames: meta.presentedFrames,
      currentTime: video.currentTime,
      rms: Math.sqrt(power / samples.length),
    });
    if (watching) video.requestVideoFrameCallback(tick);
  };
  video.requestVideoFrameCallback(tick);
  const ended = new Promise((resolve, reject) => {
    video.addEventListener('ended', resolve, { once: true });
    video.addEventListener('error', () => reject(new Error('playback media error')), { once: true });
    setTimeout(() => reject(new Error('playback timed out')), (video.duration + 20) * 1000);
  });
  const started = performance.now();
  await video.play();
  await ended;
  const wallMs = performance.now() - started;
  watching = false;
  const quality = video.getVideoPlaybackQuality();

  // The file's own audio, decoded separately: running sums of the mono power
  // give the level over any 2048-sample window ending at a given time.
  const track = await context.decodeAudioData(await (await fetch('/video.mp4')).arrayBuffer());
  const channels = Array.from({ length: track.numberOfChannels }, (_, index) => track.getChannelData(index));
  const sums = new Float64Array(track.length + 1);
  for (let index = 0; index < track.length; index++) {
    let x = 0;
    for (const channel of channels) x += channel[index];
    x /= channels.length;
    sums[index + 1] = sums[index] + x * x;
  }
  const level = (seconds) => {
    const end = Math.min(track.length, Math.max(analyser.fftSize, Math.round(seconds * track.sampleRate)));
    return Math.sqrt((sums[end] - sums[end - analyser.fftSize]) / analyser.fftSize);
  };
  const inside = callbacks.filter((item) => item.currentTime > 0.5 && item.currentTime < video.duration - 0.5);
  const pearson = (a, b) => {
    const mean = (list) => list.reduce((sum, x) => sum + x, 0) / list.length;
    const ma = mean(a);
    const mb = mean(b);
    let ab = 0;
    let aa = 0;
    let bb = 0;
    for (let index = 0; index < a.length; index++) {
      ab += (a[index] - ma) * (b[index] - mb);
      aa += (a[index] - ma) ** 2;
      bb += (b[index] - mb) ** 2;
    }
    return ab / Math.sqrt(aa * bb);
  };
  const heard = inside.map((item) => item.rms);
  let best = { lagMs: 0, correlation: -2 };
  for (let lag = -250; lag <= 250; lag++) {
    const correlation = pearson(heard, inside.map((item) => level(item.currentTime + lag / 1000)));
    if (correlation > best.correlation) best = { lagMs: lag, correlation };
  }
  const atZero = pearson(heard, inside.map((item) => level(item.currentTime)));
  await context.close();
  return {
    duration: video.duration,
    videoWidth: video.videoWidth,
    videoHeight: video.videoHeight,
    ended: video.ended,
    wallMs,
    quality: { totalVideoFrames: quality.totalVideoFrames, droppedVideoFrames: quality.droppedVideoFrames },
    callbacks,
    audioTrack: { sampleRate: track.sampleRate, channels: track.numberOfChannels, seconds: track.duration },
    envelope: { callbacks: inside.length, correlationAtZero: atZero, best },
  };
}

async function compare({ frame, fps, keep }) {
  const video = document.querySelector('video');
  const shown = new Promise((resolve) => video.requestVideoFrameCallback((now, meta) => resolve(meta.mediaTime)));
  video.currentTime = (frame + 0.5) / fps;
  await new Promise((resolve) => video.addEventListener('seeked', resolve, { once: true }));
  const mediaTime = await shown;
  const draw = (source) => {
    const canvas = document.createElement('canvas');
    canvas.width = 1920;
    canvas.height = 1080;
    const context = canvas.getContext('2d', { willReadFrequently: true });
    context.drawImage(source, 0, 0, 1920, 1080);
    return { canvas, pixels: context.getImageData(0, 0, 1920, 1080).data };
  };
  const decoded = draw(video);
  const image = new Image();
  image.src = `/frame/${frame}.png`;
  await image.decode();
  const rendered = draw(image).pixels;
  const signed = [0, 0, 0];
  const absolute = [0, 0, 0];
  const histogram = new Array(256).fill(0);
  let squares = 0;
  for (let index = 0; index < rendered.length; index += 4) {
    let worst = 0;
    for (let channel = 0; channel < 3; channel++) {
      const difference = decoded.pixels[index + channel] - rendered[index + channel];
      signed[channel] += difference;
      absolute[channel] += Math.abs(difference);
      squares += difference * difference;
      worst = Math.max(worst, Math.abs(difference));
    }
    histogram[worst]++;
  }
  const count = rendered.length / 4;
  let seen = 0;
  let p99 = 0;
  while (seen < 0.99 * count) seen += histogram[p99++];
  const mse = squares / (count * 3);
  return {
    frame,
    mediaTime,
    psnr: mse ? 10 * Math.log10((255 * 255) / mse) : null,
    meanSigned: signed.map((value) => value / count),
    meanAbsolute: absolute.map((value) => value / count),
    p99MaxChannel: p99 - 1,
    capture: keep ? decoded.canvas.toDataURL('image/png') : null,
  };
}

const options = args();
if (fs.existsSync(options.out)) throw new Error(`output exists; use a new file: ${options.out}`);
const fps = Number(options.fps || 30);
const server = await serve(path.resolve(options.video), path.resolve(options.frames));
const browser = await chromium.launch({
  headless: true,
  executablePath: options.chrome,
  args: ['--autoplay-policy=no-user-gesture-required', '--force-color-profile=srgb',
    ...(options.flags ? options.flags.split(' ') : [])],
});
try {
  const context = await browser.newContext({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1 });
  const tab = await context.newPage();
  const errors = [];
  tab.on('pageerror', (error) => errors.push(error.message));
  await tab.goto(`http://127.0.0.1:${server.address().port}/`, { waitUntil: 'load' });
  const playback = await tab.evaluate(play);
  const frames = [];
  if (options.captures) fs.mkdirSync(options.captures, { recursive: true });
  for (const frame of options.at.split(',').map(Number)) {
    const { capture, ...result } = await tab.evaluate(compare, { frame, fps, keep: Boolean(options.captures) });
    if (capture) {
      const file = path.join(options.captures, `chrome-${String(frame).padStart(6, '0')}.png`);
      fs.writeFileSync(file, Buffer.from(capture.split(',')[1], 'base64'));
    }
    frames.push(result);
  }
  const times = playback.callbacks.map((item) => item.mediaTime);
  const steps = times.slice(1).map((time, index) => Math.round((time - times[index]) * fps));
  const report = {
    chrome: browser.version(),
    extraFlags: options.flags || null,
    video: path.basename(options.video),
    duration: playback.duration,
    size: [playback.videoWidth, playback.videoHeight],
    ended: playback.ended,
    wallSeconds: playback.wallMs / 1000,
    playbackQuality: playback.quality,
    frameCallbacks: {
      count: times.length,
      distinctMediaTimes: new Set(times).size,
      firstMediaTime: times[0],
      lastMediaTime: times.at(-1),
      // Frames between consecutive callbacks: 1 everywhere means none was skipped.
      stepHistogram: steps.reduce((counts, step) => ({ ...counts, [step]: (counts[step] || 0) + 1 }), {}),
      // [last frame shown before the gap, frames advanced, wall ms since the previous callback]
      skips: steps.flatMap((step, index) => (step > 1
        ? [[Math.round(times[index] * fps), step, Math.round(playback.callbacks[index + 1].now - playback.callbacks[index].now)]]
        : [])),
    },
    audio: {
      track: playback.audioTrack,
      callbacksWithSignal: playback.callbacks.filter((item) => item.rms > 1e-4).length,
      envelope: playback.envelope,
    },
    frames,
    pageErrors: errors,
  };
  fs.writeFileSync(options.out, JSON.stringify(report, null, 2) + '\n', 'utf8');
  console.log(JSON.stringify({ ...report, frameCallbacks: { ...report.frameCallbacks, skips: report.frameCallbacks.skips.length } }));
} finally {
  await browser.close();
  server.close();
}
