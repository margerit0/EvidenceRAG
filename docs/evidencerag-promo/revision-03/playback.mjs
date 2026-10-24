process.stdin.setEncoding('utf8');
import { chromium } from "../../../frontend/node_modules/playwright/index.mjs";
let input = "";
for await (const c of process.stdin) input += c;
const p = JSON.parse(input);
const browser = await chromium.launch({
  channel: "chrome",
  headless: true,
  args: ["--autoplay-policy=no-user-gesture-required"],
});
const context = await browser.newContext({
  viewport: { width: 1920, height: 1080 },
  deviceScaleFactor: 1,
  offline: true,
});
const page = await context.newPage();
await page.setContent(
  '<style>body{margin:0;background:#e6eff4}video{display:block;width:1920px;height:1080px}</style><video id="movie" playsinline preload="auto"></video>',
);
await page.evaluate((src) => {
  document.querySelector("video").src = src;
}, "data:video/mp4;base64," + p.videoBase64);
const report = await page.evaluate(async () => {
  const v = document.querySelector("video");
  await new Promise((resolve, reject) => {
    if (v.readyState >= 2) return resolve();
    v.addEventListener("loadeddata", resolve, { once: true });
    v.addEventListener(
      "error",
      () => reject(new Error("Media error " + v.error?.code)),
      { once: true },
    );
  });
  const ctx = new AudioContext({ sampleRate: 48000 }),
    source = ctx.createMediaElementSource(v),
    analyser = ctx.createAnalyser();
  analyser.fftSize = 2048;
  source.connect(analyser);
  analyser.connect(ctx.destination);
  await ctx.resume();
  const buffer = new Float32Array(analyser.fftSize),
    audio = [],
    presented = [],
    thumbnails = [];
  const canvas = document.createElement("canvas");
  canvas.width = 640;
  canvas.height = 360;
  const g = canvas.getContext("2d");
  let nextSample = 0,
    active = true;
  function observe(now, metadata) {
    presented.push({
      time: metadata.mediaTime,
      frames: metadata.presentedFrames,
    });
    analyser.getFloatTimeDomainData(buffer);
    let power = 0,
      peak = 0;
    for (const x of buffer) {
      power += x * x;
      peak = Math.max(peak, Math.abs(x));
    }
    audio.push({
      time: v.currentTime,
      rms: Math.sqrt(power / buffer.length),
      peak,
    });
    if (metadata.mediaTime >= nextSample - 0.01) {
      g.drawImage(v, 0, 0, 640, 360);
      thumbnails.push({
        time: metadata.mediaTime,
        data: canvas.toDataURL("image/jpeg", 0.86),
      });
      nextSample += 0.5;
    }
    if (active) v.requestVideoFrameCallback(observe);
  }
  v.requestVideoFrameCallback(observe);
  v.volume = 1;
  v.muted = false;
  v.playbackRate = 1;
  const started = performance.now();
  const ended = new Promise((resolve, reject) => {
    v.addEventListener("ended", resolve, { once: true });
    v.addEventListener(
      "error",
      () => reject(new Error("Playback media error")),
      { once: true },
    );
    setTimeout(() => reject(new Error("Playback timed out")), 22000);
  });
  await v.play();
  await ended;
  active = false;
  const quality = v.getVideoPlaybackQuality();
  const result = {
    ended: v.ended,
    duration: v.duration,
    currentTime: v.currentTime,
    playbackRate: v.playbackRate,
    muted: v.muted,
    audioContextState: ctx.state,
    wallMs: performance.now() - started,
    totalVideoFrames: quality.totalVideoFrames,
    droppedVideoFrames: quality.droppedVideoFrames,
    presentedFrames: presented.length,
    presentationTimes: presented,
    audioAudits: audio,
    thumbnails,
    width: v.videoWidth,
    height: v.videoHeight,
  };
  await ctx.close();
  return result;
});
await browser.close();
console.log(JSON.stringify(report));
