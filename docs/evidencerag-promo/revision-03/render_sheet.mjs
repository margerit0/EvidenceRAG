process.stdin.setEncoding('utf8');
import { chromium } from "../../../frontend/node_modules/playwright/index.mjs";
let input = "";
for await (const c of process.stdin) input += c;
const p = JSON.parse(input);
const b = await chromium.launch({ headless: true });
const page = await b.newPage({
  viewport: { width: p.width || 1920, height: 1080 },
  deviceScaleFactor: 1,
});
await page.setContent(p.html);
await page.evaluate(async () => {
  await document.fonts.ready;
  await Promise.all([...document.images].map((x) => x.decode()));
});
await page.screenshot({ path: p.output, fullPage: true });
await b.close();
console.log(p.output);
