process.stdin.setEncoding('utf8');
import {chromium} from "../../../frontend/node_modules/playwright/index.mjs";
import path from "node:path";
let input="";for await(const c of process.stdin)input+=c;
const p=JSON.parse(input);
const browser=await chromium.launch({headless:true,args:[`--log-file=${path.join(p.outputDir,"chromium-debug.log")}`]});
const context=await browser.newContext({viewport:{width:1920,height:1000},deviceScaleFactor:1,locale:"zh-CN",reducedMotion:"reduce"});
await context.route('**/api/**',r=>r.abort());
const page=await context.newPage();const errors=[];page.on("pageerror",e=>errors.push(e.message));
await page.goto('http://127.0.0.1:4176/workbench/',{waitUntil:'networkidle'});
await page.evaluate(async()=>{document.documentElement.dataset.theme='light';await document.fonts.ready});
const snapshots={};
async function capture(name){
  snapshots[name]=await page.evaluate(()=>({html:document.querySelector('#root').innerHTML,metrics:Object.fromEntries(['.app-shell','.workbench','.conversation-pane','.execution-pane','.inspector-slot','.page-heading h1','.submitted-query','.answer-body > p','.pane-heading > span:first-child','.input-wrap textarea'].map(sel=>{const e=document.querySelector(sel);if(!e)return[sel,null];const r=e.getBoundingClientRect();return[sel,{x:r.x,y:r.y,width:r.width,height:r.height,fontSize:getComputedStyle(e).fontSize}]}))}));
  await page.screenshot({path:path.join(p.outputDir,`native-${name}.png`),animations:'disabled'});
}
await capture('question');
await page.getByRole('button',{name:'运行演示',exact:true}).click();
await page.waitForFunction(()=>document.querySelector('.answer-status')?.textContent.includes('已生成答案'),{timeout:30000});
await page.locator('.timeline-row').filter({hasText:'读取证据'}).last().click();
await page.waitForFunction(()=>document.querySelector('.step-detail h2')?.textContent==='读取证据 · 第 2 次');
await page.evaluate(()=>new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r))));
await capture('answer');
await page.getByRole('button',{name:'查看引用 2',exact:true}).click();
await page.waitForFunction(()=>document.querySelector('.evidence-detail h3')?.textContent==='升级演练 / 回退条件');
await page.evaluate(()=>new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r))));
await capture('source');
const css=await page.evaluate(()=>[...document.styleSheets].map(s=>[...s.cssRules].map(r=>r.cssText).join('\n')).join('\n'));
await browser.close();console.log(JSON.stringify({snapshots,css,errors,viewport:{width:1920,height:1000,scale:1}}));
