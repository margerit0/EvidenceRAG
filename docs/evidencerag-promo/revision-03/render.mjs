process.stdin.setEncoding('utf8');
import {chromium} from "../../../frontend/node_modules/playwright/index.mjs";
import path from "node:path";
let input='';for await(const c of process.stdin)input+=c;const p=JSON.parse(input);
const browser=await chromium.launch({headless:true,args:[`--log-file=${path.join(p.outputDir,'chromium-debug.log')}`]});
const context=await browser.newContext({viewport:{width:1920,height:1080},deviceScaleFactor:1,locale:'zh-CN',offline:true});
await context.route('**/*',r=>r.abort());
const page=await context.newPage();const errors=[];page.on('pageerror',e=>errors.push(e.message));
await page.setContent(p.html,{waitUntil:'load'});await page.evaluate(async()=>{await window.previewReady;await document.fonts.ready});
const frames=[],issues=[],images=[];
for(const f of p.frames){
 await page.evaluate(f=>window.renderFrame(f),f.frame);
 const qa=await page.evaluate(f=>{
  const iframe=document.querySelector('iframe');const doc=iframe?iframe.contentDocument:document;const offset=iframe?40:0;
  const rect=e=>{const r=e.getBoundingClientRect();return{x:r.x,y:r.y+offset,w:r.width,h:r.height,right:r.right,bottom:r.bottom+offset}};
  const visible=e=>{for(let a=e;a&&a!==doc.body;a=a.parentElement){const s=getComputedStyle(a);if(s.display==='none'||s.visibility==='hidden'||Number(s.opacity)<.02)return false}return true};
  const measured=[...doc.querySelectorAll('[data-check]')].filter(visible).map(e=>{const r=rect(e);const c=getComputedStyle(e);const block=!['inline','inline-block'].includes(c.display);return{name:e.dataset.check,...r,fontSize:c.fontSize,outside:r.x<0||r.y<0||r.right>1920.5||r.bottom>1040.5,overflow:block&&(e.scrollWidth>e.clientWidth+2||e.scrollHeight>e.clientHeight+2)}});
  const tip=document.querySelector('[data-cursor-tip]');const target=f===45?doc.querySelector('[data-click-target="run"]'):f===210?doc.querySelector('[data-citation="2"]'):f===135?doc.querySelector('[data-inspect-target]'):null;
  let hit=null;if(target){const r=rect(target);const[x,y]=tip.dataset.cursorTip.split(',').map(Number);hit={x,y,inside:x>=r.x&&x<=r.right&&y>=r.y&&y<=r.bottom}}
  const metrics=Object.fromEntries(['.app-shell','.conversation-pane','.execution-pane','.inspector-slot','.page-heading h1','.submitted-query','.answer-body > p','.pane-heading > span:first-child'].map(s=>{const e=doc.querySelector(s);return[s,e?{...rect(e),fontSize:getComputedStyle(e).fontSize}:null]}));
  const graphCanvas=doc.querySelector('.graph-canvas');
  let graph=null;
  if(graphCanvas){
   const canvas=rect(graphCanvas),pane=rect(doc.querySelector('.execution-pane')),caption=rect(doc.querySelector('.graph-caption'));
   const nodes=[...doc.querySelectorAll('.react-flow__node-stage')].map(e=>({id:e.dataset.id,title:e.querySelector('.node-heading > span:nth-child(2)').textContent,state:e.querySelector('.node-meta').textContent,inspection:e.querySelector('.node-inspection').textContent.trim(),...rect(e)}));
   const definitions={'agent-search':['agent','out','search','in'],'agent-read':['agent','out','read','in'],'search-agent':['search','loop-left','agent','return-left'],'read-agent':['read','loop-right','agent','return-right'],'agent-finish':['agent','out','finish','in']};
   const edges=[...doc.querySelectorAll('.react-flow__edge')].map(edge=>{
    const id=edge.dataset.id,path=edge.querySelector('.react-flow__edge-path'),length=path.getTotalLength(),matrix=path.getScreenCTM();
    const points=Array.from({length:41},(_,i)=>{const q=path.getPointAtLength(length*i/40).matrixTransform(matrix);return{x:q.x,y:q.y+offset}});
    const [source,sourceHandle,target,targetHandle]=definitions[id];
    const sr=rect(doc.querySelector(`.react-flow__node[data-id="${source}"] [data-handleid="${sourceHandle}"].source`));
    const tr=rect(doc.querySelector(`.react-flow__node[data-id="${target}"] [data-handleid="${targetHandle}"].target`));
    const distance=(q,r)=>Math.hypot(q.x-(r.x+r.w/2),q.y-(r.y+r.h/2));
    const marker=path.getAttribute('marker-end'),markerId=marker?.match(/^url\(["']?#(.+?)["']?\)$/)?.[1];
    return{id,length,source,target,sourceError:distance(points[0],sr),targetError:distance(points.at(-1),tr),markerExists:!!doc.getElementById(markerId),clipped:points.some(q=>q.x<canvas.x||q.x>canvas.right||q.y<canvas.y||q.y>canvas.bottom)};
   });
   graph={nodes,edges,canvas,crop:{x:pane.x,y:pane.y,width:pane.w,height:caption.bottom-pane.y},selected:doc.querySelector('.react-flow__node.selected')?.dataset.id??null};
  }
  return{frame:f,measured,hit,metrics,graph,bodyText:doc.body.innerText,sourceText:doc.querySelector('.evidence-text mark')?.textContent??null};
 },f.frame);
 const png=await page.screenshot({path:path.join(p.outputDir,f.filename)});
 if(f.frame===195)await page.screenshot({path:path.join(p.outputDir,'03-graph-detail.png'),clip:qa.graph.crop});
 images.push({frame:f.frame,data:'data:image/png;base64,'+png.toString('base64')});
 for(const m of qa.measured)if(m.outside||m.overflow)issues.push({frame:f.frame,...m});
 if(qa.hit&&!qa.hit.inside)issues.push({frame:f.frame,cursor:'missed'});
 if(qa.graph){
  for(const edge of qa.graph.edges)if(edge.length<1||edge.clipped||!edge.markerExists||edge.sourceError>3||edge.targetError>3)issues.push({frame:f.frame,edge});
  if(qa.graph.nodes.length!==4||qa.graph.edges.length!==5)issues.push({frame:f.frame,graph:'node/edge count'});
  for(const node of qa.graph.nodes)if(node.x<qa.graph.canvas.x||node.right>qa.graph.canvas.right||node.y<qa.graph.canvas.y||node.bottom>qa.graph.canvas.bottom)issues.push({frame:f.frame,node:'clipped',id:node.id});
 }
 frames.push(qa);
}
const esc=s=>s.replaceAll('&','&amp;').replaceAll('<','&lt;');
const markup=`<!doctype html><meta charset="utf-8"><style>@font-face{font-family:Inter;src:url('${p.font}');font-weight:100 900}*{box-sizing:border-box}body{width:1920px;margin:0;padding:44px 48px;background:#e6eff4;color:#263845;font-family:Inter,'Microsoft YaHei',sans-serif}.heading{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:32px}h1{margin:0;font-size:36px;font-weight:600}p{color:#536b7c;font-size:19px;line-height:1.6}.badge{border:1px solid #a9c0d0;border-radius:24px;padding:12px 20px;font-size:20px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:24px}.tile{border-radius:22px;overflow:hidden;background:#f6fbfd;border:1px solid white}.tile img{display:block;width:100%;aspect-ratio:16/9}.caption{padding:20px 24px}.caption h2{display:flex;margin:0;font-size:24px;font-weight:600}.caption h2 span{margin-left:auto;font-size:20px;color:#527b96}.caption p{margin:10px 0 0;font-size:18px}.notes{padding:32px}.notes h2{font-size:29px;margin:0 0 24px}.notes p{font-size:22px;line-height:1.8}.notes hr{border:0;border-top:1px solid #ccdbe4;margin:25px 0}.notes small{font-size:18px;line-height:1.8;color:#536b7c}.foot{margin:25px 0 0;font-size:17px;color:#536b7c}</style><div class="heading"><div><h1>EvidenceRAG / 浏览器原比例分镜</h1><p>1920 × 1080 成片 · 1920 × 1000 网页视口居中 · 100% CSS 缩放</p></div><div class="badge">修订预览 · 待确认</div></div><div class="grid">${p.frames.slice(0,5).map((f,i)=>`<article class="tile"><img src="${images[i].data}"><div class="caption"><h2>0${i+1} · ${f.title}<span>${f.time}</span></h2><p>主要动作：${f.action}</p><p>转场：${f.transition}</p></div></article>`).join('')}<aside class="tile notes"><h2>恢复原生 Agent 流程图</h2><p>上方：Agent 决策<br>中间：搜索文档 / 读取证据<br>下方：校验与输出<br>五条连线保留工具返回路径</p><hr><p>调用顺序留在时间线</p><small>4.5 秒点击第二次读取的时间线记录，展开「步骤」。<br>7 秒点击 [2]，同一检查器切到「证据」。<br>保持浏览器正常字号与三栏比例。</small><hr><small>直接复用前端节点和 SVG 连线；不另画顺序图。<br>五张图片均为代码实渲染，当前未更新 MP4。</small></aside></div><div class="foot">合成演示 · 流程已压缩　/　[2] 与「升级演练 / 回退条件」逐字对应。</div>`;
await page.setContent(markup,{waitUntil:'load'});await page.evaluate(async()=>{await document.fonts.ready;await Promise.all([...document.images].map(i=>i.decode()))});
await page.screenshot({path:path.join(p.outputDir,'storyboard-overview.png'),fullPage:true});
await browser.close();console.log(JSON.stringify({errors,issues,frames,images:images.map(x=>({frame:x.frame})),overview:path.join(p.outputDir,'storyboard-overview.png')}));
