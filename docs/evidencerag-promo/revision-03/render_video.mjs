process.stdin.setEncoding('utf8');
import { chromium } from '../../../frontend/node_modules/playwright/index.mjs';
import path from 'node:path';
import crypto from 'node:crypto';
let input='';for await(const c of process.stdin)input+=c;const p=JSON.parse(input);
if(p.html.includes('\uFFFD'))throw new Error('Invalid replacement character in render input');
const browser=await chromium.launch({headless:true,args:[`--log-file=${path.join(p.outputDir,'chromium-debug.log')}`]});
const context=await browser.newContext({viewport:{width:1920,height:1080},deviceScaleFactor:1,locale:'zh-CN',colorScheme:'light',offline:true});
await context.route('**/*',r=>r.abort());
const page=await context.newPage(),errors=[],samples=[],bounds=[],hashes=[];
page.on('pageerror',e=>errors.push(e.message));
await page.setContent(p.html,{waitUntil:'load'});await page.evaluate(async()=>{await window.movieReady;await document.fonts.ready;});
for(let i=0;i<p.frames.length;i++){
 const f=p.frames[i];
 const qa=await page.evaluate(f=>{
  window.renderMovieFrame(f);
  const iframe=document.querySelector('iframe'),doc=f<300?iframe.contentDocument:document;
  const offset=f<300?40:0;
  const rect=e=>{const r=e.getBoundingClientRect();return{x:r.x,y:r.y+offset,w:r.width,h:r.height,right:r.right,bottom:r.bottom+offset}};
  const visible=e=>{for(let a=e;a;a=a.parentElement){const s=getComputedStyle(a);if(s.display==='none'||s.visibility==='hidden'||Number(s.opacity)<.02)return false;}return e.getBoundingClientRect().width>0};
  const selector=f<300?'[data-check]':'.movie-brand [data-check]';
  const checks=[...doc.querySelectorAll(selector)].filter(visible).filter(e=>!(f<149&&e.closest('.inspector-slot')));
  const bad=checks.map(e=>{
   const r=rect(e),s=getComputedStyle(e),block=!['inline','inline-block'].includes(s.display);
   let overflow=block&&(e.scrollWidth>e.clientWidth+2||e.scrollHeight>e.clientHeight+2);
   if(e.classList.contains('node-heading')){
    // Rotated 13px SVGs can extend a few pixels past their flex item; the native
    // node has 10px padding for this. Check painted SVG and text against the card.
    const card=rect(e.closest('.flow-node'));
    const outside=q=>q.x<card.x||q.y<card.y||q.right>card.right||q.bottom>card.bottom;
    overflow=[...e.querySelectorAll('svg')].some(svg=>outside(rect(svg)));
    const walker=doc.createTreeWalker(e,NodeFilter.SHOW_TEXT);
    for(let t=walker.nextNode();t;t=walker.nextNode())if(t.textContent.trim()){
     const range=doc.createRange();range.selectNodeContents(t);
     for(const q of range.getClientRects())if(outside({x:q.x,y:q.y+offset,right:q.right,bottom:q.bottom+offset}))overflow=true;
    }
   }
   return{name:e.dataset.check,...r,overflow};
  }).filter(r=>r.x<-.5||r.y<-.5||r.right>1920.5||r.bottom>1080.5||r.overflow);
  const tip=document.querySelector('[data-cursor-tip]');let hit=null;
  if(window.motionState.click){const target=doc.querySelector(window.motionState.click==='run'?'[data-click-target="run"]':window.motionState.click==='inspect'?'[data-inspect-target]':'[data-citation="2"]'),r=rect(target);const[x,y]=tip.dataset.cursorTip.split(',').map(Number);hit={kind:window.motionState.click,x,y,bounds:r,inside:x>=r.x&&x<=r.right&&y>=r.y&&y<=r.bottom};}
  let graph=null;
  if(f<300){
   const canvas=rect(doc.querySelector('.graph-canvas'));
   const definitions={'agent-search':['agent','out','search','in'],'agent-read':['agent','out','read','in'],'search-agent':['search','loop-left','agent','return-left'],'read-agent':['read','loop-right','agent','return-right'],'agent-finish':['agent','out','finish','in']};
   const nodes=[...doc.querySelectorAll('.react-flow__node-stage')].map(e=>({id:e.dataset.id,status:e.querySelector('.flow-node').className,...rect(e)}));
   const edges=[...doc.querySelectorAll('.react-flow__edge')].map(e=>{
    const id=e.dataset.id,path=e.querySelector('.react-flow__edge-path'),len=path.getTotalLength(),m=path.getScreenCTM();
    const pts=Array.from({length:25},(_,i)=>{const q=path.getPointAtLength(len*i/24).matrixTransform(m);return{x:q.x,y:q.y+40}});
    const [src,sh,dst,dh]=definitions[id];
    const sr=rect(doc.querySelector(`.react-flow__node[data-id="${src}"] [data-handleid="${sh}"].source`));
    const tr=rect(doc.querySelector(`.react-flow__node[data-id="${dst}"] [data-handleid="${dh}"].target`));
    const dist=(q,r)=>Math.hypot(q.x-r.x-r.w/2,q.y-r.y-r.h/2);
    const marker=path.getAttribute('marker-end')?.match(/^url\(["']?#(.+?)["']?\)$/)?.[1];
    return{id,length:len,marker:!!doc.getElementById(marker),sourceError:dist(pts[0],sr),targetError:dist(pts.at(-1),tr),clipped:pts.some(q=>q.x<canvas.x||q.x>canvas.right||q.y<canvas.y||q.y>canvas.bottom)};
   });
   const signals=[...doc.querySelectorAll('.graph-edge-signal')].filter(signal=>getComputedStyle(signal).display!=='none').map(signal=>({edge:signal.parentElement.dataset.id,pathMatches:signal.getAttribute('d')===signal.parentElement.querySelector('.react-flow__edge-path').getAttribute('d'),pathLength:signal.getAttribute('pathLength'),bounds:rect(signal),dasharray:getComputedStyle(signal).strokeDasharray,offset:parseFloat(getComputedStyle(signal).strokeDashoffset),strokeWidth:parseFloat(getComputedStyle(signal).strokeWidth),animationTime:signal.getAnimations()[0]?.currentTime,paused:signal.getAnimations()[0]?.playState==='paused'}));
   const spinners=[...doc.querySelectorAll('.flow-node.running .node-status svg')].map(svg=>({node:svg.closest('.react-flow__node').dataset.id,bounds:rect(svg.closest('.node-status')),transform:getComputedStyle(svg).transform,animationTime:svg.getAnimations()[0]?.currentTime,paused:svg.getAnimations()[0]?.playState==='paused'}));
   graph={canvas,motion:{...window.graphMotionState,signals,spinners},nodes:nodes.map(n=>({id:n.id,status:n.status})),edges,selected:doc.querySelector('.react-flow__node.selected')?.dataset.id??null,valid:nodes.length===4&&edges.length===5&&edges.every(e=>e.length>1&&e.marker&&!e.clipped&&e.sourceError<3&&e.targetError<3)&&nodes.every(n=>n.x>=canvas.x&&n.y>=canvas.y&&n.right<=canvas.right&&n.bottom<=canvas.bottom)};
  }
  const labelCount=[...iframe.contentDocument.querySelectorAll('.promo-demo-note')].filter(visible).length*(f<308?1:0)+[...document.querySelectorAll('.brand-badge')].filter(visible).length;
  return{...window.motionState,bad,hit,graph,answerTexts:[...doc.querySelectorAll('[data-check="answer-1"],[data-check="answer-2"]')].map(e=>e.textContent.trim()),replacementCharacters:doc.body.innerText.includes('\uFFFD'),sourceText:f>=210&&f<300?doc.querySelector('.evidence-text mark')?.textContent:null,labelCount,inspectorTab:f>=149&&f<300?doc.querySelector('.tabs-trigger[data-state="active"]')?.textContent:null,columnWidths:f>=149&&f<300?['.conversation-pane','.execution-pane','.inspector-slot'].map(s=>rect(doc.querySelector(s)).w):null};
 },f);
 const png=await page.screenshot({path:path.join(p.outputDir,`frame-${String(f).padStart(4,'0')}.png`)});
 hashes.push({frame:f,hash:crypto.createHash('sha256').update(png).digest('hex')});samples.push(qa);
 if(qa.bad.length)bounds.push({frame:f,transition:false,items:qa.bad});
 if(i%30===0)console.log(JSON.stringify({progress:i+1,total:p.frames.length,frame:f}));
}
await browser.close();console.log(JSON.stringify({done:true,rendered:p.frames.length,errors,samples,bounds,hashes}));
