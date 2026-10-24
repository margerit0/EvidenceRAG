// Deterministic 30 fps timeline. Native DOM and SVG routes stay mounted throughout.
const clamp01=x=>Math.max(0,Math.min(1,x));
const ease=(f,a,b)=>{const t=clamp01((f-a)/(b-a));return t*t*t*(10+t*(-15+6*t));};
const settle=(f,a,b)=>{const t=clamp01((f-a)/(b-a));if(t===0||t===1)return t;return 1-Math.exp(-8*t)*(Math.cos(8*t)+Math.sin(8*t));};
const mix=(a,b,t)=>a+(b-a)*t;
const movieTemplates=Object.fromEntries(['question','investigation','inspection','answer','source'].map(k=>[k,new DOMParser().parseFromString(createDoc(k),'text/html')]));
const movieFrames={question:0,search1:60,read1:75,return1:90,investigation:105,read2:120,inspection:135,answer:150,source:210,brand:300};
let movieDoc,movieIframe,movieBrand,moviePointer,movieRing,lastScene=null;
let runPoint,inspectPoint,citePoint;
const pointFor=selector=>{const r=movieDoc.querySelector(selector).getBoundingClientRect();return{x:r.x+r.width*.7,y:r.y+40+r.height*.62};};
function cursorAt(f,point,clickFrame){
 moviePointer.style.left=point.x+'px';moviePointer.style.top=point.y+'px';moviePointer.dataset.cursorTip=`${point.x},${point.y}`;
 const pressed=clickFrame!==null&&f>=clickFrame&&f<clickFrame+6;
 moviePointer.style.transform=`scale(${pressed?1-.08*Math.sin(Math.PI*(f-clickFrame+1)/7):1})`;
 const t=pressed?(f-clickFrame)/6:1;
 movieRing.style.left=(point.x-13)+'px';movieRing.style.top=(point.y-13)+'px';
 movieRing.style.opacity=pressed?String(.8*(1-t)):'0';movieRing.style.transform=`scale(${1+t*.9})`;
}
function changeScene(kind){
 if(kind===lastScene)return;
 const template=movieTemplates[kind];
 movieDoc.body.className='promo-'+kind;
 movieDoc.querySelector('.conversation-pane').innerHTML=template.querySelector('.conversation-pane').innerHTML;
 movieDoc.querySelector('.inspector-slot').innerHTML=template.querySelector('.inspector-slot').innerHTML;
 movieDoc.querySelector('.page-heading h1').textContent=template.querySelector('.page-heading h1').textContent;
 lastScene=kind;
}
window.renderMovieFrame=f=>{
 f=Math.max(0,Math.min(359,Number(f)));
 // Hold frames have no DOM mutations, so SVG/text rasterization stays unchanged.
 const previous=window.motionState?.frame;
 if((f>=225&&f<300&&previous>=225&&previous<300)||(f>=330&&previous>=330)){
  stage.dataset.frame=f;window.currentFrame=f;window.motionState={...window.motionState,frame:f};return;
 }
 const scene=f<60?'question':f<135?'investigation':f<150?'inspection':f<210?'answer':'source';
 changeScene(scene);
 const $=s=>movieDoc.querySelector(s),all=s=>[...movieDoc.querySelectorAll(s)];
 movieDoc.querySelector('.app-shell').style.opacity='1';
 movieIframe.style.opacity='1';movieIframe.style.visibility=f<315?'visible':'hidden';
 movieIframe.style.transform='none';movieIframe.style.clipPath='none';
 movieDoc.body.style.background='';
 const opening=settle(f,135,149),bench=$('.workbench');
 bench.classList.remove('inspector-collapsed');
 bench.style.gridTemplateColumns=`minmax(290px,1fr) minmax(390px,1.3fr) 32px ${500*opening}px`;
 $('.inspector-slot').style.width='auto';
 const aside=$('.inspector-desktop');
 aside.style.width='500px';aside.style.visibility=opening>0?'visible':'hidden';
 aside.style.opacity=String(ease(f,135,142));aside.style.transform='none';
 const toggle=$('.inspector-toggle');toggle.setAttribute('aria-expanded',String(f>=135));
 $('.execution-pane').style.opacity=String(f<60?.62:f>=210?mix(1,.72,ease(f,210,225)):mix(.62,1,ease(f,60,72)));
 $('.current-action').classList.toggle('is-running',f>=60&&f<135);
 const stateKey=GRAPH_BEATS.findLast(([start])=>f>=start)[1];
 restoreGraph(movieDoc,stateKey);fitNativeGraph(movieDoc);
 animateNativeGraph(movieDoc,f);
 const current=f<45?'等待发起调查':f<60?'Agent 决策':f<75?'正在搜索文档':f<90?'正在读取证据':f<105?'准备补充搜索':f<120?'正在补充搜索':f<135?'正在读取证据':f<150?'正在校验答案':'已生成答案';
 $('.current-action').innerHTML=icon(f>=60&&f<75||f>=105&&f<120?'search':'file',16)+`<span>${current}</span>`;
 const count=f<60?0:f<75?1:f<105?2:f<120?3:4;
 const active=f<60?-1:f<75?0:f<90?1:f<105?-1:f<120?2:f<135?3:-1;
 const inspected=f>=135&&f<210;
 $('.timeline-heading').innerHTML=`<span>执行时间线 ${count?`<span class="count-badge">${count}</span>`:''}</span><span style="margin-left:auto;font-size:10px">${f>=135?'调查记录':'自动执行'}</span>`;
 $('.timeline').innerHTML=count?labels.slice(0,count).map((s,i)=>`<button class="promo-timeline-row ${i===active?'active':''} ${i===3&&inspected?'selected':''}" ${i===3?'data-inspect-target="true"':''} data-check="timeline-${i+1}"><span class="ordinal">${String(i+1).padStart(2,'0')}</span>${icon(i===active?'search':'check',13)}<span>${s} · 第 ${i<2?1:2} 次</span><span class="result">${i===active?'执行中':i===3&&inspected?'正在查看':'已返回'}</span>${icon('right',12)}</button>`).join(''):'<div class="promo-waiting" style="padding:0 20px">运行演示后，查看资料查找过程。</div>';
 // Reveal the submitted question locally; the graph and shell remain stable.
 const scroll=$('.conversation-scroll');scroll.style.transform='none';scroll.style.opacity='1';
 if(f>=60&&f<75){scroll.style.transform=`translateY(${18*(1-ease(f,60,74))}px)`;scroll.style.opacity=String(ease(f,60,70));}
 const run=$('[data-click-target="run"]');
 if(run){runPoint=pointFor('[data-click-target="run"]');run.style.transform=`translateY(${f>=45&&f<51?1.4*Math.sin(Math.PI*(f-44)/7):0}px)`;}
 if(f>=120&&f<=135)inspectPoint=pointFor('[data-inspect-target]');
 if(f>=150){
  all('.answer-body > p').forEach((p,i)=>{const progress=ease(f,150+i*15,164+i*15);p.style.opacity=String(progress);p.style.clipPath=`inset(0 ${100*(1-progress)}% 0 0)`;});
  if(f<210)citePoint=pointFor('[data-citation="2"]');
 }
 if(f>=210){
  const p=$('.selected-summary'),selection=ease(f,210,222);
  p.style.padding=`${8*selection}px ${10*selection}px`;p.style.marginLeft=(-10*selection)+'px';p.style.marginRight=(-10*selection)+'px';
  const tab=$('.tabs-content[data-state="active"]');tab.style.opacity=String(ease(f,210,219));tab.style.transform=`translateX(${18*(1-ease(f,210,224))}px)`;
 }
 const parked={x:1840,y:973};let cursor=parked,clickFrame=null;
 if(f<60){const t=ease(f,10,37);cursor={x:mix(1030,runPoint.x,t),y:mix(830,runPoint.y,t)};clickFrame=45;
  if(f>51){const q=ease(f,51,60);cursor={x:mix(runPoint.x,parked.x,q),y:mix(runPoint.y,parked.y,q)};}
 }else if(f>=120&&f<156){const t=ease(f,121,130);cursor={x:mix(parked.x,inspectPoint.x,t),y:mix(parked.y,inspectPoint.y,t)};clickFrame=135;
  if(f>=136){const q=ease(f,136,155);cursor={x:mix(inspectPoint.x,parked.x,q),y:mix(inspectPoint.y,parked.y,q)};}
 }else if(f>=180&&f<225){const t=ease(f,180,203);cursor={x:mix(parked.x,citePoint.x,t),y:mix(parked.y,citePoint.y,t)};clickFrame=210;
  if(f>211){const q=ease(f,211,224);cursor={x:mix(citePoint.x,citePoint.x+50,q),y:mix(citePoint.y,citePoint.y+50,q)};}
 }else if(f>=225){cursor={x:citePoint.x+50,y:citePoint.y+50};}
 moviePointer.style.visibility=f<300?'visible':'hidden';movieRing.style.visibility=f<300?'visible':'hidden';cursorAt(f,cursor,clickFrame);
 // Gather the complete workbench into the approved recap before revealing the brand.
 const closing=settle(f,300,315),recap=movieBrand.querySelector('.recap');
 movieBrand.style.visibility=f>=300?'visible':'hidden';
 stage.querySelector(':scope > .stage-wallpaper').style.opacity=String(1-ease(f,300,315));
 movieBrand.querySelector('.brand-wallpaper').style.opacity=String(.4*ease(f,300,315));
 movieBrand.querySelector('.brand-badge').style.opacity=String(ease(f,300,309));
 recap.style.opacity=String(ease(f,308,315));
 recap.style.transform=`translate(${40*(1-closing)}px,${12*(1-closing)}px)`;
 recap.style.borderRadius=`${mix(18,24,closing)}px`;
 const brandProgress=ease(f,315,330);
 for(const el of movieBrand.querySelectorAll('.brand-copy,.brand-footer')){el.style.opacity=String(brandProgress);el.style.transform=`translateY(${16*(1-brandProgress)}px)`;}
 if(f>=300&&f<315){
  const r=bench.getBoundingClientRect(),sx=mix(1,910/r.width,closing),sy=mix(1,430/r.height,closing);
  movieIframe.style.transformOrigin='0 0';
  movieIframe.style.transform=`translate(${mix(0,880-r.x*sx,closing)}px,${mix(0,330-40-r.y*sy,closing)}px) scale(${sx},${sy})`;
  movieIframe.style.clipPath=`inset(${r.y*ease(f,300,306)}px ${(1920-r.right)*ease(f,300,306)}px ${(1000-r.bottom)*ease(f,300,306)}px ${r.x*ease(f,300,306)}px round ${24*closing}px)`;
  movieIframe.style.opacity=String(1-ease(f,308,315));
 }
 stage.dataset.frame=f;stage.dataset.scene=f>=300?'brand':scene;
 window.currentFrame=f;
 window.motionState={frame:f,scene:stage.dataset.scene,graphState:stateKey,click:[45,135,210].includes(f)?({45:'run',135:'inspect',210:'citation'})[f]:null,hold:f>=225&&f<300?'source':f>=330?'brand':null};
};
window.movieReady=(async()=>{
 await window.previewReady;
 stage.innerHTML=`<div class="stage-wallpaper">${D.demo.glacierSvg}</div><div class="movie-brand" style="position:absolute;inset:0;z-index:2">${brand()}</div>`;
 movieBrand=stage.querySelector('.movie-brand');
 movieIframe=document.createElement('iframe');movieIframe.id='frontend';movieIframe.title='EvidenceRAG 原比例工作台';
 const ready=new Promise(resolve=>movieIframe.onload=resolve);movieIframe.srcdoc=createDoc('answer');stage.insertBefore(movieIframe,movieBrand);await ready;
 movieDoc=movieIframe.contentDocument;await movieDoc.fonts.ready;
 const css=movieDoc.createElement('style');css.textContent='.inspector-slot{border-radius:24px}.answer-body>p{will-change:opacity,clip-path}';movieDoc.head.append(css);
 prepareNativeGraphMotion(movieDoc);
 pointer(1840,973,false);moviePointer=stage.querySelector('.cursor');
 stage.insertAdjacentHTML('beforeend','<span class="click-ring" style="opacity:0"></span>');movieRing=stage.querySelector('.click-ring');
 // Cache targets in both end layouts for direct seeks as well as sequential renders.
 window.renderMovieFrame(0);window.renderMovieFrame(120);window.renderMovieFrame(195);window.renderMovieFrame(0);
 window.renderFrame=async f=>{await window.movieReady;window.renderMovieFrame(f);};
})();
window.addEventListener('keydown',e=>{if(e.code==='Space'){e.preventDefault();window.playMovie?.();}});
window.playMovie=async()=>{await window.movieReady;const begin=performance.now();const tick=now=>{const f=Math.min(359,Math.floor((now-begin)*.03));window.renderMovieFrame(f);if(f<359)requestAnimationFrame(tick);};requestAnimationFrame(tick);};
