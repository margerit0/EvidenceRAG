const D=window.NATIVE_DATA;
const stage=document.getElementById('stage');
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const paths={check:'<path d="m5 12 4 4 10-10"/>',search:'<circle cx="10.5" cy="10.5" r="6.5"/><path d="m16 16 5 5"/>',file:'<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8Z"/><path d="M14 2v6h6M8 12h8M8 16h6"/>',right:'<path d="m9 5 7 7-7 7"/>',layers:'<path d="m16.02 12 5.48 3.13a1 1 0 0 1 0 1.74L13 21.74a2 2 0 0 1-2 0l-8.5-4.87a1 1 0 0 1 0-1.74L7.98 12"/><path d="M13 13.74a2 2 0 0 1-2 0L2.5 8.87a1 1 0 0 1 0-1.74L11 2.26a2 2 0 0 1 2 0l8.5 4.87a1 1 0 0 1 0 1.74Z"/>'};
const icon=(name,size=16)=>`<svg width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">${paths[name]}</svg>`;
const labels=['搜索文档','读取证据','补充搜索','读取证据'];
const summaries=['核对版本、配置与兼容性','验证升级演练与回退条件'];
// Keep the real React Flow DOM, SVG routes, handles, dimensions and icons.
// Only change the synthetic execution state, exactly as ExecutionGraph.tsx does.
function restoreGraph(doc, kind){
 const state=D.demo.graphStates[kind];
 doc.querySelectorAll('.react-flow__node-stage').forEach(wrapper=>{
  const id=wrapper.dataset.id,data=state.nodes[id],node=wrapper.querySelector('.flow-node');
  const inspected=!!data.inspection;
  wrapper.classList.toggle('selected',inspected);wrapper.style.zIndex=inspected?'1000':'0';
  node.className=`flow-node ${data.status} ${inspected?'is-selected':''}`;
  node.dataset.graphNode=id;
  for(const selector of ['.node-heading','.node-meta','.node-inspection'])node.querySelector(selector).dataset.check=`graph-${id}-${selector.slice(1)}`;
  const status=node.querySelector('.node-status');
  status.innerHTML=data.status==='done'?icon('check',13):data.status==='running'?'<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2a10 10 0 1 1-7.1 2.9"/></svg>':'<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-dasharray="3 3"><circle cx="12" cy="12" r="9"/></svg>';
  node.querySelector('.node-meta > span').textContent=data.status==='running'?'执行中':data.visits?(['search','read'].includes(id)?`最近第 ${data.visits} 次`:`${data.visits} 次调用`):'等待';
  const selection=node.querySelector('.node-inspection');
  selection.textContent=inspected?`正在查看${data.inspection}`:'\u00a0';selection.setAttribute('aria-hidden',String(!inspected));
  wrapper.setAttribute('aria-label',`${node.querySelector('.node-heading > span:nth-child(2)').textContent}，${data.status==='running'?'执行中':data.visits?`已调用 ${data.visits} 次`:'等待执行'}${inspected?`，正在查看${data.inspection}`:''}`);
 });
 doc.querySelectorAll('.react-flow__edge').forEach(edge=>{
  const flowing=edge.dataset.id===state.flowing,traversed=state.traversed.includes(edge.dataset.id);
  edge.classList.remove('is-flowing','is-traversed');edge.classList.toggle('is-flowing',flowing);edge.classList.toggle('is-traversed',traversed&&!flowing);
  const path=edge.querySelector('.react-flow__edge-path');
  path.style.stroke=flowing?'var(--brand)':traversed?'var(--graph-visited)':'var(--graph-line)';
  path.style.strokeWidth=flowing?'1.6':'1.15';path.style.opacity=flowing?'.55':traversed?'.85':'.7';
 });
}
function fitNativeGraph(doc){
 const canvas=doc.querySelector('.graph-canvas'),viewport=doc.querySelector('.react-flow__viewport');
 if(!canvas||!viewport)return;
 // Native stages occupy 412 x 316. The existing component caps fitting at 1x.
 const scale=Math.min(1,canvas.clientWidth/476,canvas.clientHeight/328);
 viewport.style.transform=`translate(${canvas.clientWidth/2-206*scale}px,${canvas.clientHeight/2-158*scale}px) scale(${scale})`;
}
function createDoc(kind){
 const n=kind==='question'?'question':kind==='source'?'source':'answer';
 const doc=new DOMParser().parseFromString(`<!doctype html><html lang="zh-CN" data-theme="light"><head><meta charset="utf-8"><style>${D.css}\n${D.overrides}</style></head><body><div id="root">${D.snapshots[n].html}</div></body></html>`,'text/html');
 const $=s=>doc.querySelector(s), all=s=>[...doc.querySelectorAll(s)];
 doc.body.className='promo-'+kind;
 $('.brand-home > span:last-child').textContent='EvidenceRAG';
 $('.topbar-note').innerHTML='<span class="tiny-dot"></span> 合成演示 · 流程已压缩';
 $('.topbar-note').classList.add('promo-demo-note');
 $('.topbar-note').setAttribute('data-check','demo-label');
 $('.connection-note').innerHTML='<span class="tiny-dot"></span>原创合成资料 · 无模型调用';
 $('.page-heading h1').textContent={question:'从一个问题开始。',investigation:'调查过程看得见。',inspection:'每次调用，都能查看。',answer:'答案，自带出处。',source:'答案出处查得到。'}[kind];
 $('.page-heading h1').setAttribute('data-check','headline');
 $('.page-heading p').textContent='调查过程看得见，答案出处查得到。';
 all('[style]').forEach(e=>{if(e.style.opacity==='0')e.style.opacity='1';if(e.style.transform&&!(e.closest('.react-flow')))e.style.transform='none'});
 all('.action-elapsed,.timeline-duration,.graph-selection-note,.query-meta > span:nth-child(2)').forEach(e=>e.remove());
 if($('.query-meta'))$('.data-tag').textContent='合成演示';
 if($('.submitted-query')){$('.submitted-query').textContent=D.demo.question;$('.submitted-query').setAttribute('data-check','question')}
 const complete=['answer','source','inspection'].includes(kind),active=kind==='question'?-1:complete?4:2;

 $('.current-action').innerHTML=icon('file',16)+`<span>${kind==='question'?'等待发起调查':kind==='inspection'?'资料读取完成':complete?'已生成答案':'正在补充搜索'}</span>`;
 $('.current-action').classList.toggle('is-running',kind==='investigation');
 restoreGraph(doc,kind);

 const count=kind==='question'?0:complete?4:3;
 $('.timeline-heading').innerHTML=`<span>执行时间线 ${count?`<span class="count-badge">${count}</span>`:''}</span><span style="margin-left:auto;font-size:10px">${complete?'调查记录':'自动执行'}</span>`;
 $('.timeline').innerHTML=kind==='question'?'<div class="promo-waiting" style="padding:0 20px">运行演示后，查看资料查找过程。</div>':labels.slice(0,count).map((s,i)=>`<button class="promo-timeline-row ${i===active?'active':''} ${i===3&&(kind==='answer'||kind==='inspection')?'selected':''}" ${i===3?'data-inspect-target="true"':''} data-check="timeline-${i+1}"><span class="ordinal">${String(i+1).padStart(2,'0')}</span>${icon(i===active?'search':'check',13)}<span>${i===2?'补充搜索':s} · 第 ${i<2?1:2} 次</span><span class="result">${i===active?'执行中':i===3&&(kind==='answer'||kind==='inspection')?'正在查看':'已返回'}</span>${icon('right',12)}</button>`).join('');
 all('.step-detail .detail-properties > div').forEach(e=>{if(e.querySelector('dt')?.textContent!=='调用状态')e.remove()});
 if($('.step-detail > .section-kicker'))$('.step-detail > .section-kicker').textContent='单次工具调用';
 if($('.step-detail h2'))$('.step-detail h2').setAttribute('data-check','call-title');
 if($('.step-detail .detail-description'))$('.step-detail .detail-description').setAttribute('data-check','call-description');
 if($('.step-evidence'))$('.step-evidence').setAttribute('data-check','call-evidence');
 if(kind==='question'){
   $('textarea').textContent=D.demo.question;$('textarea').setAttribute('data-check','question');
   $('.composer button[type="submit"]').setAttribute('data-click-target','run');
 }else{
   $('.answer-heading > span:nth-child(2)').textContent='EvidenceRAG';
   if(kind==='investigation'||kind==='inspection'){
      $('.answer-status').innerHTML='<span class="status-label tone-brand"><span class="tiny-dot"></span>正在调查</span>';
      $('.answer-body').innerHTML='<p class="promo-waiting">正在查找与问题相关的资料。</p><div class="promo-muted"></div><div class="promo-muted" style="width:70%"></div>';
      $('.answer-tools')?.remove();
      if(kind==='investigation'){$('.workbench').className='workbench inspector-collapsed';$('.inspector-slot').style.width='0'}
      if(kind==='inspection'){
        $('.step-evidence button').disabled=true;
        const evidenceTab=all('.tabs-trigger').find(e=>e.textContent.includes('证据'));if(evidenceTab)evidenceTab.textContent='证据';
      }
   }else{
      $('.answer-body').innerHTML=summaries.map((s,i)=>`<p data-check="answer-${i+1}" class="${kind==='source'&&i===1?'selected-summary':''}">${s}<span class="inline-citations"><button aria-label="查看引用 ${i+1}" aria-pressed="${kind==='source'&&i===1}" data-citation="${i+1}" data-click-target="cite-${i+1}"><span>[${i+1}]</span></button></span></p>`).join('');
   }
 }
 if(kind==='source'){
  $('.evidence-detail h3').setAttribute('data-check','source-title');
  $('.evidence-detail .section-kicker').textContent='引用 [2]';
  const parts=D.demo.sources[1].text.split('\n\n');
  $('.evidence-text').innerHTML=`${esc(parts[0])}\n\n<mark data-check="source-quote">${esc(parts[1])}</mark>\n\n${esc(parts[2])}`;
  all('.evidence-list .citation-square').forEach((e,i)=>e.textContent=`[${i+1}]`);
 }
 return '<!doctype html>'+doc.documentElement.outerHTML;
}
function pointer(x,y,click=false){stage.insertAdjacentHTML('beforeend',`${click?`<span class="click-ring" style="left:${x-13}px;top:${y-13}px"></span>`:''}<div class="cursor" data-cursor-tip="${x},${y}" style="left:${x}px;top:${y}px"><svg viewBox="0 0 36 43"><path d="M1 1 2 31 10 24 17 39 24 35 17 21 30 20Z" fill="#263845" stroke="#fff" stroke-width="2.1" stroke-linejoin="round"/></svg></div>`)}
function brand(){const excerpt=D.demo.sources[1].text.split('\n\n')[1].split('。')[0]+'。';return `<div class="brand-wallpaper">${D.demo.glacierSvg}</div><div class="brand-badge">● 合成演示 · 流程已压缩</div><div class="brand-copy">${icon('layers',50)}<h1 data-check="brand-name">EvidenceRAG</h1><p data-check="brand-tagline">让答案有据可查</p></div><article class="recap" data-check="product-card"><header>${icon('layers',18)}文档调查<span>已完成</span></header><div class="recap-question" data-check="question">${esc(D.demo.question)}</div><div class="recap-flow"><span>执行流程 · 02</span><span>调用时间线</span>${labels.map((s,i)=>`${i?icon('right',12):''}<span>${icon('check',12)}${s}</span>`).join('')}</div><div class="recap-body"><section><div class="kicker">调查任务 · 01</div>${summaries.map((s,i)=>`<p data-check="answer-${i+1}">${s}<b>[${i+1}]</b></p>`).join('')}</section><section><div class="kicker">检查器 · 03 / 证据</div><h2 data-check="source-title">[2] 升级演练 / 回退条件</h2><p class="excerpt" data-check="source-quote"><mark>${esc(excerpt)}</mark></p></section></div></article><div class="brand-footer">调查过程看得见，答案出处查得到。</div>`}
window.renderFrame=async(frame)=>{
 const f=Number(frame),kind=f===135?'inspection':f<60?'question':f<150?'investigation':f<210?'answer':f<300?'source':'brand';
 stage.innerHTML=`<div class="stage-wallpaper">${D.demo.glacierSvg}</div>`;stage.dataset.frame=f;stage.dataset.scene=kind;
 if(kind==='brand'){stage.innerHTML=brand();return;}
 const iframe=document.createElement('iframe');iframe.id='frontend';iframe.title='EvidenceRAG 原比例前端';
 const loaded=new Promise(resolve=>iframe.onload=resolve);iframe.srcdoc=createDoc(kind);stage.append(iframe);await loaded;await iframe.contentDocument.fonts.ready;
 const doc=iframe.contentDocument;fitNativeGraph(doc);
 const sel=kind==='question'?'[data-click-target="run"]':kind==='source'?'[data-citation="2"]':kind==='inspection'?'[data-inspect-target]':null;
 if(sel){const r=doc.querySelector(sel).getBoundingClientRect();pointer(kind==='source'&&f>210?r.right+16:r.x+r.width*.70,kind==='source'&&f>210?r.bottom+40+14:r.y+40+r.height*.62,kind==='question'||kind==='inspection')}
 else pointer(1840,973);
 window.currentFrame=f;
};
window.addEventListener('keydown',e=>{const n=Number(e.key);if(n>=1&&n<=5)window.renderFrame([45,105,195,255,345][n-1]);if(e.key.toLowerCase()==='i')window.renderFrame(135)});
window.fit=()=>{const k=Math.min(innerWidth/1920,innerHeight/1080);stage.style.transform=`scale(${k})`;stage.style.left=`${(innerWidth-1920*k)/2}px`;stage.style.top=`${(innerHeight-1080*k)/2}px`};
window.addEventListener('resize',window.fit);window.fit();window.previewReady=window.renderFrame(195);
