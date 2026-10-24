process.stdin.setEncoding('utf8');
import ts from "../../../frontend/node_modules/typescript/lib/typescript.js";
let input='';for await(const c of process.stdin)input+=c;
const {demoTs}=JSON.parse(input);
const js=ts.transpileModule(demoTs,{compilerOptions:{target:ts.ScriptTarget.ES2022,module:ts.ModuleKind.ES2022}}).outputText;
const demo=await import(`data:text/javascript;base64,${Buffer.from(js).toString('base64')}`);
const {events}=demo.demoScript('answered','promo-native-agent-graph');
const nodeFor=a=>a==='decide'?'agent':a==='search_docs'?'search':a==='read_passage'?'read':'finish';
function state(list,selected=null){
 const calls=list.filter(e=>e.phase==='started').map(e=>({id:e.invocation_id,node:nodeFor(e.action),done:list.some(x=>x.invocation_id===e.invocation_id&&x.phase==='completed')}));
 const latest=calls.at(-1),active=latest&&!latest.done?latest.node:null;
 const nodes=Object.fromEntries(['agent','search','read','finish'].map(n=>{const visits=calls.filter(c=>c.node===n);return[n,{visits:visits.length,status:active===n?'running':visits.at(-1)?.done?'done':'waiting',inspection:n===selected?'第 2 次':null}]}));
 const traversed=calls.slice(1).map((c,i)=>`${calls[i].node}-${c.node}`);
 return {nodes,active,traversed,flowing:active&&calls.length>1?`${calls.at(-2).node}-${active}`:null,selected};
}
const searches=events.filter(e=>e.action==='search_docs'&&e.phase==='started');
const reads=events.filter(e=>e.action==='read_passage'&&e.phase==='completed');
const until=e=>events.slice(0,events.indexOf(e)+1);
const readStarts=events.filter(e=>e.action==='read_passage'&&e.phase==='started');
const decisions=events.filter(e=>e.action==='decide'&&e.phase==='started');
const validation=events.find(e=>e.action==='validate_answer'&&e.phase==='started');
console.log(JSON.stringify({question:state([]),investigation:state(until(searches[1])),inspection:state(until(reads[1]),'read'),answer:state(events,'read'),source:state(events,'read'),search1:state(until(searches[0])),read1:state(until(readStarts[0])),return1:state(until(decisions[2])),read2:state(until(readStarts[1])),agentStart:state(until(decisions[0])),validate:state(until(validation),'read')}));
