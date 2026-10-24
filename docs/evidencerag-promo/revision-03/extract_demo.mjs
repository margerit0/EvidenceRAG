import ts from '../../../frontend/node_modules/typescript/lib/typescript.js';

process.stdin.setEncoding('utf8');
let input = '';
for await (const chunk of process.stdin) input += chunk;
const { demoTs } = JSON.parse(input);
const js = ts.transpileModule(demoTs, {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ES2022 },
}).outputText;
const demo = await import(`data:text/javascript;base64,${Buffer.from(js).toString('base64')}`);
const result = demo.demoScript('answered', 'promo-original-synthetic').result;
console.log(JSON.stringify({
  question: demo.scenarios.find(s => s.id === 'answered').question,
  sources: result.sources,
  blocks: result.blocks,
  actions: result.events
    .filter(event => ['search_docs', 'read_passage'].includes(event.action))
    .map(event => event.action),
}));
