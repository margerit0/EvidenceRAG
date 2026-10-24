"""Render six real HTML keyframes at native browser scale, offline."""
from pathlib import Path
import base64,json,subprocess,sys
from zhrag.io_utils import read_bytes,read_json,read_text,write_json,write_text
HERE=Path(__file__).resolve().parent
BASE=HERE.parent
REPO=BASE.parents[1]
from media_data import node_data
FRAMES=[
 {"frame":45,"filename":"01-question.png","time":"1.5 s","title":"提出问题","action":"问题已填好，点击运行演示。","transition":"保留浏览器尺度，输入内容进入调查区。"},
 {"frame":105,"filename":"02-investigation.png","time":"3.5 s","title":"看见调查过程","action":"原生 Agent 图显示工具往返，时间线记录搜索与读取顺序。","transition":"4.5 秒点击完成的读取记录，展开右侧检查器。"},
 {"frame":195,"filename":"03-answer.png","time":"6.5 s","title":"答案与检查器","action":"两条带引用结论出现，检查器保留单次调用详情。","transition":"三栏保持前端原比例；光标移向 [2]。"},
 {"frame":255,"filename":"04-source.png","time":"8.5 s","title":"对照原文","action":"点击 [2]，同一检查器切到证据并高亮对应原文。","transition":"页面不整体放大；原文保持至少 1.5 秒。"},
 {"frame":345,"filename":"05-brand.png","time":"11.5 s","title":"品牌收尾","action":"产品内容收拢为卡片，展示 EvidenceRAG 和标语。","transition":"品牌稳定停留至少 1 秒。"},
 {"frame":135,"filename":"02b-inspector.png","time":"4.5 s","title":"查看调用详情","action":"第二次读取完成后点击时间线。","transition":"此图为检查器展开后的落位构图。"}
]
def main():
 sys.stdout.reconfigure(encoding="utf-8");sys.stderr.reconfigure(encoding="utf-8")
 out=HERE/"preview";out.mkdir(exist_ok=True)
 native=read_json(HERE/"reference/native.json")
 font="data:font/woff2;base64,"+base64.b64encode(read_bytes(REPO/"frontend/node_modules/@fontsource-variable/inter/files/inter-latin-wght-normal.woff2")).decode("ascii")
 css=native["css"].replace('/workbench/node_modules/@fontsource-variable/inter/files/inter-latin-wght-normal.woff2',font)
 demo=node_data()
 demo["glacierSvg"]=read_text(REPO/"frontend/src/assets/glacier.svg")
 graph_run=subprocess.run(["node",str(HERE/"demo_graph.mjs")],input=json.dumps({"demoTs":read_text(REPO/"frontend/src/lib/demo.ts")},ensure_ascii=False),text=True,encoding="utf-8",capture_output=True,check=True)
 demo["graphStates"]=json.loads(graph_run.stdout)
 data={"snapshots":native["snapshots"],"css":css,"overrides":read_text(HERE/"native-overrides.css"),"demo":demo}
 html=read_text(HERE/"frame.html").replace("__INTER_FONT__",font).replace("/*__DATA__*/",json.dumps(data,ensure_ascii=False).replace("</","<\\/")).replace("/*__SCRIPT__*/",read_text(HERE/"storyboard.js"))
 write_text(HERE/"preview.html",html)
 payload={"html":html,"frames":FRAMES,"outputDir":str(out),"font":font}
 proc=subprocess.run(["node",str(HERE/"render.mjs")],input=json.dumps(payload,ensure_ascii=False),text=True,encoding="utf-8",capture_output=True,check=False)
 if proc.stderr:print(proc.stderr,file=sys.stderr)
 proc.check_returncode();report=json.loads(proc.stdout)
 write_json(out/"qa-report.json",report)
 print(json.dumps({"frames":len(report["frames"]),"errors":report["errors"],"issues":report["issues"],"overview":report["overview"]},ensure_ascii=False,indent=2))
 for f in report["frames"]:
  assert f["hit"] is None or f["hit"]["inside"]
  if f["frame"] in [195,255]:
   assert f["metrics"][".page-heading h1"]["fontSize"]=="29px"
   assert f["metrics"][".submitted-query"]["fontSize"]=="16px"
   assert f["metrics"][".answer-body > p"]["fontSize"]=="13px"
   assert [f["metrics"][s]["w"] for s in [".conversation-pane",".execution-pane",".inspector-slot"]]==[484,650,500]
  if f["graph"]:
   assert [n["id"] for n in f["graph"]["nodes"]]==["agent","search","read","finish"]
   assert {e["id"] for e in f["graph"]["edges"]}=={"agent-search","agent-read","search-agent","read-agent","agent-finish"}
   if f["frame"] in [135,195,255]:assert f["graph"]["selected"]=="read"
  if f["frame"]==255:assert f["sourceText"]==demo["sources"][1]["text"].split("\n\n")[1]
 assert not report["errors"] and not report["issues"]
if __name__=="__main__":main()
