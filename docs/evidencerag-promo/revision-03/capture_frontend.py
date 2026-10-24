"""Capture the current synthetic frontend at its real 100% CSS scale."""
from pathlib import Path
import subprocess,sys,json
from zhrag.io_utils import write_json
sys.stdout.reconfigure(encoding="utf-8");sys.stderr.reconfigure(encoding="utf-8")
HERE=Path(__file__).resolve().parent
out=HERE/"reference";out.mkdir(exist_ok=True)
p=subprocess.run(["node",str(HERE/"capture_frontend.mjs")],input=json.dumps({"outputDir":str(out)}),text=True,encoding="utf-8",capture_output=True,check=False)
if p.stderr:print(p.stderr,file=sys.stderr)
p.check_returncode();result=json.loads(p.stdout)
write_json(out/"native.json",result)
print(json.dumps({"errors":result["errors"],"viewport":result["viewport"],"metrics":result["snapshots"]["answer"]["metrics"]},ensure_ascii=False,indent=2))
