import json
import subprocess
import sys
from pathlib import Path


def test_cache_not_accepted_refuses_before_loading_model(tmp_path):
    gate=tmp_path/'gate.json'
    gate.write_text(json.dumps({'cache_audit_passed':False,'exclusive_b0_window':True}))
    runner=Path(__file__).resolve().parents[1]/'scripts/run_oft_b0_smoke.py'
    args=[sys.executable,str(runner)]
    for name in ['model','source-lock','train-manifest','eval-manifest','statistics']:
        args += ['--'+name,str(tmp_path/'not_created')]
    args+=['--gpu-window',str(gate),'--output',str(tmp_path/'output')]
    result=subprocess.run(args,capture_output=True,text=True)
    assert result.returncode!=0
    assert 'Resource/cache gate not passed' in result.stderr
    assert not (tmp_path/'output').exists()
