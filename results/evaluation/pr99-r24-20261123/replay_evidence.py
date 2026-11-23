#!/usr/bin/env python3
"""Replay PR99 scores from preserved predictions and verify exact report identity; no inference/training."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--evidence',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve();e=args.evidence.resolve()
    assert not args.output.exists()
    paths={'pr35':'pr35','reference':'reference','candidate':'candidate'}
    def command(script,out,candidate='candidate'):
        cmd=[sys.executable,'-B',str(e/script),'--root',str(root),'--output',str(out)]
        for role,source in paths.items():
            if role=='candidate':source=candidate
            cmd += ['--'+role,str(e/'inference'/('eval-'+source)/'predictions.jsonl'),
                    '--'+role+'-manifest',str(e/'checkpoint-receipts'/source/'checkpoint-manifest.json')]
        return cmd
    checks={}
    with tempfile.TemporaryDirectory(prefix='pr99-score-replay-') as tmp:
        temp=Path(tmp)
        for script,name,role in [('measure_pr99.py','fixed-comparison.json','candidate'),('measure_r23_same_runtime.py','r23-same-runtime-comparison.json','r23-candidate')]:
            result=subprocess.run(command(script,temp/name,role),capture_output=True,text=True)
            if result.returncode:raise RuntimeError(result.stderr+result.stdout)
            assert (temp/name).read_bytes()==(e/name).read_bytes(),name
            checks[name]={'exact_bytes_equal':True,'sha256':sha(temp/name)}
        result=subprocess.run([sys.executable,'-B',str(e/'reference_sensitivity.py'),'--root',str(root),'--evidence',str(e),'--pr35',str(e/'inference/eval-pr35/predictions.jsonl'),'--output',str(temp/'reference-sensitivity.json')],capture_output=True,text=True)
        if result.returncode:raise RuntimeError(result.stderr+result.stdout)
        assert (temp/'reference-sensitivity.json').read_bytes()==(e/'reference-sensitivity.json').read_bytes()
        checks['reference-sensitivity.json']={'exact_bytes_equal':True,'sha256':sha(temp/'reference-sensitivity.json')}
    report={'schema_version':1,'kind':'PR99 saved-prediction exact replay','script_sha256':sha(__file__),'checks':checks,'limits':['This replay checks preserved numerical evidence and checkpoint receipts; it does not reverify external checkpoint weight bytes or rerun inference.','Original hash-matching validation images and the pinned project scripts/references remain required.']}
    args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
