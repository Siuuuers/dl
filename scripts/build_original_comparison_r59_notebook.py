#!/usr/bin/env python3
"""Build the four-cell original-versus-proposed comparison notebook."""
import argparse
import hashlib
import json
from pathlib import Path
import re
from build_inference_ablation_r53_notebooks import SPARSE, cell


def build(commit,sha):
    if not re.fullmatch('[0-9a-f]{40}',commit) or not re.fullmatch('[0-9a-f]{64}',sha):raise ValueError('Complete published source identities required')
    cells=[cell('markdown','''
        # Original supplied model and proposed corner method

        **Select a GPU runtime and Run all four code cells. No training is required.**
        This notebook evaluates the original supplied cassette detector and the
        trained detector with equal original/reflected corner averaging on the
        same 20 development photographs, in the same runtime.

        Both models use box confidence 0.25, original-image input size 1280,
        FP32, ordinary NMS at 0.70 and the same additional box-suppression pass.
        Corner scoring uses confidence 0.50 and box assignment overlap 0.50.
        The proposed method runs its trained model on original and reflected
        images, then averages eligible corresponding corners equally while
        preserving original-view boxes and scores. All alternative-view
        associations use predictions alone.

        The main comparison uses the reviewed annotations. The earlier fixed
        annotations remain a sensitivity check. They describe the same images,
        not independent test sets. The two internal trained views are retained
        for audit. No model, threshold or annotation is selected from this run.
        '''),cell('markdown','''
        ## 1 Load the published source

        Source and results use separate folders. Incompatible existing work is
        preserved. Choose a new `BAOMAI_ORIGINAL_COMPARISON_BASE` for a separate run.
        '''),cell('code',f'''
        from pathlib import Path
        import hashlib, json, os, subprocess, sys
        SOURCE_COMMIT="{commit}"
        DRIVER_SHA256="{sha}"
        BASE=Path(os.environ.get("BAOMAI_ORIGINAL_COMPARISON_BASE","/content/baomai-original-comparison")).resolve()
        SOURCE,WORK,CACHE=BASE/"source",BASE/"results",BASE/"cache"
        BASE.mkdir(parents=True,exist_ok=True)
        ENV={{**os.environ,"GIT_LFS_SKIP_SMUDGE":"1","GIT_TERMINAL_PROMPT":"0"}}
        def run(command,cwd=None):subprocess.run(list(map(str,command)),cwd=cwd,env=ENV,check=True)
        def git(*args):return subprocess.check_output(["git",*args],cwd=SOURCE,env=ENV,text=True).strip()
        if not (SOURCE/".git").exists():
            if SOURCE.exists() and any(SOURCE.iterdir()):raise RuntimeError("Nonempty source preserved; choose a new base folder.")
            run(["git","init",SOURCE])
            run(["git","remote","add","origin","https://github.com/Siuuuers/dl.git"],SOURCE)
            run(["git","fetch","--depth=1","--filter=blob:none","origin",SOURCE_COMMIT],SOURCE)
            run(["git","sparse-checkout","init","--no-cone"],SOURCE)
            run(["git","sparse-checkout","set",*{SPARSE!r}],SOURCE)
            run(["git","checkout","--detach",SOURCE_COMMIT],SOURCE)
        if git("rev-parse","HEAD")!=SOURCE_COMMIT or git("status","--porcelain","--untracked-files=all"):
            raise RuntimeError("Changed source preserved; choose a new base folder.")
        run(["git","sparse-checkout","init","--no-cone"],SOURCE)
        run(["git","sparse-checkout","set",*{SPARSE!r}],SOURCE)
        DRIVER=SOURCE/"scripts/run_original_comparison_r59.py"
        assert hashlib.sha256(DRIVER.read_bytes()).hexdigest()==DRIVER_SHA256
        def command(action,*extra):
            return [sys.executable,"-B",DRIVER,action,"--root",SOURCE,"--source-commit",SOURCE_COMMIT,
                    "--script-sha256",DRIVER_SHA256,"--cache-dir",CACHE,"--work",WORK,*extra]
        print("Verified source:",SOURCE_COMMIT)
        '''),cell('markdown','''
        ## 2 Verify both models images and annotations

        The supplied epoch-173 checkpoint is identified by its complete hash.
        The trained checkpoint is recovered from its audited training return.
        Every photograph and reference label is checked. The archived original
        model's three confidence settings and 60 per-image score records are
        replayed before the fresh comparison. The older trainer's mAP figures
        are not used as a substitute for these shared pixel measurements.
        '''),cell('code','''
        run(command("prepare"))
        preflight=json.loads((WORK/"preflight.json").read_text())
        print("Original model:",preflight["original_model_sha256"])
        print("Trained model:",preflight["trained_model_sha256"])
        print("Shared settings:",preflight["protocol"])
        '''),cell('markdown','''
        ## 3 Run the common inference comparison

        Both model files run in the same recorded runtime. The trained model
        additionally runs on explicitly reflected pixels; returned coordinates
        are mapped back and left/right corner slots are exchanged. Equal corner
        averaging follows the fixed association rule. The original model also
        receives the same additional suppression, so this measures supplied
        weights with common processing rather than reproducing a different
        live-camera application setting.

        The runtime's compatible Torch/TorchVision pair is retained. CPU is
        supported by explicitly setting `BAOMAI_ORIGINAL_COMPARISON_DEVICE=cpu`.
        No camera stream, timing benchmark or training run is involved.
        '''),cell('code','''
        from importlib.metadata import version
        import torch
        DEVICE=os.environ.get("BAOMAI_ORIGINAL_COMPARISON_DEVICE","0")
        assert DEVICE=="cpu" or torch.cuda.is_available(),"Select a GPU runtime or explicitly choose CPU."
        constraints=BASE/"torch-constraints.txt"
        constraints.write_text("".join(f"{name}=={version(name)}\\n" for name in ("torch","torchvision")))
        run([sys.executable,"-m","pip","install","--quiet","-c",constraints,
             "ultralytics==8.4.182","opencv-python==5.0.0.93","numpy==2.4.6","Pillow==13.0.0"])
        run(command("run","--device",DEVICE))
        '''),cell('markdown','''
        ## 4 Verify and download the result

        Verification reconstructs the inverse reflection, common suppression,
        associations, averaging, all reference scores and individual gains/losses
        from saved predictions. Every eligible two-corner object stays in the
        denominator, including missed detections. The archive also checks
        numerical continuity with historical outputs and records any difference.

        The compact result ZIP contains the source/input identities, predictions,
        paired cases and measurements; model files stay at their existing
        immutable sources. Upload this result ZIP directly when returning the run.
        Original corner scores are comparison thresholds and are not calibrated
        confidence in changed coordinates. Existing development images cannot
        establish performance on a new capture session.
        '''),cell('code','''
        run(command("verify"))
        summary=json.loads((WORK/"summary.json").read_text())
        print("Historical prediction continuity:",summary["historical_reproduction_passed"])
        print("Reference | model | missed | extra | duplicate | both corners within5 | within10")
        for reference in ("r50","frozen666"):
            for arm in ("original","proposed"):
                result=summary["measurement"]["references"][reference][arm]
                d,p=result["detection"],result["pairs"]
                print(reference,arm,d["fn"],d["fp"],d["duplicate_fp"],
                      f"{p['within5']}/{p['eligible']}",f"{p['within10']}/{p['eligible']}")
        ARCHIVE=WORK/"baomai-original-comparison-r59-results.zip"
        from IPython.display import FileLink,display
        try:
            from google.colab import files
        except ImportError:
            display(FileLink(str(ARCHIVE)))
        else:
            files.download(str(ARCHIVE))
        ''')]
    return {'nbformat':4,'nbformat_minor':5,'cells':cells,'metadata':{
        'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'language_info':{'name':'python'},'colab':{'name':'baomai_original_comparison_r59.ipynb','provenance':[]},
        'accelerator':'GPU','source_commit':commit,'driver_sha256':sha,'training':False,'fresh_inference':True}}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-commit',required=True);p.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parents[1]/'notebooks');a=p.parse_args()
    sha=hashlib.sha256(Path(__file__).with_name('run_original_comparison_r59.py').read_bytes()).hexdigest()
    a.output_dir.mkdir(parents=True,exist_ok=True);target=a.output_dir/'baomai_original_comparison_r59.ipynb'
    target.write_text(json.dumps(build(a.source_commit,sha),indent=1)+'\n')
    print(json.dumps({'notebook':str(target),'source_commit':a.source_commit,'driver_sha256':sha}))

if __name__=='__main__':main()
