#!/usr/bin/env python3
"""Create a small source-pinned CPU notebook for two-view corner averaging."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import textwrap


def cell(kind, text):
    source = textwrap.dedent(text).strip()+'\n'
    result = {'cell_type': kind, 'id': hashlib.sha256((kind+source).encode()).hexdigest()[:12],
              'metadata': {}, 'source': source}
    if kind == 'code':
        result.update(execution_count=None, outputs=[])
    return result


def build(commit, driver_sha):
    if not re.fullmatch('[0-9a-f]{40}', commit) or not re.fullmatch('[0-9a-f]{64}', driver_sha):
        raise ValueError('Use complete source commit and driver SHA-256 values.')
    cells = [cell('markdown', '''
        # Two view corner averaging

        **CPU runtime is sufficient. Run all four code cells.**

        This notebook uses the returned original and horizontally reflected predictions.
        It keeps every original detection box, score and record identifier. A same-class,
        mutual-best predicted-box match with overlap at least 0.50 supplies an alternative
        corner pair. If both original and alternative pairs are valid at corner score 0.50,
        their corresponding coordinates are averaged equally. Other records stay unchanged.

        No annotation selects a view, match, weight or cutoff. The annotations are used
        afterward to measure performance. Original scores define a fixed comparison
        cohort; they are not calibrated confidence in the averaged coordinates.
        This is an exploratory analysis of the existing 20 development photographs.
        It performs no model inference or training.
        '''), cell('markdown', '''
        ## 1 Load the published experiment source

        The notebook uses one immutable source revision. Choose a fresh
        `BAOMAI_CORNER_AVERAGE_BASE` for a separate run; incompatible existing files
        are preserved. Python and its standard library are sufficient.
        '''), cell('code', f'''
        from pathlib import Path
        import hashlib, json, os, subprocess, sys, urllib.request

        SOURCE_COMMIT = "{commit}"
        DRIVER_SHA256 = "{driver_sha}"
        BASE = Path(os.environ.get("BAOMAI_CORNER_AVERAGE_BASE", "/content/baomai-corner-average")).resolve()
        SOURCE, WORK, CACHE = BASE/"source", BASE/"results", BASE/"cache"
        BASE.mkdir(parents=True, exist_ok=True)
        ENV = {{**os.environ, "GIT_LFS_SKIP_SMUDGE": "1", "GIT_TERMINAL_PROMPT": "0"}}
        def run(command, cwd=None):
            subprocess.run(list(map(str, command)), cwd=cwd, env=ENV, check=True)
        def git(*args):
            return subprocess.check_output(["git", *args], cwd=SOURCE, env=ENV, text=True).strip()
        if not (SOURCE/".git").exists():
            if SOURCE.exists() and any(SOURCE.iterdir()):
                raise RuntimeError("Nonempty source preserved. Choose a fresh base folder.")
            run(["git", "init", SOURCE])
            run(["git", "remote", "add", "origin", "https://github.com/Siuuuers/dl.git"], SOURCE)
            run(["git", "fetch", "--depth=1", "--filter=blob:none", "origin", SOURCE_COMMIT], SOURCE)
            run(["git", "sparse-checkout", "init", "--cone"], SOURCE)
            run(["git", "sparse-checkout", "set", "scripts", "data"], SOURCE)
            run(["git", "checkout", "--detach", SOURCE_COMMIT], SOURCE)
        if git("rev-parse", "HEAD") != SOURCE_COMMIT or git("status", "--porcelain", "--untracked-files=all"):
            raise RuntimeError("Changed source preserved. Choose a fresh base folder.")
        DRIVER = SOURCE/"scripts/run_corner_average_r58.py"
        assert hashlib.sha256(DRIVER.read_bytes()).hexdigest() == DRIVER_SHA256
        print("Source verified:", SOURCE_COMMIT)
        '''), cell('markdown', '''
        ## 2 Fetch and verify the returned predictions

        The exact reflection result archive is downloaded from its immutable results
        commit. Its size and SHA-256 are checked before analysis. The driver additionally
        verifies the member inventory, source identity, geometry and reference labels.
        '''), cell('code', '''
        CACHE.mkdir(parents=True, exist_ok=True)
        RETURN_SHA256 = "88689ce9f3d03178683ea98ce98d50f5ddc4e07ba1fb24262e7820e9488bd114"
        RETURN_BYTES = 2931102
        URL = "https://raw.githubusercontent.com/Siuuuers/dl/dfe5392109433fcee51f41716710001bb123a3d5/results/a100/baomai-flip-inference-r56-a100-001/baomai-flip-inference-r56-results.zip"
        INPUT = CACHE/"reflection-results.zip"
        if INPUT.exists():
            payload = INPUT.read_bytes()
        else:
            with urllib.request.urlopen(URL, timeout=90) as response:
                payload = response.read(RETURN_BYTES+1)
        if len(payload) != RETURN_BYTES or hashlib.sha256(payload).hexdigest() != RETURN_SHA256:
            raise RuntimeError("Result archive differs; existing cache was preserved.")
        if not INPUT.exists():
            INPUT.write_bytes(payload)
        def command(action):
            return [sys.executable, "-B", DRIVER, action, "--root", SOURCE,
                    "--source-commit", SOURCE_COMMIT, "--script-sha256", DRIVER_SHA256,
                    "--archive", INPUT, "--work", WORK]
        print("Verified reflection archive:", len(payload), "bytes")
        '''), cell('markdown', '''
        ## 3 Compare coordinates with all detection records fixed

        The original and reflected saved tables are reproduced before interpreting
        the new coordinate average. All eligible two-corner targets remain in the
        denominator, including missed fronts. Both five- and ten-pixel pair criteria
        are reported, together with gained and lost targets and accepted-output groups.
        '''), cell('code', '''
        run(command("run"))
        summary = json.loads((WORK/"summary.json").read_text())
        print("Association counts:", summary["association_counts"])
        print("reference | coordinates | misses | extras | duplicates | pair5 | pair10")
        for reference in ("frozen666", "r50"):
            for arm in ("control", "reflection", "average"):
                item = summary["references"][reference]["arms"][arm]
                d, p = item["detection"], item["pairs"]
                print(reference, arm, d["fn"], d["fp"], d["duplicate_fp"],
                      f"{p['within5']}/{p['eligible']}", f"{p['within10']}/{p['eligible']}")
            print("Averaging gains and losses:", summary["references"][reference]["paired_changes"])
        '''), cell('markdown', '''
        ## 4 Verify and download the complete result

        Replay verifies the output bundle against a fresh calculation. The archive
        includes coordinate predictions, view associations, all target changes,
        complete summary tables, source identities and a member hash inventory.
        A better five-pixel result can accompany a worse ten-pixel result; read both.
        '''), cell('code', '''
        run(command("verify"))
        ARCHIVE = WORK/"baomai-corner-average-r58-results.zip"
        from IPython.display import FileLink, display
        try:
            from google.colab import files
        except ImportError:
            display(FileLink(str(ARCHIVE)))
        else:
            files.download(str(ARCHIVE))
        ''')]
    return {'cells':cells,'nbformat':4,'nbformat_minor':5,
            'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
                'language_info':{'name':'python'},'source_commit':commit,'driver_sha256':driver_sha,
                'experiment':'fixed-box corner average','training':False,'fresh_inference':False,
                'colab':{'name':'baomai_corner_average_r58.ipynb','provenance':[]}}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-commit',required=True)
    parser.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parents[1]/'notebooks')
    args=parser.parse_args()
    sha=hashlib.sha256(Path(__file__).with_name('run_corner_average_r58.py').read_bytes()).hexdigest()
    nb=build(args.source_commit,sha)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    target=args.output_dir/'baomai_corner_average_r58.ipynb'
    target.write_text(json.dumps(nb,indent=1)+'\n')
    print(json.dumps({'notebook':str(target),'source_commit':args.source_commit,'driver_sha256':sha,'code_cells':4}))


if __name__=='__main__':
    main()
