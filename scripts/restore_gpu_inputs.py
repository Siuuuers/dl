#!/usr/bin/env python3
"""Restore a verified, offline GPU input package to a new Git checkout.

This file is also copied to the package root. It deliberately uses only the
standard library: PyTorch and Ultralytics are installed after restoration.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True

KIND = "baomai-gpu-inputs"
CHECKPOINT = "weights/cassette_lower_epoch173.pt"
CANDIDATE = "data/candidates/dedup-2026-11-06/manifest.json"
SOURCE_METADATA = ("data/manifest.json", "metadata/model_card.json", CANDIDATE)
HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
HEX_GIT_ID = re.compile(r"[0-9a-f]{40}\Z")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_name(value):
    require(isinstance(value, str) and value, "file path must be a nonempty string")
    path = PurePosixPath(value)
    require(bool(path.parts) and not path.is_absolute() and "\\" not in value and ":" not in value
            and all(part not in (".", "..") for part in path.parts)
            and str(path) == value, f"unsafe or noncanonical relative path: {value!r}")
    return value


def contained_file(root, relative):
    relative_name(relative)
    path = root
    for part in PurePosixPath(relative).parts:
        path = path / part
        require(not path.is_symlink(), f"symlink is not an input file: {relative}")
    require(path.is_file(), f"missing regular input file: {relative}")
    require(root.resolve() in path.resolve().parents, f"input escapes its root: {relative}")
    return path


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path):
    def reject_constant(value):
        raise ValueError(f"non-finite JSON value: {value}")

    return json.loads(path.read_bytes(), object_pairs_hook=_unique_object,
                      parse_constant=reject_constant)


def check_descriptor(root, descriptor, expected_path=None):
    require(isinstance(descriptor, dict), "invalid file descriptor")
    relative = relative_name(descriptor.get("path"))
    if expected_path is not None:
        require(relative == expected_path, f"unexpected package path: {relative}")
    size = descriptor.get("size_bytes")
    digest = descriptor.get("sha256")
    require(type(size) is int and size > 0, f"invalid byte size: {relative}")
    require(isinstance(digest, str) and HEX_SHA256.fullmatch(digest),
            f"invalid SHA-256: {relative}")
    path = contained_file(root, relative)
    require(path.stat().st_size == size, f"byte size mismatch: {relative}")
    require(file_sha256(path) == digest, f"SHA-256 mismatch: {relative}")
    return path


def read_package(package):
    """Validate manifest bytes, all payload hashes and the allowed asset paths."""
    manifest_path = contained_file(package, "manifest.json")
    checksum = contained_file(package, "manifest.sha256").read_bytes()
    expected = (file_sha256(manifest_path) + "  manifest.json\n").encode("ascii")
    require(checksum == expected, "manifest.sha256 does not match manifest.json")
    manifest = read_json(manifest_path)
    require(isinstance(manifest, dict) and manifest.get("schema_version") == 1
            and manifest.get("kind") == KIND, "unsupported GPU input manifest")
    for key in ("source_revision", "source_tree"):
        value = manifest.get(key)
        require(isinstance(value, str) and HEX_GIT_ID.fullmatch(value), f"invalid {key}")
    hashes = manifest.get("source_metadata_sha256")
    require(isinstance(hashes, dict) and set(hashes) == set(SOURCE_METADATA),
            "source metadata inventory differs")
    require(all(isinstance(value, str) and HEX_SHA256.fullmatch(value)
                for value in hashes.values()), "invalid source metadata SHA-256")
    for key, relative in (("source_bundle", "source.bundle"),
                          ("bootstrap", "restore_gpu_inputs.py"), ("guide", "README.md")):
        check_descriptor(package, manifest.get(key), relative)
    assets = manifest.get("assets")
    require(isinstance(assets, list) and len(assets) == 158,
            "package must contain 157 original images and one checkpoint")
    seen = set()
    split_counts = {"train": 0, "val": 0, "checkpoint": 0}
    for asset in assets:
        require(isinstance(asset, dict), "invalid asset descriptor")
        relative = relative_name(asset.get("restore_path"))
        require(relative not in seen, f"duplicate restore path: {relative}")
        seen.add(relative)
        match = re.fullmatch(r"data/images/(train|val)/[^/]+\.png", relative)
        require(match is not None or relative == CHECKPOINT,
                f"unexpected GPU asset: {relative}")
        split_counts[match.group(1) if match else "checkpoint"] += 1
        check_descriptor(package, asset, "assets/" + relative)
    require(split_counts == {"train": 137, "val": 20, "checkpoint": 1},
            "GPU asset split counts differ from the baomai foundation")
    require(isinstance(manifest.get("counts"), dict), "missing dataset counts")
    return manifest


def offline_environment():
    environment = os.environ.copy()
    for key in list(environment):
        if key.startswith("GIT_"):
            del environment[key]
    environment.update({"GIT_LFS_SKIP_SMUDGE": "1", "GIT_TERMINAL_PROMPT": "0",
                        "GIT_ALLOW_PROTOCOL": "file", "GIT_CONFIG_NOSYSTEM": "1",
                        "GIT_CONFIG_GLOBAL": os.devnull, "PYTHONDONTWRITEBYTECODE": "1"})
    return environment


def run(command, cwd=None):
    completed = subprocess.run(command, cwd=cwd, env=offline_environment(),
                               capture_output=True, check=False)
    if completed.returncode:
        error = completed.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"{command[0]} {command[1]} failed: {error[-3000:]}")
    return completed.stdout


def verify_bundle(bundle, revision):
    """An empty repository prevents hidden local prerequisites being accepted."""
    with tempfile.TemporaryDirectory(prefix="verify-gpu-bundle-") as temporary:
        empty = Path(temporary) / "empty.git"
        run(["git", "init", "--quiet", "--bare", str(empty)])
        run(["git", "bundle", "verify", str(bundle)], cwd=empty)
        heads = run(["git", "bundle", "list-heads", str(bundle)], cwd=empty)
        require(heads == f"{revision} HEAD\n".encode("ascii"),
                "bundle HEAD differs from the recorded source revision")


def fresh_output(output, package):
    require(not output.exists() and not output.is_symlink(),
            f"output already exists; choose a new directory: {output}")
    output = output.resolve()
    package = package.resolve()
    require(output != package and package not in output.parents and output not in package.parents,
            "restore output must be outside the extracted package")
    return output


def verify_source_assets(root, manifest):
    """Tie each payload to both source metadata and its committed LFS pointer."""
    for relative, digest in manifest["source_metadata_sha256"].items():
        require(file_sha256(contained_file(root, relative)) == digest,
                f"source metadata hash differs: {relative}")
    for key, relative in (("bootstrap", "scripts/restore_gpu_inputs.py"),
                          ("guide", "docs/gpu-handoff.md")):
        require(file_sha256(contained_file(root, relative)) == manifest[key]["sha256"],
                f"package {key} differs from its recorded Git source")
    reference = read_json(root / "data/manifest.json")
    card = read_json(root / "metadata/model_card.json")
    records = reference.get("records")
    require(isinstance(records, list) and len(records) == 157,
            "source must contain exactly 157 original images")
    expected = {}
    for record in records:
        relative = "data/" + relative_name(record["image"])
        require(relative not in expected, f"duplicate source image: {relative}")
        expected[relative] = record["image_sha256"]
    require(card.get("checkpoint") == CHECKPOINT, "unexpected source checkpoint")
    expected[CHECKPOINT] = card["sha256"]
    assets = {asset["restore_path"]: asset for asset in manifest["assets"]}
    require(set(assets) == set(expected), "package asset inventory differs from the source")
    run(["git", "ls-files", "--error-unmatch", "--", *sorted(expected)], cwd=root)
    for relative, digest in expected.items():
        asset = assets[relative]
        require(asset["sha256"] == digest, f"asset hash differs from source metadata: {relative}")
        pointer = ("version https://git-lfs.github.com/spec/v1\n"
                   f"oid sha256:{digest}\nsize {asset['size_bytes']}\n").encode("ascii")
        require(contained_file(root, relative).read_bytes() == pointer,
                f"committed LFS pointer differs from payload: {relative}")
    require(assets[CHECKPOINT]["size_bytes"] == card["size_bytes"],
            "checkpoint size differs from the model card")


def refresh_verified_assets(root, manifest):
    """Refresh LFS index stats without changing the committed tree contents."""
    expected_tree = manifest["source_tree"].encode("ascii") + b"\n"
    require(run(["git", "write-tree"], cwd=root) == expected_tree,
            "restored index differs from the source before LFS refresh")
    paths = []
    for asset in manifest["assets"]:
        relative = asset["restore_path"]
        check_descriptor(root, {**asset, "path": relative}, relative)
        paths.append(relative)
    # A skipped-smudge checkout records pointer-file sizes in its index. After
    # copying full payloads, status can report them as modified even when the
    # LFS clean filter produces the exact original pointers. Adding only these
    # verified files refreshes that cache; the tree equality below proves that
    # no changed content was accepted into the index.
    run(["git", "add", "--", *paths], cwd=root)
    require(run(["git", "write-tree"], cwd=root) == expected_tree,
            "LFS refresh changed the recorded source tree")


SOURCE_CHECK = """
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path('scripts').resolve()))
from stage_pose_training import source_state
from verify_pose_candidate import DEFAULT_CANDIDATE
from evaluate_pose import preflight_dataset
root = Path.cwd()
state = source_state(root, root / DEFAULT_CANDIDATE)
model = preflight_dataset(root, require_weights=True)
print(json.dumps({'counts': state['counts'], 'model_sha256': model['model_sha256'],
                  'original_images_verified': len(state['records']),
                  'checkpoint_bytes_verified': model['weights_verified']}))
"""


def restore_inputs(package, output):
    package = package.resolve()
    output = fresh_output(output, package)
    manifest = read_package(package)
    run(["git", "--version"])
    run(["git", "lfs", "version"])
    bundle = package / "source.bundle"
    verify_bundle(bundle, manifest["source_revision"])
    output.parent.mkdir(parents=True, exist_ok=True)
    # Reserve a new directory. An existing checkout is never reused or removed.
    output.mkdir(exist_ok=False)
    run(["git", "clone", "--no-checkout", str(bundle), str(output)])
    run(["git", "lfs", "install", "--local"], cwd=output)
    run(["git", "checkout", "--detach", manifest["source_revision"]], cwd=output)
    for expression, key in (("HEAD", "source_revision"), ("HEAD^{tree}", "source_tree")):
        actual = run(["git", "rev-parse", expression], cwd=output).decode("ascii").strip()
        require(actual == manifest[key], f"restored {key} differs")
    verify_source_assets(output, manifest)
    for asset in manifest["assets"]:
        source = contained_file(package, asset["path"])
        destination = contained_file(output, asset["restore_path"])
        shutil.copyfile(source, destination)
    checked = json.loads(run([sys.executable, "-B", "-c", SOURCE_CHECK], cwd=output))
    require(checked["counts"] == manifest["counts"], "restored dataset counts differ")
    checkpoint = next(asset for asset in manifest["assets"] if asset["restore_path"] == CHECKPOINT)
    require(checked["model_sha256"] == checkpoint["sha256"]
            and checked["checkpoint_bytes_verified"] and checked["original_images_verified"] == 157,
            "restored source checks did not verify all images and the original checkpoint")
    refresh_verified_assets(output, manifest)
    status = run(["git", "status", "--porcelain=v1", "--untracked-files=all"], cwd=output)
    require(status == b"", "restored Git checkout is not clean")
    return {"status": "verified", "source_root": str(output),
            "source_revision": manifest["source_revision"], "git_status": "clean",
            "offline_restore": True, **checked}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path,
                        help="new source directory, outside the extracted input package")
    args = parser.parse_args()
    try:
        require(sys.version_info >= (3, 11), "Python 3.11 or newer is required")
        result = restore_inputs(Path(__file__).resolve().parent, args.output.expanduser())
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"GPU input restore failed: {error}\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
