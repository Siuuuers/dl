#!/usr/bin/env python3
"""Execute both R36 setup cells in order against a synthetic local Git remote."""
import ast
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import zlib

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/baomai_front_review_r36.ipynb"


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root, stderr=subprocess.DEVNULL).decode().strip()


def png_fixture():
    def chunk(kind, payload):
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind+payload))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(b"\x00\x80\x80\x80")) + chunk(b"IEND", b""))


def setup_source(source, replacements):
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in replacements:
                node.value = ast.Constant(replacements[name])
                if name == "RUN_BASE":
                    node.value = ast.Call(func=ast.Name(id="Path", ctx=ast.Load()), args=[node.value], keywords=[])
    return compile(ast.fix_missing_locations(tree), "R36 complete setup cell 1", "exec")


def check_pending_guards():
    from front_review_r36_training import pending_fields_preserved
    audit = json.loads((ROOT / "results/review/front-review-20261126-r36/row-review.json").read_bytes())
    rows = {(row["image_id"], row["row"]): row for row in audit["rows"]}
    records = {row["id"]: row for row in json.loads((ROOT / "data/manifest.json").read_bytes())["records"]}
    accepted, rejected, partial, full = 0, 0, 0, 0
    for pending in audit["held_request_rows"]:
        key = (pending["image_id"], pending["row"])
        row, size = rows[key], records[key[0]]["original_size"]
        preserve = pending["preserve_row"]
        pending_fields_preserved(row["before_line"], row["after_line"], pending["pending_fields"],
                                 size, str(key), preserve)
        accepted += 1
        full += int(preserve)
        partial += int(not preserve and row["before_line"] != row["after_line"])
        for field in pending["pending_fields"]:
            # A fully held row must also reject a change outside its named pending field.
            index = 1 if preserve else {"left.x": 5, "left.y": 6, "right.x": 8, "right.y": 9}[field]
            axis = 1 if field.endswith(".y") and not preserve else 0
            tokens = row["after_line"].split()
            tokens[index] = f"{float(tokens[index]) + 1 / size[axis]:.9f}"
            try:
                pending_fields_preserved(row["before_line"], " ".join(tokens), pending["pending_fields"],
                                         size, str(key), preserve)
            except ValueError:
                rejected += 1
            else:
                raise AssertionError(f"An unresolved field could change silently: {key} {field}")
    assert accepted and partial and full, "Exercise both partial-field and whole-row preservation."
    return {"accepted_pending_rows": accepted, "accepted_partial_box_changes": partial,
            "fully_held_rows": full, "forbidden_coordinate_changes_rejected": rejected}


def main():
    pending_receipt = check_pending_guards()
    notebook = json.loads(NOTEBOOK.read_bytes())
    code = {i: "".join(cell["source"]) for i, cell in enumerate(notebook["cells"]) if cell["cell_type"] == "code"}
    definitions = [(i, node) for i, source in code.items() for node in ast.parse(source).body
                   if isinstance(node, ast.FunctionDef) and node.name == "assemble_source"]
    assert len(definitions) == 1 and definitions[0][0] == 1, "A later cell overrides the source loader."
    assert all(name not in "\n".join(code.values()) for name in
               ("BASE_REVISION", "PUBLISHED_SOURCE_REVISION", "SOURCE_TREE", "ASSEMBLY_MESSAGE"))
    for index, source in code.items():
        compile(source, f"R36 cell {index}", "exec")
    with tempfile.TemporaryDirectory(prefix="r36-complete-setup-") as temporary:
        base = Path(temporary)
        origin = base / "origin"
        origin.mkdir()
        git(origin, "init", "-q")
        image = png_fixture()
        records = []
        for index in range(157):
            split = "train" if index < 137 else "val"
            name = f"images/{split}/synthetic_{index:06d}.png"
            path = origin / "data" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(image)
            records.append({"id": f"synthetic_{index:06d}", "image": name, "image_sha256": sha(image)})
        weight_path = "weights/cassette_lower_epoch173.pt"
        (origin / "weights").mkdir()
        weights = b"Synthetic initializer bytes for setup integrity checks; never loaded as a model.\n"
        (origin / weight_path).write_bytes(weights)
        (origin / "data/manifest.json").write_text(json.dumps({"records": records,
            "model_checkpoint": weight_path, "model_epoch": 173}))
        protocol_path = "experiments/front-review-2026-11-26-r36.json"
        (origin / "experiments").mkdir()
        protocol = json.dumps({"initial_checkpoint_sha256": sha(weights)}).encode()
        (origin / protocol_path).write_bytes(protocol)
        (origin / "README.md").write_text("Synthetic source loader fixture.\n")
        git(origin, "add", ".")
        git(origin, "-c", "user.name=R36 setup check", "-c", "user.email=r36-check@localhost",
            "commit", "-qm", "Synthetic source and original-input fixtures")
        revision = git(origin, "rev-parse", "HEAD")
        setup = setup_source(code[1], {"SOURCE_REVISION": revision, "PROTOCOL_PATH": protocol_path,
            "PROTOCOL_SHA256": sha(protocol), "SOURCE_URL": str(origin), "RUN_BASE": str(base / "run")})
        restore = compile(code[3], "R36 complete setup cell 3", "exec")
        scope = {}
        exec(setup, scope)
        exec(restore, scope)
        assert scope["assemble_source"].__code__.co_filename == "R36 complete setup cell 1"
        assert git(scope["REPO"], "rev-parse", "HEAD") == revision
        assert not git(scope["REPO"], "status", "--porcelain")
        assert len(list((scope["REPO"] / "data/images").rglob("*.png"))) == 157
        assert (scope["REPO"] / weight_path).read_bytes() == weights
        exec(setup, scope)
        exec(restore, scope)
        marker = scope["REPO"] / "README.md"
        marker.write_text("Keep this local edit.\n")
        try:
            exec(restore, scope)
        except AssertionError as error:
            assert "local changes" in str(error)
        else:
            raise AssertionError("Setup accepted a dirty source checkout.")
        assert marker.read_text() == "Keep this local edit.\n"
        git(scope["REPO"], "checkout", "--", "README.md")
        scope["SOURCE_REVISION"] = "0" * 40
        try:
            exec(restore, scope)
        except AssertionError as error:
            assert "different pins" in str(error)
        else:
            raise AssertionError("Setup accepted the wrong source revision.")
    print(json.dumps({"status": "passed", "notebook_sha256": sha(NOTEBOOK.read_bytes()),
        "complete_setup_cells_executed": [1, 3], "fresh_checkout": True, "input_files_verified": 158,
        "clean_rerun": True, "dirty_source_preserved_and_rejected": True,
        "wrong_source_revision_rejected": True, "pending_request_guards": pending_receipt,
        "training_executed": False}, indent=2))


if __name__ == "__main__":
    main()
