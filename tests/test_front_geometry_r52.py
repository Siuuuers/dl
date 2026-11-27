"""Target the false-positive and protected-coordinate risks in the R52 audit."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import audit_front_geometry_r52 as audit
import build_front_review_r52 as build


def row(sid, box, points, image="synthetic"):
    return {"stable_id": sid, "image_id": image, "output_row": int(sid[-1]),
            "source_row": int(sid[-1]), "box_xyxy_px": box, "points": points}


class FourSideReviewTest(unittest.TestCase):
    def test_each_side_and_strict_interior_are_distinct_cues(self):
        target = row("target1", [10,10,30,30], [{"side":"left","x":12,"y":28}])
        cases = ((10,20,"xmin"),(20,10,"ymin"),(30,20,"xmax"),(20,30,"ymax"))
        for x,y,edge in cases:
            with self.subTest(edge=edge):
                neighbor = row("other2",[x-1,y-1,x+1,y+1],[{"side":"left","x":x,"y":y}])
                result = audit.screen([target,neighbor])["rows"][0]
                self.assertEqual(result["foreign_interior_corners"], [])
                self.assertEqual(result["foreign_edge_contacts"][0]["edges"], [edge])
        interior = row("other2",[19,19,21,21],[{"side":"left","x":20,"y":20}])
        result = audit.screen([target,interior])["rows"][0]
        self.assertEqual(len(result["foreign_interior_corners"]),1)
        self.assertEqual(result["foreign_edge_contacts"],[])

    def test_touching_corner_can_flag_two_edges_without_mutation(self):
        rows=[row("target1",[10,10,30,30],[]),
              row("other2",[29,29,31,31],[{"side":"right","x":30,"y":30}])]
        original=deepcopy(rows)
        result=audit.screen(rows)["rows"][0]
        self.assertEqual(set(result["foreign_edge_contacts"][0]["edges"]),{"xmax","ymax"})
        self.assertEqual(rows,original)

    def test_unavailable_corner_does_not_acquire_location(self):
        box,points=audit.geometry("0 0.5 0.5 0.1 0.1 0 0 0 0.53 0.53 2")
        self.assertEqual(len(points),1)
        self.assertEqual(points[0]["side"],"right")

    def test_other_photo_cannot_create_contact(self):
        rows=[row("target1",[10,10,30,30],[]),
              row("other2",[29,29,31,31],[{"side":"right","x":30,"y":30}],image="unrelated")]
        self.assertEqual(audit.screen(rows)["rows"][0]["flags"],[])

    def test_protected_class_y_and_corner_tokens_reject_drift(self):
        before="0 0.5 0.5 0.1 0.1 0 0 0 0.53 0.53 2"
        for index in (0,2,4,5,6,7,8,9,10):
            with self.subTest(index=index):
                changed=before.split();changed[index]="0.123"
                with self.assertRaises(ValueError):build.protect(before," ".join(changed))
        changed=before.split();changed[1]="0.49";changed[3]="0.09"
        build.protect(before," ".join(changed))

    def test_committed_ledger_replays_with_all_hold_gates(self):
        manifest,files,before,after=build.replay(Path(__file__).resolve().parents[1])
        self.assertEqual(manifest["counts"]["amended_rows"],10)
        self.assertEqual(manifest["counts"]["changed_side_boundaries"],11)
        self.assertEqual(manifest["counts"]["held_fields"],31)
        self.assertEqual(len(manifest["annotation_complete_ids"]),6)
        self.assertEqual(len(manifest["held_image_ids"]),6)
        self.assertEqual([r["points"] for r in before],[r["points"] for r in after])
        self.assertEqual([r["remaining_holds"] for r in before],[r["remaining_holds"] for r in after])
        self.assertEqual(manifest["after_geometry_counts"]["own_corner_outside_rows"],0)


if __name__ == "__main__":
    unittest.main()
