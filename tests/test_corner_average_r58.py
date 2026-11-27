import copy
from pathlib import Path
import sys
import tempfile
import unittest
from zipfile import ZipFile, ZIP_DEFLATED
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import run_corner_average_r58 as average


def prediction(box, points=None, identity=1, klass=0):
    return {'bbox_xyxy':box,'keypoints':points or [[box[0]+1,box[3]-1,.9],[box[2]-1,box[3]-1,.8]],
            'confidence':.75,'class_id':klass,'primary_prediction_id':identity}


class AveragingControls(unittest.TestCase):
    def test_crowded_tie_cannot_reuse_a_reflected_instance(self):
        original=[prediction([0,0,10,10],identity=10),prediction([2,0,12,10],identity=20)]
        alternative=[prediction([1,0,11,10],[[2,8,.6],[10,8,.7]])]
        before=copy.deepcopy((original,alternative))
        result,links=average.average_predictions(original,alternative)
        self.assertEqual([r['status'] for r in links],['averaged','no_mutual_match'])
        self.assertEqual(result[1],original[1])
        self.assertEqual(result[0]['primary_prediction_id'],10)
        self.assertEqual(result[0]['keypoints'],[[1.5,8.5,.9],[9.5,8.5,.8]])
        self.assertEqual((original,alternative),before)

    def test_inclusive_boundaries_and_class_identity(self):
        original=[prediction([0,0,30,10],[[1,9,.5],[29,9,.5]])]
        alternative=[prediction([10,0,40,10],[[11,9,.5],[39,9,.5]])]
        result,links=average.average_predictions(original,alternative)
        self.assertEqual(links[0]['box_iou'],.5)
        self.assertEqual(links[0]['status'],'averaged')
        self.assertEqual(result[0]['keypoints'],[[6,9,.5],[34,9,.5]])
        alternative[0]['class_id']=1
        self.assertEqual(average.average_predictions(original,alternative)[0],original)

    def test_unavailable_pair_retains_both_original_points(self):
        original=[prediction([0,0,30,10])]
        for bad in (None,float('nan'),1280):
            alternative=[prediction([0,0,30,10],[[bad,8,.9],[28,8,.9]])]
            result,links=average.average_predictions(original,alternative)
            self.assertEqual(result,original)
            self.assertEqual(links[0]['status'],'reflected_pair_unavailable')
        alternative=[prediction([0,0,30,10],[[2,8,.49999],[28,8,.9]])]
        self.assertEqual(average.average_predictions(original,alternative)[0],original)
        self.assertEqual(average.average_predictions(original,[])[0],original)
        self.assertEqual(average.average_predictions([],alternative),([],[]))

    def test_replay_rejects_modified_payload_even_with_new_inventory(self):
        source={'commit':'a'*40,'files_sha256':{},'python':'3.12.0'}
        expected=average.bundle({'status':'completed'},{'points.json':{'x':1}},source)
        with tempfile.TemporaryDirectory() as tmp:
            archive=Path(tmp)/'result.zip'
            with ZipFile(archive,'w',ZIP_DEFLATED) as z:
                for name,value in expected.items():z.writestr(name,value)
            average.verify_result(archive,expected)
            portable = dict(expected)
            compressed = bytearray(portable['points.json.gz'])
            compressed[9] = 255 if compressed[9] != 255 else 3
            portable['points.json.gz'] = bytes(compressed)
            portable['inventory.json'] = average.encoded({name: {'bytes':len(value), 'sha256':average.digest(value)}
                for name,value in portable.items() if name != 'inventory.json'})
            with ZipFile(archive,'w',ZIP_DEFLATED) as z:
                for name,value in portable.items():z.writestr(name,value)
            average.verify_result(archive,expected)
            altered=average.bundle({'status':'completed'},{'points.json':{'x':2}},source)
            with ZipFile(archive,'w',ZIP_DEFLATED) as z:
                for name,value in altered.items():z.writestr(name,value)
            with self.assertRaisesRegex(RuntimeError,'Recomputed result differs'):
                average.verify_result(archive,expected)


if __name__=='__main__':unittest.main()
