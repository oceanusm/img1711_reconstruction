import unittest
from scene_reconstruction.evaluation_frames import select_frames


class EvaluationFrameTests(unittest.TestCase):
    def test_irregular_video_preserves_exact_rows_and_endpoints(self):
        frames=[{'index':i,'time_seconds':t,'file':f'{i}.jpg'} for i,t in enumerate([0,.01,.02,.5,.51,1,2])]
        selected=select_frames({'frames':frames},4)
        self.assertEqual([f['index'] for f in selected],[0,4,5,6])
        self.assertTrue(all(f==frames[f['index']] for f in selected))

    def test_dense_tail_still_produces_unique_requested_count(self):
        frames=[{'index':i,'time_seconds':t} for i,t in enumerate([0,.001,.002,.003,.004,10])]
        self.assertEqual([f['index'] for f in select_frames({'frames':frames},5)],[0,2,3,4,5])

    def test_all_and_invalid_counts(self):
        m={'frames':[{'index':i,'time_seconds':i*.1} for i in range(10)]}
        self.assertEqual(select_frames(m,10),m['frames'])
        for count in [0,1,11,True,2.5]:
            with self.assertRaises(ValueError): select_frames(m,count)
