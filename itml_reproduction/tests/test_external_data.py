import unittest
import numpy as np
from itml_repro.external_data import pcd
from itml_repro.facegrabber import camera,project,depth_image,TARGET
from itml_repro.weightgait import frame_box


class ExternalDataTests(unittest.TestCase):
    def test_pcd_padding_rgb_and_truncated_payload(self):
        dtype=np.dtype(dict(names=['x','y','z','rgb'],formats=['<f4','<f4','<f4','<u4'],offsets=[0,4,8,16],itemsize=32))
        a=np.zeros(2,dtype=dtype);a['x']=[1,2];a['y']=[3,4];a['z']=[.5,.6];a['rgb']=[0x123456,0xaabbcc]
        header=b'FIELDS x y z _ rgb _\nCOUNT 1 1 1 4 1 12\nPOINTS 2\nDATA binary\n'
        xyz,rgb=pcd(header+a.tobytes())
        np.testing.assert_allclose(xyz,[[1,3,.5],[2,4,.6]])
        np.testing.assert_array_equal(rgb,[[18,52,86],[170,187,204]])
        with self.assertRaises(ValueError):pcd(header+a.tobytes()[:-1])

    def test_camera_holdout_and_sensor_depth_units(self):
        rng=np.random.default_rng(7);xyz=rng.normal(0,.03,(60,3))+[0,0,.7]
        P=np.array([[1000.,0.,960.,0.],[0.,1000.,540.,0.],[0.,0.,1.,0.]])
        pixels=project(xyz,P);fitted,error=camera(xyz,pixels)
        self.assertLess(error,1e-8)
        np.testing.assert_allclose(project(xyz,fitted),pixels,atol=1e-8)
        with self.assertRaises(ValueError):camera(xyz,rng.normal(size=(60,2))*1000)
        yy,xx=np.mgrid[:120:5,:105:5];uv=np.column_stack([xx.ravel(),yy.ravel()])
        points=np.column_stack([np.zeros(len(uv)),np.zeros(len(uv)),np.ones(len(uv))*.75])
        image=depth_image(points,uv,TARGET)
        self.assertEqual(image.dtype,np.uint16)
        self.assertEqual(image[50,50],750)

    def test_body_crop_ignores_depth_and_source_coordinate_order(self):
        joints=np.column_stack([np.linspace(40,180,17),np.linspace(100,150,17),np.ones(17)*1000])
        expected=frame_box(joints)
        self.assertEqual(expected,[90,12,160,208])
        joints[:,2]=np.nan
        self.assertEqual(frame_box(joints),expected)
        bad=joints.copy();bad[0,0]=200
        self.assertIsNone(frame_box(bad))
        bad=joints.copy();bad[:,1]=np.linspace(10,410,17)
        self.assertIsNone(frame_box(bad))
        self.assertIsNone(frame_box(np.zeros((17,3))))

    def test_alternative_retrieval_names_are_not_walking_still(self):
        from itml_repro.reid import select_groups
        rows=[dict(sample_id=f'{s}_{p}_{f}',identity=str(p),subset=s,frame=f)
              for s in ('train','normal_A','normal_B') for p in range(2) for f in range(3)]
        groups,_=select_groups(rows,dict(test_subsets=['normal_A','normal_B'],identity_split=[1,1],shots=3,test_identities=2))
        self.assertEqual(set(groups),{'train','val','normal_A','normal_B'})
