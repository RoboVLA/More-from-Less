"""Analytic fixtures only: never experiment data or evidence of robot safety."""
import threading
import unittest
import numpy as np
from morefromless.method.expansion import Rollout,acceptance,expand,real_dominant_weights,local_states
from morefromless.method.control import (AsyncReferences,ImageReference,OnlineController,
    shared_qp,task_masked_reference,pixel_translation,gate_weight)


def qp_arguments():
    return dict(lower=np.full(12,-1.),upper=np.full(12,1.),step_limit=np.ones(12),change_limit=np.ones(12),
                hard_G=np.empty((0,12)),hard_h=np.empty(0),soft_G=np.empty((0,12)),soft_h=np.empty(0),
                slack_max=np.empty(0),weight=np.ones(12),slack_penalty=10.)


class ExpansionTests(unittest.TestCase):
    def candidate(self,**kwargs):
        fields=dict(identifier='accepted',states=(0,1),actions=(1,),complete=True,
                    task_success=True,physics_valid=True,safety_valid=True)
        fields.update(kwargs);return Rollout(**fields)
    def test_complete_joint_acceptance(self):
        self.assertTrue(acceptance(self.candidate()))
        for name in ('complete','task_success','physics_valid','safety_valid'):
            self.assertFalse(acceptance(self.candidate(**{name:False})))
        self.assertFalse(acceptance(self.candidate(states=(0,))))
        self.assertFalse(acceptance(self.candidate(split='test')))
    def test_quality_cannot_overwhelm_real_mass(self):
        rw,vw=real_dominant_weights(2,[self.candidate(quality=1e8),self.candidate(quality=1)],.7)
        self.assertAlmostEqual(rw.sum(),.7);self.assertAlmostEqual(vw.sum(),.3)
    def test_first_failures_retraining_and_stop(self):
        calls=[]
        def rollout(policy,state,residual):
            if state=='discovery':return self.candidate(task_success=False,failure_state='failure')
            return self.candidate(identifier=state,task_success=state=='success')
        def retrain(policy,real,virtual,rw,vw):
            calls.append(tuple(t.identifier for t in virtual));return policy+1
        policy,accepted,log=expand(0,['real'],['discovery'],rollout=rollout,
            sample_near=lambda failure:['success','reject'],feasible=lambda state:True,retrain=retrain,
            max_rounds=3,real_mass=.7,residual_explorer='simulation-only')
        self.assertEqual(calls,[('success',)])
        self.assertEqual(policy,1);self.assertEqual(len(accepted),1)
        self.assertEqual(log[-1]['stop_reason'],'no_new_accepted')
        self.assertEqual(log[0]['candidates'],2)
    def test_bounded_local_sampling(self):
        states=local_states([.9],.3,[0],[1],20,np.random.default_rng(2))
        self.assertTrue(all(.6<=s[0]<=1 for s in states))


class ControlTests(unittest.TestCase):
    def test_qp_projection_and_bounded_slack(self):
        args=qp_arguments();raw=np.zeros(12);raw[0]=2
        out=shared_qp(raw,np.zeros(12),np.zeros(12),**args)
        np.testing.assert_allclose(out.command[0],1,atol=1e-7)
        args.update(soft_G=np.eye(12)[:1],soft_h=np.array([.8]),slack_max=np.array([.2]))
        out=shared_qp(np.zeros(12),np.zeros(12),np.zeros(12),**args)
        self.assertIsNotNone(out.command);self.assertLessEqual(out.slack[0],.2+1e-8)
        self.assertGreaterEqual(out.command[0]+out.slack[0],.8-1e-8)
        self.assertAlmostEqual(out.command[0],8/11,places=6)
    def test_qp_impossible_hard_constraint_not_relaxed(self):
        args=qp_arguments();args.update(hard_G=np.eye(12)[:1],hard_h=np.array([2.]))
        out=shared_qp(np.zeros(12),np.zeros(12),np.zeros(12),max_sweeps=50,**args)
        self.assertIsNone(out.command)
    def test_task_mapping_grippers_and_unreachable(self):
        left=np.zeros((6,10));left[:3,:3]=np.eye(3)
        right=np.zeros((6,10));right[:3,5:8]=np.eye(3)
        base=np.ones(12)*.5;current=np.zeros(12);S=np.eye(6)[:3]
        result=task_masked_reference(current,base,[left,right],[S,S],
                [np.array([.1,0,0,0,0,0]),np.zeros(6)],damping=.01,max_residual=.001,max_increment=.2)
        np.testing.assert_array_equal(result[[5,11]],base[[5,11]])
        self.assertAlmostEqual(result[0],.1/1.0001)
        with self.assertRaises(ValueError):
            task_masked_reference(current,base,[left,right],[S,S],[np.ones(6),np.zeros(6)],
                                  damping=.01,max_residual=.001,max_increment=.2)
    def test_current_depth_mapping(self):
        K=np.diag([100.,100.,1.])
        np.testing.assert_allclose(pixel_translation([10,20],2,K,np.eye(3)),[.2,.4,0])
        with self.assertRaises(ValueError):pixel_translation([1,2],0,K,np.eye(3))
    def test_async_age_is_capture_time(self):
        block=threading.Event()
        def generate(rgb,language):block.wait(2);return np.zeros((2,2)),np.array([0.,1.])
        refs=AsyncReferences(generate)
        try:
            self.assertTrue(refs.request(np.zeros((2,2,3)),'fixture',5.))
            self.assertIsNone(refs.poll(120.));self.assertTrue(refs.pending)
            block.set();refs.future.result(timeout=3)
            ref=refs.poll(125.);self.assertEqual(ref.observation_time,5.)
            self.assertEqual(ref.received_time,125.)
        finally:block.set();refs.close()
    def test_single_output_filter_low_risk_invalid_and_reject(self):
        class References:
            pending=False
            image=ImageReference(0,1,np.zeros((2,2)),np.array([0,1]))
            def poll(self,now):return self.image
            def request(self,*args):self.pending=True;return True
        refs=References();calls=[]
        def constraints(raw,*args):calls.append(raw.copy());return qp_arguments()
        observation=dict(rgb_global=np.zeros((2,2,3)),rgb_left=0,rgb_right=0,language='fixture',joints=np.zeros(12),timestamp=2)
        def policy(obs):
            self.assertEqual(set(obs),{'rgb_global','rgb_left','rgb_right','language','joints'})
            return np.full((32,12),.2)
        controller=OnlineController(policy,refs,lambda *a:np.full(12,.4),constraints,
                lambda *a:True,thresholds=np.ones(4),scales=np.ones(4),weights=np.ones(4),bias=0,
                max_reference_age=10,gate_smoothing=.5)
        high=controller.tick(observation,np.zeros(12),np.zeros(12),np.full(4,2.),2)
        self.assertGreater(high.alpha,0);np.testing.assert_allclose(high.command[[5,11]],[.2,.2])
        low=controller.tick(observation,np.zeros(12),np.zeros(12),np.zeros(4),2)
        self.assertEqual(low.alpha,0);self.assertEqual(len(calls),2)
        controller.map_reference=lambda *a:(_ for _ in ()).throw(ValueError('invalid'))
        invalid=controller.tick(observation,np.zeros(12),np.zeros(12),np.full(4,2.),2)
        self.assertEqual(invalid.alpha,0);self.assertEqual(len(calls),3)
        controller.map_reference=lambda *a:np.full(12,.4)
        waiting=controller.tick(observation,np.zeros(12),np.zeros(12),np.full(4,2.),2,request_reference=True)
        self.assertEqual(waiting.alpha,0);self.assertEqual(len(calls),4)
        controller.nonlinear_check=lambda *a:False
        rejected=controller.tick(observation,np.zeros(12),np.zeros(12),np.zeros(4),2)
        self.assertIsNone(rejected.command);self.assertEqual(rejected.mode,'hold')
    def test_gate_nonfinite_rejected(self):
        with self.assertRaises(ValueError):gate_weight([np.nan]*4,[1]*4,[1]*4,[1]*4,0,True)


if __name__=='__main__':unittest.main()
