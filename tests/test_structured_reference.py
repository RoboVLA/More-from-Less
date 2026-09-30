"""PyTorch gradient tests with analytic geometry, not a robot evaluation."""
import importlib.util
import unittest
HAS_TORCH=importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    import torch
    from morefromless.method.structured import LossSettings,structured_loss,rotation_log,train_step
    from morefromless.method.kinematics import revolute_rotation,serial_fk


@unittest.skipUnless(HAS_TORCH,'Install optional torch dependency for gradient tests')
class StructuredTests(unittest.TestCase):
    def settings(self):
        return LossSettings(torch.zeros(12,dtype=torch.double),torch.ones(12,dtype=torch.double),
             torch.zeros(7,dtype=torch.double),torch.ones(7,dtype=torch.double),torch.ones(7,dtype=torch.double),.4,.2,.1,.1)
    def geometry(self,actions,context):
        batch=actions.shape[:-1]
        left=torch.eye(4,dtype=actions.dtype).expand(*batch,4,4).clone()
        right=left.clone();right[...,0,3]=actions[...,6]-actions[...,0]
        right[...,:3,:3]=revolute_rotation(torch.tensor([0.,0.,1.]),actions[...,7])
        return left,right,(actions[...,0]-actions[...,6]).abs()+.01,actions[...,1]+.2
    def test_gradient_and_context_isolation(self):
        pred=torch.full((1,32,12),.1,dtype=torch.double,requires_grad=True)
        target=torch.zeros_like(pred,requires_grad=True)
        contact=torch.ones((1,32),dtype=torch.double,requires_grad=True)
        valid=torch.ones((1,32),dtype=torch.bool)
        losses=structured_loss(pred,target,valid,{'contact':contact},self.geometry,self.settings())
        losses['loss'].backward()
        self.assertTrue(torch.isfinite(pred.grad).all());self.assertGreater(float(pred.grad.abs().sum()),0)
        self.assertIsNone(target.grad);self.assertIsNone(contact.grad)
    def test_rotation_log_gradient(self):
        axis=torch.tensor([0.,0.,1.],dtype=torch.double)
        for angle in (0.,.1):
            a=torch.tensor([angle],dtype=torch.double,requires_grad=True)
            self.assertTrue(torch.autograd.gradcheck(lambda x:rotation_log(revolute_rotation(axis,x)),(a,)))
    def test_padding_and_fixed_scales(self):
        pred=torch.zeros((1,32,12),dtype=torch.double);pred[:,1:]=float('nan')
        valid=torch.zeros((1,32),dtype=torch.bool);valid[:,0]=True
        losses=structured_loss(pred,torch.zeros_like(pred),valid,{},self.geometry,self.settings())
        self.assertTrue(torch.isfinite(losses['loss']))
        settings=self.settings();settings.relation_scale[0]=0
        with self.assertRaises(ValueError):structured_loss(pred,torch.zeros_like(pred),valid,{},self.geometry,settings)
    def test_serial_fk_has_action_gradient(self):
        q=torch.zeros((1,5),dtype=torch.double,requires_grad=True)
        origins=torch.eye(4,dtype=torch.double).repeat(5,1,1);origins[:,0,3]=.1
        axes=torch.tensor([[0.,0.,1.]]*5,dtype=torch.double)
        ee,_=serial_fk(q,origins,axes,torch.eye(4,dtype=torch.double),torch.eye(4,dtype=torch.double))
        ee[...,1,3].sum().backward()
        self.assertGreater(float(q.grad.abs().sum()),0)
    def test_optimizer_step_and_no_extra_inference_inputs(self):
        class Policy(torch.nn.Module):
            def __init__(self):super().__init__();self.action=torch.nn.Parameter(torch.full((1,32,12),.1,dtype=torch.double))
            def forward(self,**obs):return self.action
        model=Policy();optimizer=torch.optim.SGD(model.parameters(),lr=.01)
        obs={k:None for k in ('rgb_global','rgb_left','rgb_right','language','joints')}
        before=model.action.detach().clone()
        train_step(model,optimizer,obs,torch.zeros_like(before),torch.ones((1,32),dtype=torch.bool),{},self.geometry,self.settings())
        self.assertFalse(torch.equal(model.action,before))
        with self.assertRaises(ValueError):
            train_step(model,optimizer,{**obs,'contact':1},torch.zeros_like(before),torch.ones((1,32),dtype=torch.bool),{},self.geometry,self.settings())

if __name__=='__main__':unittest.main()
