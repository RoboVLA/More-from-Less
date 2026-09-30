"""Differentiable revolute-chain FK from explicit measured/URDF transforms.

No robot dimensions or collision models are guessed. Origins are parent-to-joint
homogeneous transforms at q=0; axes are unit vectors in local joint frames.
Use actual collision geometry callbacks for distances in structured_loss.
"""
import torch


def revolute_rotation(axis, angle):
    axis=axis.to(dtype=angle.dtype,device=angle.device)
    x,y,z=axis.unbind();zero=x*0
    K=torch.stack((zero,-z,y,z,zero,-x,-y,x,zero)).reshape(3,3)
    I=torch.eye(3,dtype=angle.dtype,device=angle.device)
    return I+angle.sin()[...,None,None]*K+(1-angle.cos())[...,None,None]*(K@K)


def serial_fk(joints, origins, axes, base, tool):
    if joints.shape[-1]!=5 or origins.shape!=(5,4,4) or axes.shape!=(5,3) or base.shape!=(4,4) or tool.shape!=(4,4):
        raise ValueError("Each arm requires five joints and explicit transforms")
    if not torch.allclose(axes.norm(dim=-1),torch.ones(5,dtype=axes.dtype,device=axes.device),atol=1e-6):
        raise ValueError("Joint axes must be normalized")
    transform=base.to(joints).expand(*joints.shape[:-1],4,4)
    links=[]
    for index in range(5):
        rotation=revolute_rotation(axes[index],joints[...,index])
        column=torch.zeros(*joints.shape[:-1],3,1,dtype=joints.dtype,device=joints.device)
        row=torch.tensor([0.,0.,0.,1.],dtype=joints.dtype,device=joints.device).expand(*joints.shape[:-1],1,4)
        local=torch.cat((torch.cat((rotation,column),-1),row),-2)
        transform=transform@origins[index].to(joints)@local
        links.append(transform)
    return transform@tool.to(joints),torch.stack(links,-3)


class BimanualGeometry:
    """Collision query(links_left, links_right, absolute12, context) -> d_self,d_env.

    The caller's query must include gripper/link geometry and current object
    context. This class does not replace that query with endpoint distances.
    """
    def __init__(self,left,right,collision_query):
        self.left=left;self.right=right;self.query=collision_query
    def __call__(self,actions,context):
        left,ll=serial_fk(actions[...,:5],**self.left)
        right,rl=serial_fk(actions[...,6:11],**self.right)
        ds,de=self.query(ll,rl,actions,context)
        return left,right,ds,de
