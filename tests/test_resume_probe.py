import copy
import pytest
torch=pytest.importorskip('torch')
from vla_attention.resume_probe import optimizer_probe


def test_continuation_requires_optimizer_moments():
    torch.manual_seed(1)
    m=torch.nn.Linear(3,2)
    opt=torch.optim.AdamW(m.parameters(),lr=.01)
    sched=torch.optim.lr_scheduler.MultiStepLR(opt,[2],gamma=.1)
    x=torch.tensor([[1.,2.,3.]])
    fn=lambda:(m(x)**2).mean()
    optimizer_probe([('m',m)],opt,sched,fn)
    saved=[copy.deepcopy(z.state_dict()) for z in [m,opt,sched]]
    expected=optimizer_probe([('m',m)],opt,sched,fn)
    m.load_state_dict(saved[0]);opt.load_state_dict(saved[1]);sched.load_state_dict(saved[2])
    assert optimizer_probe([('m',m)],opt,sched,fn)==expected
    m.load_state_dict(saved[0]);opt=torch.optim.AdamW(m.parameters(),lr=.01)
    sched=torch.optim.lr_scheduler.MultiStepLR(opt,[2],gamma=.1)
    assert optimizer_probe([('m',m)],opt,sched,fn)['parameter_sha256']!=expected['parameter_sha256']
