"""CPU indexing/gradient checks with real official masks and action head.

Does not load pretrained weights; P1 and checkpoint runtime remain separate.
"""
import pytest
torch=pytest.importorskip('torch')
pytest.importorskip('prismatic')
from types import SimpleNamespace
from prismatic.models.action_heads import L1RegressionActionHead
from vla_attention.adapters.oft_forward import forward_l1


class Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tokens=torch.nn.Parameter(torch.randn(1,65,16))

    def forward(self, **kwargs):
        return SimpleNamespace(hidden_states=[self.tokens],projector_features=self.tokens[:,1:6])


def batch():
    labels=torch.full((1,60),-100,dtype=torch.long)
    labels[:,3:59]=31900
    labels[:,-1]=2
    return dict(labels=labels,input_ids=torch.zeros_like(labels),attention_mask=torch.ones_like(labels),
                pixel_values=torch.zeros(1,12,224,224),proprio=torch.zeros(1,8),actions=torch.zeros(1,8,7))


def test_full_chunk_and_policy_gradient():
    policy=Policy(); head=L1RegressionActionHead(input_dim=16,hidden_dim=16)
    result=forward_l1(policy,head,torch.nn.Identity(),batch(),device='cpu',visual_token_count=4)
    assert result['prediction'].shape==(1,8,7)
    assert torch.allclose(result['loss'],result['prediction'].abs().mean())
    result['loss'].backward()
    assert policy.tokens.grad is not None and policy.tokens.grad.abs().sum()>0


def test_reject_truncated_action_tokens():
    data=batch(); data['labels'][:,3]=-100
    with pytest.raises(ValueError,match='action token'):
        forward_l1(Policy(),L1RegressionActionHead(input_dim=16,hidden_dim=16),torch.nn.Identity(),data,
                   device='cpu',visual_token_count=4)
