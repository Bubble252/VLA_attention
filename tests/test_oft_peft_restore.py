"""Exercise the pinned PEFT save/load path without occupying a GPU."""
import copy

import pytest

torch = pytest.importorskip('torch')
peft = pytest.importorskip('peft')
from vla_attention.resume_probe import optimizer_probe


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(8, 7)

    def forward(self, x):
        return self.proj(x)


def test_bfloat16_lora_roundtrip_and_adam_continuation(tmp_path):
    torch.manual_seed(17)
    base = TinyPolicy().bfloat16()
    initial = copy.deepcopy(base.state_dict())
    policy = peft.get_peft_model(base, peft.LoraConfig(
        r=4, lora_alpha=4, lora_dropout=0., target_modules=['proj'], init_lora_weights='gaussian'))
    optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [100000], gamma=.1)
    x = torch.randn(3, 8).bfloat16()
    def loss(model):
        return model(x).float().square().mean()
    for _ in range(2):
        optimizer_probe([('policy', policy)], optimizer, scheduler, lambda: loss(policy))
    expected_prediction = policy.eval()(x).detach().clone()
    policy.save_pretrained(tmp_path / 'adapter')
    saved_optimizer = copy.deepcopy(optimizer.state_dict())
    saved_scheduler = copy.deepcopy(scheduler.state_dict())
    expected_probe = optimizer_probe([('policy', policy)], optimizer, scheduler, lambda: loss(policy))
    replacement = TinyPolicy().bfloat16()
    replacement.load_state_dict(initial)
    restored = peft.PeftModel.from_pretrained(replacement, tmp_path / 'adapter', is_trainable=True)
    assert torch.equal(restored.eval()(x), expected_prediction)
    optimizer2 = torch.optim.AdamW([p for p in restored.parameters() if p.requires_grad], lr=5e-4)
    scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer2, [100000], gamma=.1)
    optimizer2.load_state_dict(saved_optimizer); scheduler2.load_state_dict(saved_scheduler)
    actual_probe = optimizer_probe([('policy', restored)], optimizer2, scheduler2, lambda: loss(restored))
    assert actual_probe == expected_probe
