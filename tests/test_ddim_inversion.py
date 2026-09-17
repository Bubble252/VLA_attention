import unittest

from vla_attention.teachers.ddim import invert_latent


class FakeOutput:
    def __init__(self, sample): self.sample = sample


class FakeUNet:
    def __call__(self, current, timestep, encoder_hidden_states):
        return FakeOutput(current + timestep + encoder_hidden_states)


class FakeScheduler:
    def set_timesteps(self, steps, device): self.timesteps = list(range(steps)); self.device = device
    def step(self, noise, timestep, current):
        return type("Step", (), {"prev_sample": current + noise})()


class DDIMInversionTests(unittest.TestCase):
    def test_visits_every_timestep_in_order(self):
        result = invert_latent(FakeUNet(), FakeScheduler(), 1, 2, steps=3, device="cuda")
        self.assertEqual(result, 26)
