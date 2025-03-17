import unittest

import hydra
import torch

from common.buffer import Buffer
from common.logger import Logger
from common.parser import parse_cfg
from ncps.torch import CfC


class MyTestCase(unittest.TestCase):


    @staticmethod
    def _mask(value, mask):
        # Apply element-wise multiplication with broadcasting in PyTorch
        return value * mask.to(value.dtype)

    @hydra.main(config_name='config', config_path='.')
    def test_buffer_checkpointing(self, cfg: dict):
        cfg = parse_cfg(cfg)
        logger = Logger(cfg)
        buffer = Buffer(cfg)

    def test_cfc_zero_input(self):
        obs_dim = 4
        ac_dim = 1
        hidden_dim = 16
        batch_size = 256
        H = 10
        cfc = CfC(obs_dim + ac_dim, hidden_dim, None, return_sequences=False, batch_first=False)
        obs = torch.zeros(1, batch_size, obs_dim, dtype=torch.float32)
        ac = torch.zeros(1, batch_size, ac_dim, dtype=torch.float32)
        input = torch.cat([obs, ac], dim=-1)
        _, h1 = cfc(input)
        _, h2 = cfc(input, hx=h1)
        self.assertTrue(torch.allclose(h1, h2))
        self.assertTrue(torch.allclose(h1, torch.zeros(batch_size, hidden_dim)))

    def test_timesteps(self):
        obs_dim = 4
        ac_dim = 1
        hidden_dim = 16
        batch_size = 256
        H = 10
        cfc = CfC(obs_dim + ac_dim, hidden_dim, None, return_sequences=False, batch_first=False)
        # random observations and actions
        obs = torch.randn(H, batch_size, obs_dim)
        ac = torch.randn(H, batch_size, ac_dim)
        dt = torch.ones(H, batch_size, 1)
        # compute hidden state with whole sequence as input
        _, h1 = cfc(torch.cat([obs, ac], dim=-1), timespans=dt)
        h2 = None
        # compute hidden state with each timestep as input
        for _, (obs_t, ac_t, t) in enumerate(zip(obs.unbind(0), ac.unbind(0), dt.unbind(0))):
            _, h2 = cfc(torch.cat([obs_t, ac_t], dim=-1).unsqueeze(0), hx=h2, timespans=t)

        # Check if they're equal
        self.assertTrue(torch.allclose(h1, h2))

    def test_initial_h_masking(self):
        B = 256
        T = 6
        H = 512
        is_first = torch.randint(0, 2, (T, B, 1), dtype=torch.float32)
        initial_h = torch.zeros(B, H)
        h_before = torch.randn(T, B, H)
        hs = []
        for t, (is_first_t, h_before_t) in enumerate(zip(is_first.unbind(0), h_before.unbind(0))):
            h_after_t = self._mask(h_before_t, 1.0 - is_first_t.float())
            h_after_t = h_after_t + self._mask(initial_h, is_first_t.float())
            hs.append(h_after_t)
        h_after = torch.stack(hs, dim=0)
        # permute (T, B, H) -> (B, T, H)
        h_before = h_before.permute(1, 0, 2)
        h_after = h_after.permute(1, 0, 2)
        is_first = is_first.permute(1, 0, 2)
        initial_h = initial_h.unsqueeze(1).expand(-1, T, -1)

        self.assertTrue(torch.allclose(h_after[is_first.squeeze().bool()], initial_h[is_first.squeeze().bool()]))
        self.assertTrue(torch.allclose(h_after[~is_first.squeeze().bool()], h_before[~is_first.squeeze().bool()]))


if __name__ == '__main__':
    unittest.main()
