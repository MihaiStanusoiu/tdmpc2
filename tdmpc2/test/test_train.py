import unittest

import hydra
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

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

    def test_zeroed_sequences(self):
        obs_dim = 4
        ac_dim = 1
        hidden_dim = 16
        batch_size = 2
        H = 10
        cfc = CfC(obs_dim + ac_dim, hidden_dim, None, return_sequences=False, batch_first=False)

        obs = torch.zeros(H, batch_size, obs_dim)
        ac = torch.zeros(H, batch_size, ac_dim)
        dt = torch.ones(H, batch_size, 1)

        obs[4:] = torch.randn(batch_size, obs_dim)
        ac[4:] = torch.randn(batch_size, ac_dim)
        rand_nonzero_in_batch = np.random.randint(0, batch_size)
        obs[:4, rand_nonzero_in_batch] = torch.randn(4, obs_dim)
        ac[:4, rand_nonzero_in_batch] = torch.randn(4, ac_dim)
        h = torch.zeros(batch_size, hidden_dim)
        for _, (obs_t, ac_t, t) in enumerate(zip(obs.unbind(0), ac.unbind(0), dt.unbind(0))):
            non_zero_indices = torch.nonzero(obs_t.sum(-1), as_tuple=True)[0]
            if len(non_zero_indices) > 0:
                valid_z = obs_t[non_zero_indices]
                valid_a = ac_t[non_zero_indices]
                inp = torch.cat([valid_z, valid_a], dim=-1).unsqueeze(0)
                _, h[non_zero_indices] = cfc(inp, hx=h[non_zero_indices], timespans=t)

    def test_padded_sequences(self):
        obs_dim = 4
        ac_dim = 1
        hidden_dim = 16
        batch_size = 2
        H = 10
        cfc = CfC(obs_dim + ac_dim, hidden_dim, None, return_sequences=False, batch_first=False)
        # create random observations and actions sequences of variable length <= H
        obs_1 = torch.randn(3, batch_size, obs_dim)
        obs_2 = torch.randn(5, batch_size, obs_dim)
        ac_1 = torch.randn(3, batch_size, ac_dim)
        ac_2 = torch.randn(5, batch_size, ac_dim)
        padded_input = pad_sequence([torch.cat([obs_1, ac_1], dim=-1), torch.cat([obs_2, ac_2], dim=-1)], batch_first=False)
        mask = torch.zeros_like(padded_input[..., 0])

        packed_inputs = pack_padded_sequence(padded_input, [3, 5], batch_first=False, enforce_sorted=False)
        packed_output, h = cfc(packed_inputs)
        output, _ = pad_packed_sequence(packed_output, batch_first=False)

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
