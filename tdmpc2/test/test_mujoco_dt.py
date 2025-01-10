import unittest

import hydra
from envs.pomdp_gym import make_env

class TestMujoco(unittest.TestCase):
    @hydra.main(config_name='config', config_path='.')
    def test_variable_timesteps(self, cfg):
        env = make_env(cfg)
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
