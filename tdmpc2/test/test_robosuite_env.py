import unittest

import numpy as np
import robosuite as suite
from numpy.ma.testutils import assert_equal
from robosuite.scripts.print_robot_action_info import controller_config


class RobosuiteTestCase(unittest.TestCase):
    def test_env_termination(self):
        controller_config = suite.controllers.load_part_controller_config(default_controller='JOINT_TORQUE')
        env = suite.make(
            env_name='PickPlace',
            robots='IIWA',
            controller_config=controller_config,
            has_renderer=True,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            use_object_obs=True,
            horizon=200
        )
        t = 0
        done = False
        while not done:
            action = np.random.randn(*env.action_spec[0].shape) * 0.1
            obs, reward, done, info = env.step(action)
            t += 1

        assert_equal(t, 200)
        assert('success' in info.keys())

if __name__ == '__main__':
    unittest.main()
