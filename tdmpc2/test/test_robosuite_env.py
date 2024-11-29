import unittest

import numpy as np
import robosuite as suite
from numpy.ma.testutils import assert_equal


class RobosuiteTestCase(unittest.TestCase):
    def test_env_termination(self):
        # controller_config = suite.controllers.load_composite_controller_config(robot='IIWA')
        #
        # controller_config['body_parts']['right'] = {
        #     "type": "JOINT_TORQUE",
        #     "input_max": 1,
        #     "input_min": -1,
        #     "output_max": 0.1,
        #     "output_min": -0.1,
        #     "torque_limits": None,
        #     "interpolation": None,
        #     "ramp_ratio": 0.2,
        #     "gripper": {
        #         "type": "GRIP"
        #     }
        #   }

        controller_config = suite.load_part_controller_config(default_controller='JOINT_TORQUE')

        env = suite.make(
            env_name='PickPlace',
            robots='IIWA',
            controller_configs=controller_config,
            has_renderer=True,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            use_object_obs=True,
            horizon=200
        )
        env = wrappers.GymWrapper(env)
        t = 0
        done = False
        while not done:
            action = np.random.randn(*env.action_spec[0].shape) * 0.1
            obs, reward, done, _, info = env.step(action)
            t += 1

        assert_equal(t, 200)
        assert('success' in info.keys())

if __name__ == '__main__':
    unittest.main()
