import gym


class MujocoWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env

    def render(self, mode='rgb_array', width=384, height=384, camera_id=0):
        return self.env.render(mode=mode)