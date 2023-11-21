import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tf_agents.environments import py_environment, tf_py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

class CustomEnvironment(py_environment.PyEnvironment):
    def __init__(self, image_path):
        # Завантаження зображення місцевості
        self.terrain_map = plt.imread(image_path)
        self.start_position = [0.5, 0]
        self.agent_position = self.start_position  # Початкове положення агента

        # Визначення специфікацій для станів та дій
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(2,), dtype=np.float32, minimum=0, maximum=1, name='observation')
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(1,), dtype=np.int32, minimum=0, maximum=4, name='action')

        # Ініціалізація стану
        self._state = 0
        self._episode_ended = False

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._state = 0
        self._episode_ended = False
        return ts.restart(np.array([self.start_position], dtype=np.float32))

    def _step(self, action):
        # Логіка руху агента
        # Оновлюєте self.agent_position залежно від дії 'action'
        # та властивостей місцевості
        # ...
        reward=-100
        if self._episode_ended:
            return ts.termination(np.array([self.agent_position], dtype=np.float32), reward)
        else:
            return ts.transition(np.array([self.agent_position], dtype=np.float32), reward=1.0, discount=0.9)

# Створення екземпляру середовища
env = CustomEnvironment(image_path='./img/output1small.png')
tf_env = tf_py_environment.TFPyEnvironment(env)
