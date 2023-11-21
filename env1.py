import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tf_agents.environments import py_environment, tf_py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from joblib import load

class CustomEnvironment(py_environment.PyEnvironment):
    def __init__(self, image_path, agent_speed=0.05):
        # Завантаження зображення місцевості
        self.terrain_map = plt.imread(image_path)
        self.image_height, self.image_width, _ = self.terrain_map.shape
        self.start_position = [0.5, 0]
        self.agent_position = self.start_position  # Початкове положення агента

        self.model_speed_coef = load('./img_analyse/model.joblib')
        self.agent_speed = agent_speed

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
        
        
        # Припустимо, що дії визначені як:
        # 0 - вгору, 1 - вниз, 2 - вліво, 3 - вправо
        speed_coef = self.get_speed_coef(self.agent_position)
        if action == 0:  # вгору
            self.agent_position[1] += self.agent_speed*speed_coef
        elif action == 1:  # вниз
            self.agent_position[1] -= self.agent_speed*speed_coef
        elif action == 2:  # вліво
            self.agent_position[0] -= self.agent_speed*speed_coef
        elif action == 3:  # вправо
            self.agent_position[0] += self.agent_speed*speed_coef


        reward=-100
        if self._episode_ended:
            return ts.termination(np.array([self.agent_position], dtype=np.float32), reward)
        else:
            return ts.transition(np.array([self.agent_position], dtype=np.float32), reward=1.0, discount=0.9)

    def get_pixel_color(self, agent_position):
        # Перетворення нормалізованих координат у індекси пікселів
        x_index = int(agent_position[0] * self.image_width)
        y_index = int(agent_position[1] * self.image_height)

        # Перевірка, щоб індекси не вийшли за межі зображення
        x_index = max(0, min(x_index, self.image_width - 1))
        y_index = max(0, min(y_index, self.image_height - 1))

        # Отримання RGB значення пікселя
        pixel_color = self.terrain_map[y_index, x_index, :3]
        return pixel_color
    
    def get_speed_coef(self, position):
            # Отримання RGB кольору на позиції агента
            rgb = self.get_pixel_color(position)
            # Обчислення коефіцієнта швидкості
            speed_coef = self.model_speed_coef.predict([rgb])[0]
            return speed_coef
    
# Створення екземпляру середовища
env = CustomEnvironment(image_path='./img/output1small.png')
tf_env = tf_py_environment.TFPyEnvironment(env)
