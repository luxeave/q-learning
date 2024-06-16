# pip install 'gym[all]'

import gym
import pygame
from gym.utils.play import play

import numpy as np

# mapping = {(pygame.K_LEFT,): 0, (pygame.K_RIGHT,): 1}
# play(gym.make("ALE/Pong-v5", render_mode="rgb_array"), keys_to_action=mapping)

# play(gym.make("CarRacing-v2", render_mode="rgb_array"), keys_to_action={
#                                                "w": np.array([0, 0.7, 0]),
#                                                "a": np.array([-1, 0, 0]),
#                                                "s": np.array([0, 0, 1]),
#                                                "d": np.array([1, 0, 0]),
#                                                "wa": np.array([-1, 0.7, 0]),
#                                                "dw": np.array([1, 0.7, 0]),
#                                                "ds": np.array([1, 0, 1]),
#                                                "as": np.array([-1, 0, 1]),
#                                               }, noop=np.array([0,0,0]))

# play(gym.make("LunarLander-v2", render_mode="rgb_array"), keys_to_action={
#                                                "w": 2,
#                                                "a": 1,
#                                                "s": 0,
#                                                "d": 3,
#                                                "wa": 2,
#                                                "dw": 2,
#                                                "ds": 2,
#                                                "as": 2,
#                                               }, noop=0)