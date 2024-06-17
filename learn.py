import gym 
import numpy as np
from gym.utils.save_video import save_video
import matplotlib.pyplot as plt

#--------- ENVIRONMENT ------------
# render_mode="rgb_array_list"
render_mode = "none"
#env = gym.make("MountainCar-v0", render_mode)
env = gym.make("MountainCar-v0")

#--------- CONFIGS ------------
LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 25000
epsilon = 0.5 # exploration rate
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2 # result always INT
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)
SHOW_EVERY = 500

#--------- CALCULATE BUCKET SIZE ------------
OBSERVATION_SPACE_SIZE = len(env.observation_space.high) # 2 -> [position, velocity] -> [0.6  0.07] 
DISCRETE_OS_SIZE = [20] * OBSERVATION_SPACE_SIZE # 20 BUCKETS OF VALUES -> [20, 20]
CONTINUOUS_OS_SIZE = env.observation_space.high - env.observation_space.low # [1.8000001 0.14]
BUCKET_SIZE = CONTINUOUS_OS_SIZE / DISCRETE_OS_SIZE # [0.09  0.007]

#--------- CREATE Q-TABLE ------------
TABLE_SIZE = DISCRETE_OS_SIZE + [env.action_space.n] # [20, 20] + [3] -> [20, 20, 3]
q_table = np.random.uniform(low=-2, high=0, size=TABLE_SIZE) # initialized with random values
# print(q_table.shape) # [20, 20, 3]
# print(q_table)

ep_rewards = []
aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low)/BUCKET_SIZE
    return tuple(discrete_state.astype(int))  # we use this tuple to look up the 3 Q values for the available actions in the q-table

render_once = False

for episode in range(EPISODES):    

    episode_reward = 0

    # it's good if we can visualize this via animating the act of retrieving q_values at coordinate `discrete_state`
    # print(env.reset()) # (array([-0.5832684,  0.       ], dtype=float32), {})
    discrete_state = get_discrete_state(env.reset()[0]) # (7, 10)

    terminated = False 

    if render_once == False:
        frame_index = 0

    while not terminated:
        #----------------- DECIDE ACTION -----------------
        if np.random.random() > epsilon:
            # Get action from Q table
            action = np.argmax(q_table[discrete_state])
        else:
            # Get random action
            action = np.random.randint(0, env.action_space.n)

        #----------------- TAKE ACTION -----------------
        new_state, reward, terminated, truncated, info = env.step(action)  # (array([-0.5832684,  0.       ], dtype=float32), {})
        new_discrete_state = get_discrete_state(new_state) #  (9, 10)

        #----------------- INCREMENT REWARD -----------------
        episode_reward += reward     

        if render_once == False:
            frame_index += 1
        
        if not terminated:
            # q_table[new_discrete_state] -> [-0.35635789 -0.86828597 -0.84140828]

            # it's good if we can visualize this via animating the act of retrieving q_values at coordinate `new_discrete_state`

            max_future_q = np.max(q_table[new_discrete_state]) # -0.84140828 - the max q value of best action
            current_q = q_table[discrete_state + (action,)] # -0.35635789 - the current q value of taken action

            # using Qnew formula 
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

            # it's good if we can visualize this via animating the act of updating single q_value at coordinate `discrete_state + action`

            # update the table entry
            q_table[discrete_state+(action, )] = new_q
        elif new_state[0] >= env.goal_position: # goal == 0.5
            #print("new_state[0]: ", new_state[0])
            #print(f"we made it on episode {episode}")
            q_table[discrete_state + (action,)] = 0 # set state + action -> q_values to zero -> we win the game

            if render_once == False and render_mode == "rgb_array_list":
                save_video(
                    env.render(),
                    "videos",
                    fps=env.metadata["render_fps"],
                    step_starting_index=frame_index-20,
                    episode_index=episode
                )
                render_once = True

            observation, info = env.reset()

        discrete_state = new_discrete_state

        # END WHILE
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value

    #----------------- AGGREGATE REWARDS -----------------
    ep_rewards.append(episode_reward)

    if not episode % SHOW_EVERY: # every SHOW_EVERY episodes
        average_reward = sum(ep_rewards[-SHOW_EVERY:])/len(ep_rewards[-SHOW_EVERY:]) # average of last SHOW_EVERY episodes
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['min'].append(min(ep_rewards[-SHOW_EVERY:]))
        aggr_ep_rewards['max'].append(max(ep_rewards[-SHOW_EVERY:]))

        print(f"Episode: {episode} avg: {average_reward} min: {min(ep_rewards[-SHOW_EVERY:])} max: {max(ep_rewards[-SHOW_EVERY:])}")

    # save the q_table every 1000 episodes
    if not episode % 1000:
        np.save(f"qtables/{episode}-qtable.npy", q_table)

# END FOR
env.close()

plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label="avg")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label="min")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label="max")
plt.legend(loc=4)
plt.show()


# ------------ 20 x 3 --------------
#  [[-0.35635789 -0.86828597 -0.84140828]
#   [-1.70799927 -0.0952704  -0.3920174 ]
#   [-0.93194453 -0.2431437  -0.80917254]
#   ...
#   [-1.53842557 -1.29764129 -0.34016108]
#   [-1.95512571 -1.24638773 -1.07880625]
#   [-1.90455684 -0.37332617 -0.8106289 ]]

#  [[-0.98238233 -1.45585304 -0.77284993]
#   [-1.71410326 -1.90618366 -1.9402937 ]
#   [-0.91877897 -0.56774635 -0.109718  ]
#   ...
#   [-1.51469126 -1.02063034 -1.36694996]
#   [-0.41118421 -1.84164896 -1.4180057 ]
#   [-0.9814291  -0.99179284 -1.49884963]]

#  [[-1.05273326 -0.63977697 -0.19686706]
#   [-0.39900045 -1.35425659 -1.70020886]
#   [-1.60680593 -1.86108679 -1.8799276 ]
#   ...
#   [-0.17824073 -0.21708192 -0.60378458]
#   [-0.35185099 -1.82056008 -0.80754891]
#   [-1.15538065 -1.01564085 -0.96344411]]
#--------------- x 20 ---------------
#...........

# done = False

# while not done:
#     action = 2 # 0, 1, 2
#     new_state, reward, done , _, info = env.step(action)
#     print(reward, new_state) # -1, [-0.57193965  0.00137146]
#     env.render() 

# env.close()

# 20 buckets of positions
# each bucket contains -> 20 buckets of velocity
# each bucket of velocity contains [q_value_action_0, q_value_action_1, q_value_action_2]

# 20 x 20 -> 400 combinations of q_values [action_0 , action_1, action_2]
# --------------- q_table ----------------
# position bucket #1 (of #20)
# velocity bucket #1 (of #20)
# row #1    | q_0 | q_1 | q2 |
# ----------------------------------------
# row #2    |     |     |    |
# ----------------------------------------
#           |     |     |    |
# ----------------------------------------
#           |     |     |    |
# ----------------------------------------
# row #20   |     |     |    |
# ----------------------------------------
