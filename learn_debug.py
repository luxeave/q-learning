import gym 
import numpy as np

render_mode = "none"

env = gym.make("MountainCar-v0")

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 25000

# SHOW_EVERY = 2000

epsilon = 0.5 # exploration rate
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2 # result always INT
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

# observation_space -> [position velocity]
print("upper boundary: ", env.observation_space.high) # [0.6  0.07] 
print("lower boundary: ", env.observation_space.low) # [-1.2  -0.07]
print(env.action_space.n) # 3 

OBSERVATION_SPACE_SIZE = len(env.observation_space.high) # 2 -> [position, velocity]

DISCRETE_OS_SIZE = [20] * OBSERVATION_SPACE_SIZE # 20 BUCKETS OF VALUES
print(DISCRETE_OS_SIZE) # [20, 20]

CONTINUOUS_OS_SIZE = env.observation_space.high - env.observation_space.low
print("CONTINUOUS_OS_SIZE: ", CONTINUOUS_OS_SIZE) # [1.8000001 0.14]

discrete_os_win_size = CONTINUOUS_OS_SIZE / DISCRETE_OS_SIZE
print("discrete_os_win_size: ", discrete_os_win_size) # [0.09  0.007]

TABLE_SIZE = DISCRETE_OS_SIZE + [env.action_space.n] # [20, 20] + [3]
print(TABLE_SIZE) # [20, 20, 3]

q_table = np.random.uniform(low=-2, high=0, size=TABLE_SIZE) # initialized with random values
print(q_table.shape) # [20, 20, 3]
print(q_table)
print("env.goal_position: ", env.goal_position)

render_once = False

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low)/discrete_os_win_size
    return tuple(discrete_state.astype(int))  # we use this tuple to look up the 3 Q values for the available actions in the q-table

for episode in range(EPISODES):    

    # it's good if we can visualize this via animating the act of retrieving q_values at coordinate `discrete_state`
    print(env.reset()) # (array([-0.5832684,  0.       ], dtype=float32), {})
    discrete_state = get_discrete_state(env.reset()[0])
    print("discrete_state: ",  discrete_state) #  (7, 10)

    print("q at initial_state: ", q_table[discrete_state]) # [-0.54471588 -1.7597477  -1.76844693]
    print("best q at initial_state: ", np.argmax(q_table[discrete_state])) # 0 - the index of the best action

    terminated = False 

    if render_once == False:
        frame_index = 0

    while not terminated:
        if np.random.random() > epsilon:
            # Get action from Q table
            action = np.argmax(q_table[discrete_state])
        else:
            # Get random action
            action = np.random.randint(0, env.action_space.n)

        new_state, reward, terminated, truncated, info = env.step(action)

        print("new_state: ", new_state)

        new_discrete_state = get_discrete_state(new_state) #  (9, 10)
        print("new_discrete_state: ", new_discrete_state)

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
        elif new_state[0] >= env.goal_position:
            print("new_state[0]: ", new_state[0])
            print(f"we made it on episode {episode}")
            q_table[discrete_state + (action,)] = 0 # set state + action -> q_values to zero -> we win the game
            observation, info = env.reset()

        discrete_state = new_discrete_state

        # END WHILE
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value




env.close()


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
