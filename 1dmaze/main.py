import numpy as np
import time


def get_q_table(maze_length, action_list_length):
  q_table = np.zeros(shape=[maze_length, action_list_length])

  return q_table


def get_reward_table(maze_len, act_list_len):
  reward_table = np.zeros(shape=[maze_len, act_list_len])
  reward_table[maze_len - 2, 1] = 1

  return reward_table


def choose_action(cur_state, q_table, EPSILON):
  if (np.random.uniform() > EPSILON) or (np.sum(np.abs(q_table[cur_state])) == 0):
    action_index = np.random.randint(0, len(q_table[0]))
  else:
    action_index = np.argmax(q_table[cur_state])
  return action_index


def get_env_feed(reward_table, cur_state, action_index):
  reward = reward_table[cur_state, action_index]
  if action_index == 0:
    next_state = cur_state - 1
    next_state = max(0, next_state)
  else:
    next_state = cur_state + 1

  return next_state, reward


def update_env(cur_state, maze_len, regresh_time):
  # This is how environment be updated
  env_list = ['-'] * (maze_len - 1) + ['T']  # '---------T' our environment

  env_list[cur_state] = 'o'
  interaction = ''.join(env_list)
  print('\r{}'.format(interaction), end='')
  time.sleep(regresh_time)


def main():
  MAZE_LENGTH = 6
  ACTION_LIST = ['LEFT', 'RIGHT']
  EPSILON = 0.9  # greedy police
  LEARNING_RATE = 0.1  # learning rate
  GAMMA = 0.9  # discount factor
  MAX_EPISODES = 13  # maximum episodes
  REFRESH_TIME = 0.2  # fresh time for one move

  q_table = get_q_table(MAZE_LENGTH, len(ACTION_LIST))
  reward_table = get_reward_table(MAZE_LENGTH, len(ACTION_LIST))
  # print(q_table)
  # print(reward_table)

  for episode in range(MAX_EPISODES):
    step_counter = 0

    cur_state = 0
    is_terminated = False

    # update_env(cur_state, MAZE_LENGTH, REFRESH_TIME)
    while not is_terminated:
      action_index = choose_action(cur_state, q_table, EPSILON)
      next_state, reward = get_env_feed(reward_table, cur_state, action_index)

      # Bellman Equation: Q[st,at]+=α(reward +γ*max(Q[st+1,at+1])-Q[st,at])
      q_target = reward + GAMMA * np.max(q_table[next_state])
      q_predict = q_table[cur_state, action_index]
      q_table[cur_state, action_index] += LEARNING_RATE * (q_target - q_predict)

      cur_state = next_state
      step_counter += 1

      # update_env(cur_state, MAZE_LENGTH, REFRESH_TIME)
      if next_state == MAZE_LENGTH - 1:
        is_terminated = True

    print(episode, step_counter)
    print(q_table)


if __name__ == '__main__':
  np.random.seed(2)
  main()
