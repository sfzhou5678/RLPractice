import tensorflow as tf
import numpy as np
import time
import os


class DQNet:
  def __init__(self):
    self


def get_q_table(maze_length, action_list_length):
  q_table = np.zeros(shape=[maze_length, action_list_length])

  return q_table


def choose_action(cur_state, q_table, EPSILON):
  if (np.random.uniform() < EPSILON) or (np.sum(np.abs(q_table[cur_state])) == 0):
    action_index = np.random.randint(0, len(q_table[cur_state]))
  else:
    action_index = np.argmax(q_table[cur_state])
  return action_index


def get_env_feed(cur_state, action_index, holes, target):
  global MAZE_WIDTH
  global MAZE_HEIGHT

  x = cur_state[1]
  y = cur_state[0]
  if action_index == 0:
    x -= 1
    x = max(0, x)
  elif action_index == 1:
    x += 1
    x = min(x, MAZE_WIDTH)
  elif action_index == 2:
    y -= 1
    y = max(0, y)
  else:
    y += 1
    y = min(y, MAZE_HEIGHT)
  next_state = (y, x)

  if next_state in holes:
    reward = -5
  elif next_state == target:
    reward = 1
  else:
    reward = 0
  return next_state, reward


def update_env(cur_state, MAZE_WIDTH, MAZE_HEIGHT, holes, target,
               refresh_time):
  maze = np.zeros(shape=[MAZE_HEIGHT, MAZE_WIDTH], dtype=np.int)
  for hole in holes:
    row = hole[0]
    col = hole[1]
    maze[row, col] = -1
  maze[target[0], target[1]] = 1
  maze[cur_state[0], cur_state[1]] = 10

  def map_item(item):
    if item == 1:
      return 'T'
    if item == -1:
      return 'X'
    if item == 10:
      return 'S'
    else:
      return item

  t = os.system('cls')
  interaction = '\n'.join('  '.join('%s' % map_item(item) for item in row) for row in maze)
  print('\r{}'.format(interaction), end='')
  # print('\r')
  # for row in maze:
  #   print('  '.join('%s' % map_item(item) for item in row))
  time.sleep(refresh_time)


def main():
  global MAZE_WIDTH
  global MAZE_HEIGHT

  MAZE_WIDTH = 6
  MAZE_HEIGHT = 6
  MAZE_HELL_NUM = 2

  ACTION_LIST = ['LEFT', 'RIGHT', 'UP', 'DOWN']
  EPSILON = 0.4  # greedy police
  LEARNING_RATE = 0.1  # learning rate
  GAMMA = 0.9  # discount factor
  MAX_EPISODES = 100  # maximum episodes
  REFRESH_TIME = 1  # fresh time for one move

  # q_table = get_q_table(MAZE_LENGTH, len(ACTION_LIST))
  q_table = {}

  holes = []
  while len(holes) < MAZE_HELL_NUM:
    x = np.random.randint(0, MAZE_WIDTH)
    y = np.random.randint(0, MAZE_HEIGHT)
    if x == 0 and y == 0:
      continue
    if (y, x) in holes:
      continue
    holes.append((y, x))
  # print(holes)

  while True:
    x = np.random.randint(0, MAZE_WIDTH)
    y = np.random.randint(0, MAZE_HEIGHT)

    if x == 0 and y == 0:
      continue
    if (y, x) in holes:
      continue
    target = (y, x)
    break
  # print(target)

  for episode in range(MAX_EPISODES):
    step_counter = 0

    cur_state = (0, 0)
    is_terminated = False

    # update_env(cur_state, MAZE_WIDTH, MAZE_HEIGHT, holes, target, REFRESH_TIME)
    while not is_terminated:
      if cur_state not in q_table:
        q_table[cur_state] = np.zeros([len(ACTION_LIST)])

      action_index = choose_action(cur_state, q_table, EPSILON / np.exp(episode / 50))
      next_state, reward = get_env_feed(cur_state, action_index, holes, target)
      if next_state not in q_table:
        q_table[next_state] = np.zeros([len(ACTION_LIST)])

      # Bellman Equation: Q[st,at]+=α(reward +γ*max(Q[st+1,at+1])-Q[st,at])
      q_target = reward + GAMMA * np.max(q_table[next_state])
      q_predict = q_table[cur_state][action_index]
      q_table[cur_state][action_index] += LEARNING_RATE * (q_target - q_predict)

      cur_state = next_state
      step_counter += 1

      # update_env(cur_state, MAZE_WIDTH, MAZE_HEIGHT, holes, target, REFRESH_TIME)

      if next_state == target:
        is_terminated = True

      if next_state in holes:
        is_terminated = True

    print(episode, step_counter)
    for key in q_table:
      print(str(key) + '\t' + '\t'.join('%.2f' % item for item in q_table[key]))
      # print(q_table)


if __name__ == '__main__':
  np.random.seed(3)
  main()
