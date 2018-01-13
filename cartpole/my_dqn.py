import gym
import tensorflow as tf
import numpy as np
import random
from collections import deque

ENV_NAME = 'CartPole-v0'
EPISODE = 10000
STEP = 300

# Hyper Parameters for DQN
GAMMA = 0.9  # discount factor for target Q
INITIAL_EPSILON = 0.5  # starting value of epsilon
FINAL_EPSILON = 0.01  # final value of epsilon
REPLAY_SIZE = 10000  # experience replay buffer size
BATCH_SIZE = 32  # size of minibatch

HIDDEN_SIZE = 20


class DQN(object):
  def __init__(self, env):
    self.replay_buffer = deque()
    self.time_step = 0
    self.epsilon = INITIAL_EPSILON  # TODO 这个是什么？
    self.state_dim = env.observation_space.shape[0]
    self.action_dim = env.action_space.n

    self._create_Q_network()
    self._create_training_method()

    self.session = tf.InteractiveSession()
    self.session.run(tf.global_variables_initializer())

  def _create_Q_network(self):
    """
    根据输入的state产生当前state下的Qtable(qvalue)
    用途：np.argmax(q_value)就可得到当前环境下应该采取哪个action
    :return:
    """

    self.state_input = tf.placeholder(tf.float32, [None, self.state_dim])
    # 2 layers MLP
    w1 = tf.get_variable('w1', [self.state_dim, HIDDEN_SIZE])
    b1 = tf.get_variable('b1', [HIDDEN_SIZE])
    layer1 = tf.nn.relu(tf.matmul(self.state_input, w1) + b1)

    w2 = tf.get_variable('w2', [HIDDEN_SIZE, self.action_dim])
    b2 = tf.get_variable('b2', [self.action_dim])
    self.q_value = tf.matmul(layer1, w2) + b2

  def _create_training_method(self):
    self.action_input = tf.placeholder(tf.float32, [None, self.action_dim])
    self.y_input = tf.placeholder(tf.float32, [None])  # y_true
    q_action = tf.reduce_sum(tf.multiply(self.q_value, self.action_input), reduction_indices=1)
    # y_input 表示真实的未来回报，而q_action表示预测的未来汇报，目标就是让它们之间的差距减小
    self.loss = tf.reduce_mean(tf.square(self.y_input - q_action))
    self.opt = tf.train.AdamOptimizer(1e-5).minimize(self.loss)

  def egreedy_action(self, state):
    # 默认一次只处理一个batch的数据，所以q_value=xxx[0]
    q_value = self.q_value.eval(feed_dict={
      self.state_input: [state]
    })[0]
    self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000
    if random.random() <= self.epsilon:
      return random.randint(0, self.action_dim - 1)
    else:
      return np.argmax(q_value)

  def perceive(self, state, action, reward, next_state, done):
    one_not_action = tf.one_hot(action, depth=self.action_dim)
    self.replay_buffer.append((state, one_not_action, reward, next_state, done))
    if len(self.replay_buffer) > REPLAY_SIZE:
      self.replay_buffer.popleft()
    if len(self.replay_buffer) > BATCH_SIZE:
      self._train_q_network()

  def _train_q_network(self):
    self.time_step += 1
    mini_batch = random.sample(self.replay_buffer, BATCH_SIZE)
    state_batch = [data[0] for data in mini_batch]
    action_batch = [data[1] for data in mini_batch]
    reward_batch = [data[2] for data in mini_batch]
    next_state_batch = [data[3] for data in mini_batch]
    done_batch = [data[4] for data in mini_batch]

    y_batch = []
    q_value_batch = self.q_value.eval(feed_dict={
      self.state_input: next_state_batch  # TODO: 为什么是nextstate?
    })
    for i in range(BATCH_SIZE):
      done = done_batch[i]
      if done:
        y_batch.append(reward_batch[i])
      else:
        y_batch.append(reward_batch[i] + GAMMA * np.max(q_value_batch[i]))

    self.opt.run(feed_dict={
      self.y_input: y_batch,
      self.action_input: action_batch,
      self.state_input: state_batch
    })


def main():
  env = gym.make(ENV_NAME)
  agent = DQN(env)

  for episode in range(EPISODE):
    # initialize task
    state = env.reset()
    # Train
    for step in range(STEP):
      action = agent.egreedy_action(state) # e-greedy action for train
      next_state,reward,done,_ = env.step(action)
      # Define reward for agent
      reward_agent = -1 if done else 0.1
      agent.perceive(state,action,reward,next_state,done)
      state = next_state
      if done:
        break
    # Test every 100 episodes
    # if episode % 100 == 0:
    #   total_reward = 0
    #   for i in range(TEST):
    #     state = env.reset()
    #     for j in range(STEP):
    #       env.render()
    #       action = agent.action(state) # direct action for test
    #       state,reward,done,_ = env.step(action)
    #       total_reward += reward
    #       if done:
    #         break
    #   ave_reward = total_reward/TEST
    #   print ('episode: ',episode,'Evaluation Average Reward:',ave_reward)
    #   if ave_reward >= 200:
    #     break


if __name__ == '__main__':
  main()
