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


class DQN(object):
  def __init__(self, env):
    self.replay_buffer = deque()

    self.time_step = 0
    self.epsilon = INITIAL_EPSILON
    self.state_dim = env.observation_space.shape[0]
    self.action_dim = env.action_space.n

    self._create_network()
    self._create_training_method()

    self.sess = tf.InteractiveSession()
    self.sess.run(tf.global_variables_initializer())

  def _create_network(self):
    w1 = tf.get_variable('w1', shape=[self.state_dim, 20])  # 20表示MLP的隐藏层单元数
    b1 = tf.get_variable('b1', [20])

    w2 = tf.get_variable('w2',[20, self.action_dim])
    b2 = tf.get_variable('b2',[self.action_dim])

    self.state_input = tf.placeholder(tf.float32, [None, self.state_dim])
    net = tf.nn.relu(tf.matmul(self.state_input, w1) + b1)
    self.q_value = tf.matmul(net, w2) + b2

  def _create_training_method(self):
    # FIXME: action 是一种01吗？
    self.action_input = tf.placeholder(tf.float32, [None, self.action_dim])  # one hot presentation
    self.y_input = tf.placeholder(tf.float32, [None])  # TODO? y???

    q_action = tf.reduce_sum(tf.matmul(self.q_value, self.action_input))
    self.loss = tf.reduce_mean(tf.square(self.y_input - q_action))

    self.opt = tf.train.AdadeltaOptimizer(1e-5).minimize(self.loss)

  def perceive(self, state, action, reward, next_state, done):
    one_hot_action = np.zeros(self.action_dim)
    one_hot_action[action] = 1
    self.replay_buffer.append((state, one_hot_action, reward, next_state, done))
    if len(self.replay_buffer) > REPLAY_SIZE:
      self.replay_buffer.popleft()

    if len(self.replay_buffer) > BATCH_SIZE:
      self._train_net()

  def _train_net(self):
    self.time_step += 1
    # step 1: obtain random minibatch from replay memory
    minibatch = random.sample(self.replay_buffer, BATCH_SIZE)
    state_batch = [data[0] for data in minibatch]
    action_batch = [data[1] for data in minibatch]
    reward_batch = [data[2] for data in minibatch]
    next_state_batch = [data[3] for data in minibatch]
    done_batch = [data[4] for data in minibatch]

    # step 2: calculate y
    y_batch = []
    q_value_batch = self.q_value.eval(feed_dict={self.state_input: next_state_batch})  # TODO: 为什么输入的是nextState？
    for i in range(BATCH_SIZE):
      done = done_batch[i]
      if done:
        y_batch.append(reward_batch[i])
      else:
        y_batch.append(reward_batch[i] + GAMMA * np.max(q_value_batch[i]))

    self.opt.run(feed_dict={
      self.y_input: y_batch, self.action_input: action_batch, self.state_input: state_batch})

  def egreedy_action(self, state):
    # for training
    q_value = self.q_value.eval(feed_dict={self.state_input: [state]})[0]  # FIXME: 为什么state要加[]? 为什么要加0？
    if random.random() < self.epsilon:
      return random.randint(0, self.action_dim - 1)
    else:
      return np.argmax(q_value)

  def action(self, state):
    # for testing
    return np.argmax(self.q_value.eval(feed_dict={
      self.state_input: [state]
    })[0])


def main():
  env = gym.make(ENV_NAME)
  agent = DQN(env)

  for episode in range(EPISODE):
    state = env.reset()
    for step in range(STEP):
      action = agent.egreedy_action(state)
      next_state, reward, done, _ = env.step(action)

      reward_agent = -1 if done else 0.1  # TODO: ???
      agent.perceive(state, action, reward, next_state, done)  # TODO ??
      state = next_state
      if done:
        break

if __name__ == '__main__':
    main()