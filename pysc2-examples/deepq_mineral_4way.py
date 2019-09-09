import numpy as np
import os
import dill
import tempfile
import tensorflow as tf
import zipfile

from absl import flags

import baselines.common.tf_util as U

from baselines import logger
from baselines.common.schedules import LinearSchedule
from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer

from pysc2.lib import actions as sc2_actions
from pysc2.env import environment
from pysc2.lib import features
from pysc2.lib import actions

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_FRIENDLY = 1
_PLAYER_NEUTRAL = 3  # beacon/minerals
_PLAYER_HOSTILE = 4
_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_NOT_QUEUED = [0]
_SELECT_ALL = [0]

FLAGS = flags.FLAGS


class ActWrapper(object):
  def __init__(self, act):
    self._act = act
    #self._act_params = act_params

  @staticmethod
  def load(path, act_params, num_cpu=16):
#    with open(path, "rb") as f:
#      model_data = dill.load(f)
    act = deepq.build_act(**act_params)
    sess = U.make_session(num_cpu=num_cpu)
    sess.__enter__()
    U.load_state(path)
#    with tempfile.TemporaryDirectory() as td:
#      arc_path = os.path.join(td, "packed.zip")
#      with open(arc_path, "wb") as f:
#        f.write(model_data)
#
#      zipfile.ZipFile(arc_path, 'r', zipfile.ZIP_DEFLATED).extractall(td)
#      U.load_state(os.path.join(td, "model"))

    return ActWrapper(act)

  def __call__(self, *args, **kwargs):
    return self._act(*args, **kwargs)

  def save(self, path):
    """Save model to a pickle located at `path`"""
    with tempfile.TemporaryDirectory() as td:
      U.save_state(os.path.join(td, "model"))
      arc_name = os.path.join(td, "packed.zip")
      with zipfile.ZipFile(arc_name, 'w') as zipf:
        for root, dirs, files in os.walk(td):
          for fname in files:
            file_path = os.path.join(root, fname)
            if file_path != arc_name:
              zipf.write(file_path,
                         os.path.relpath(file_path, td))
      with open(arc_name, "rb") as f:
        model_data = f.read()
    with open(path, "wb") as f:
      dill.dump((model_data), f)


def load(path, act_params, num_cpu=16):
  """学习函数返回的加载行为函数.

Parameters
----------
path: str
    到act函数pickle的路径
num_cpu: int
    用于执行策略的cpu数量

Returns
-------
act: ActWrapper
    函数，该函数接受一批观察值
    并返回操作.
"""
  return ActWrapper.load(path, num_cpu=num_cpu, act_params=act_params)


def learn(env,
          q_func,
          num_actions=4,
          lr=5e-4,
          max_timesteps=100000,
          buffer_size=50000,
          exploration_fraction=0.1,
          exploration_final_eps=0.02,
          train_freq=1,
          batch_size=32,
          print_freq=1,
          checkpoint_freq=10000,
          learning_starts=1000,
          gamma=1.0,
          target_network_update_freq=500,
          prioritized_replay=False,
          prioritized_replay_alpha=0.6,
          prioritized_replay_beta0=0.4,
          prioritized_replay_beta_iters=None,
          prioritized_replay_eps=1e-6,
          num_cpu=16,
          param_noise=False,
          param_noise_threshold=0.05,
          callback=None):
  """训练一个DQN模型.

Parameters
-------
env: pysc2.env.SC2Env
    训练环境
q_func: (tf.Variable, int, str, bool) -> tf.Variable
    接受以下输入的模型:
        observation_in: object
            观察占位符的输出
        num_actions: int
            数量的行为
        scope: str
        reuse: bool
            是否应该传递到外部变量范围
    并返回一个形状张量(batch_size, num_actions)，其中包含每个动作的值.
lr: float
    亚当优化学习率
max_timesteps: int
    要优化的env步骤的数量
buffer_size: int
    重放缓冲区的大小
exploration_fraction: float
    勘探率退火后的整个训练周期的一部分
exploration_final_eps: float
    随机作用概率的最终值
train_freq: int
    更新模型的每一个' train_freq '步骤。
    设置为None以禁用打印
batch_size: int
    从回放缓冲区中取样用于训练的批量大小
print_freq: int
    多久打印一次培训进度
    设置为None以禁用打印
checkpoint_freq: int
    保存模型的频率。这是为了恢复最好的版本
    在训练结束时。如果不希望还原最佳版本，请在
    训练结束时，将该变量设置为None。
learning_starts: int
    在开始学习之前，模型中有多少步骤需要收集转换
gamma: float
    折现系数
target_network_update_freq: int
    更新目标网络的每个“target_network_update_freq”步骤。
prioritized_replay: True
    如果真，优先重放缓冲区将被使用。
prioritized_replay_alpha: float
    优先级重放缓冲区的alpha参数
prioritized_replay_beta0: float
    优先级重放缓冲区的beta初始值
prioritized_replay_beta_iters: int
    从初始值开始对beta进行退火的迭代次数
    到1.0。如果设置为None，则等于max_timesteps。
prioritized_replay_eps: float
    在更新优先级时增加TD错误。
num_cpu: int
    用于培训的cpu数量
callback: (locals, globals) -> None
    函数调用的每一步与算法的状态。
    如果回调返回真训练停止。

Returns
-------
act: ActWrapper
    封装行为函数。添加保存和加载它的功能。
    有关act函数的详细信息，请参见基线/deepq/ category .py的头部。
"""
  # 创建训练模型所需的所有函数

  sess = U.make_session(num_cpu=num_cpu)
  sess.__enter__()

  def make_obs_ph(name):
    return U.BatchInput((32, 32), name=name)

  act, train, update_target, debug = deepq.build_train(
    make_obs_ph=make_obs_ph,
    q_func=q_func,
    num_actions=num_actions,
    optimizer=tf.train.AdamOptimizer(learning_rate=lr),
    gamma=gamma,
    grad_norm_clipping=10,
    scope="deepq")
  #
  # act_y, train_y, update_target_y, debug_y = deepq.build_train(
  #   make_obs_ph=make_obs_ph,
  #   q_func=q_func,
  #   num_actions=num_actions,
  #   optimizer=tf.train.AdamOptimizer(learning_rate=lr),
  #   gamma=gamma,
  #   grad_norm_clipping=10,
  #   scope="deepq_y"
  # )

  act_params = {
    'make_obs_ph': make_obs_ph,
    'q_func': q_func,
    'num_actions': num_actions,
  }

  # Create the replay buffer
  if prioritized_replay:
    replay_buffer = PrioritizedReplayBuffer(
      buffer_size, alpha=prioritized_replay_alpha)
    # replay_buffer_y = PrioritizedReplayBuffer(buffer_size, alpha=prioritized_replay_alpha)

    if prioritized_replay_beta_iters is None:
      prioritized_replay_beta_iters = max_timesteps
    beta_schedule = LinearSchedule(
      prioritized_replay_beta_iters,
      initial_p=prioritized_replay_beta0,
      final_p=1.0)

    # beta_schedule_y = LinearSchedule(prioritized_replay_beta_iters,
    #                                  initial_p=prioritized_replay_beta0,
    #                                  final_p=1.0)
  else:
    replay_buffer = ReplayBuffer(buffer_size)
    # replay_buffer_y = ReplayBuffer(buffer_size)

    beta_schedule = None
    # beta_schedule_y = None
  # 从1开始创建探索计划。
  exploration = LinearSchedule(
    schedule_timesteps=int(exploration_fraction * max_timesteps),
    initial_p=1.0,
    final_p=exploration_final_eps)

  # 初始化参数并将它们复制到目标网络。
  U.initialize()
  update_target()
  # update_target_y()

  episode_rewards = [0.0]
  saved_mean_reward = None

  obs = env.reset()
  # 首先选择所有矿兵
  obs = env.step(
    actions=[sc2_actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])])

  player_relative = obs[0].observation["screen"][_PLAYER_RELATIVE]

  screen = (player_relative == _PLAYER_NEUTRAL).astype(int)  #+ path_memory

  player_y, player_x = (player_relative == _PLAYER_FRIENDLY).nonzero()
  player = [int(player_x.mean()), int(player_y.mean())]

  if (player[0] > 16):
    screen = shift(LEFT, player[0] - 16, screen)
  elif (player[0] < 16):
    screen = shift(RIGHT, 16 - player[0], screen)

  if (player[1] > 16):
    screen = shift(UP, player[1] - 16, screen)
  elif (player[1] < 16):
    screen = shift(DOWN, 16 - player[1], screen)

  reset = True
  with tempfile.TemporaryDirectory() as td:
    model_saved = False
    model_file = os.path.join("model/", "mineral_shards")
    print(model_file)

    for t in range(max_timesteps):
      if callback is not None:
        if callback(locals(), globals()):
          break
      # 采取行动，将探索更新到最新的价值
      kwargs = {}
      if not param_noise:
        update_eps = exploration.value(t)
        update_param_noise_threshold = 0.
      else:
        update_eps = 0.
        if param_noise_threshold >= 0.:
          update_param_noise_threshold = param_noise_threshold
        else:
            # 计算阈值，使微扰和非微扰之间的KL发散
            # policy与eps = explorer .value(t)的eps贪婪探索类似。
            #参见附录C.1的参数空间噪声勘探，Plappert等，2017
            #详细说明。
          update_param_noise_threshold = -np.log(
            1. - exploration.value(t) +
            exploration.value(t) / float(num_actions))
        kwargs['reset'] = reset
        kwargs[
          'update_param_noise_threshold'] = update_param_noise_threshold
        kwargs['update_param_noise_scale'] = True

      action = act(
        np.array(screen)[None], update_eps=update_eps, **kwargs)[0]

      # action_y = act_y(np.array(screen)[None], update_eps=update_eps, **kwargs)[0]

      reset = False

      coord = [player[0], player[1]]
      rew = 0

      if (action == 0):  #UP

        if (player[1] >= 8):
          coord = [player[0], player[1] - 8]
          #path_memory_[player[1] - 16 : player[1], player[0]] = -1
        elif (player[1] > 0):
          coord = [player[0], 0]
          #path_memory_[0 : player[1], player[0]] = -1
          #else:
          #  rew -= 1

      elif (action == 1):  #DOWN

        if (player[1] <= 23):
          coord = [player[0], player[1] + 8]
          #path_memory_[player[1] : player[1] + 16, player[0]] = -1
        elif (player[1] > 23):
          coord = [player[0], 31]
          #path_memory_[player[1] : 63, player[0]] = -1
          #else:
          #  rew -= 1

      elif (action == 2):  #LEFT

        if (player[0] >= 8):
          coord = [player[0] - 8, player[1]]
          #path_memory_[player[1], player[0] - 16 : player[0]] = -1
        elif (player[0] < 8):
          coord = [0, player[1]]
          #path_memory_[player[1], 0 : player[0]] = -1
          #else:
          #  rew -= 1

      elif (action == 3):  #RIGHT

        if (player[0] <= 23):
          coord = [player[0] + 8, player[1]]
          #path_memory_[player[1], player[0] : player[0] + 16] = -1
        elif (player[0] > 23):
          coord = [31, player[1]]
          #path_memory_[player[1], player[0] : 63] = -1

      if _MOVE_SCREEN not in obs[0].observation["available_actions"]:
        obs = env.step(actions=[
          sc2_actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])
        ])

      new_action = [
        sc2_actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, coord])
      ]

      # else:
      #   new_action = [sc2_actions.FunctionCall(_NO_OP, [])]

      obs = env.step(actions=new_action)

      player_relative = obs[0].observation["screen"][_PLAYER_RELATIVE]
      new_screen = (player_relative == _PLAYER_NEUTRAL).astype(
        int)  #+ path_memory

      player_y, player_x = (
        player_relative == _PLAYER_FRIENDLY).nonzero()
      player = [int(player_x.mean()), int(player_y.mean())]

      if (player[0] > 16):
        new_screen = shift(LEFT, player[0] - 16, new_screen)
      elif (player[0] < 16):
        new_screen = shift(RIGHT, 16 - player[0], new_screen)

      if (player[1] > 16):
        new_screen = shift(UP, player[1] - 16, new_screen)
      elif (player[1] < 16):
        new_screen = shift(DOWN, 16 - player[1], new_screen)

      rew = obs[0].reward

      done = obs[0].step_type == environment.StepType.LAST

      # Store transition in the replay buffer.
      replay_buffer.add(screen, action, rew, new_screen, float(done))
      # replay_buffer_y.add(screen, action_y, rew, new_screen, float(done))

      screen = new_screen

      episode_rewards[-1] += rew
      reward = episode_rewards[-1]

      if done:
        obs = env.reset()
        player_relative = obs[0].observation["screen"][
          _PLAYER_RELATIVE]

        screen = (player_relative == _PLAYER_NEUTRAL).astype(
          int)  #+ path_memory

        player_y, player_x = (
          player_relative == _PLAYER_FRIENDLY).nonzero()
        player = [int(player_x.mean()), int(player_y.mean())]

        # Select all marines first
        env.step(actions=[
          sc2_actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])
        ])
        episode_rewards.append(0.0)
        #episode_minerals.append(0.0)

        reset = True

      if t > learning_starts and t % train_freq == 0:
        # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
        if prioritized_replay:

          experience = replay_buffer.sample(
            batch_size, beta=beta_schedule.value(t))
          (obses_t, actions, rewards, obses_tp1, dones, weights,
           batch_idxes) = experience

          # experience_y = replay_buffer.sample(batch_size, beta=beta_schedule.value(t))
          # (obses_t_y, actions_y, rewards_y, obses_tp1_y, dones_y, weights_y, batch_idxes_y) = experience_y
        else:

          obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(
            batch_size)
          weights, batch_idxes = np.ones_like(rewards), None

          # obses_t_y, actions_y, rewards_y, obses_tp1_y, dones_y = replay_buffer_y.sample(batch_size)
          # weights_y, batch_idxes_y = np.ones_like(rewards_y), None

        td_errors = train(obses_t, actions, rewards, obses_tp1, dones,
                          weights)

        # td_errors_y = train_x(obses_t_y, actions_y, rewards_y, obses_tp1_y, dones_y, weights_y)

        if prioritized_replay:
          new_priorities = np.abs(td_errors) + prioritized_replay_eps
          # new_priorities = np.abs(td_errors) + prioritized_replay_eps
          replay_buffer.update_priorities(batch_idxes,
                                          new_priorities)
          # replay_buffer.update_priorities(batch_idxes, new_priorities)

      if t > learning_starts and t % target_network_update_freq == 0:
        # 定期更新目标网络。
        update_target()
        # update_target_y()

      mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
      num_episodes = len(episode_rewards)
      if done and print_freq is not None and len(
          episode_rewards) % print_freq == 0:
        logger.record_tabular("steps", t)
        logger.record_tabular("episodes", num_episodes)
        logger.record_tabular("reward", reward)
        logger.record_tabular("mean 100 episode reward",
                              mean_100ep_reward)
        logger.record_tabular("% time spent exploring",
                              int(100 * exploration.value(t)))
        logger.dump_tabular()

      if (checkpoint_freq is not None and t > learning_starts
          and num_episodes > 100 and t % checkpoint_freq == 0):
        if saved_mean_reward is None or mean_100ep_reward > saved_mean_reward:
          if print_freq is not None:
            logger.log(
              "Saving model due to mean reward increase: {} -> {}".
                format(saved_mean_reward, mean_100ep_reward))
          U.save_state(model_file)
          model_saved = True
          saved_mean_reward = mean_100ep_reward
    if model_saved:
      if print_freq is not None:
        logger.log("Restored model with mean reward: {}".format(
          saved_mean_reward))
      U.load_state(model_file)

  return ActWrapper(act)


def intToCoordinate(num, size=64):
  if size != 64:
    num = num * size * size // 4096
  y = num // size
  x = num - size * y
  return [x, y]


UP, DOWN, LEFT, RIGHT = 'up', 'down', 'left', 'right'


def shift(direction, number, matrix):
  ''' 将给定的二维矩阵移位到指定的行数或列数
      按照指定的(上、下、左、右)方向返回
'''
  if direction in (UP):
    matrix = np.roll(matrix, -number, axis=0)
    matrix[number:, :] = 0
    return matrix
  elif direction in (DOWN):
    matrix = np.roll(matrix, number, axis=0)
    matrix[:number, :] = 0
    return matrix
  elif direction in (LEFT):
    matrix = np.roll(matrix, -number, axis=1)
    matrix[:, number:] = 0
    return matrix
  elif direction in (RIGHT):
    matrix = np.roll(matrix, number, axis=1)
    matrix[:, :number] = 0
    return matrix
  else:
    return matrix
