# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""一个星际争霸2的环境。"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from absl import logging
import time

import enum

from pysc2 import maps
from pysc2 import run_configs
from pysc2.env import environment
from pysc2.lib import actions as actions_lib
from pysc2.lib import features
from pysc2.lib import metrics
from pysc2.lib import portspicker
from pysc2.lib import protocol
from pysc2.lib import renderer_human
from pysc2.lib import run_parallel
from pysc2.lib import stopwatch

from s2clientprotocol import common_pb2 as sc_common
from s2clientprotocol import sc2api_pb2 as sc_pb

sw = stopwatch.sw


possible_results = {
    sc_pb.Victory: 1,
    sc_pb.Defeat: -1,
    sc_pb.Tie: 0,
    sc_pb.Undecided: 0,
}


class Race(enum.IntEnum):
  random = sc_common.Random
  protoss = sc_common.Protoss
  terran = sc_common.Terran
  zerg = sc_common.Zerg


class Difficulty(enum.IntEnum):
  """机器人的困难。"""
  very_easy = sc_pb.VeryEasy
  easy = sc_pb.Easy
  medium = sc_pb.Medium
  medium_hard = sc_pb.MediumHard
  hard = sc_pb.Hard
  harder = sc_pb.Harder
  very_hard = sc_pb.VeryHard
  cheat_vision = sc_pb.CheatVision
  cheat_money = sc_pb.CheatMoney
  cheat_insane = sc_pb.CheatInsane

# Re-export these names to make it easy to construct the environment.
ActionSpace = actions_lib.ActionSpace  # pylint: disable=invalid-name
Dimensions = features.Dimensions  # pylint: disable=invalid-name
AgentInterfaceFormat = features.AgentInterfaceFormat  # pylint: disable=invalid-name
parse_agent_interface_format = features.parse_agent_interface_format


class Agent(collections.namedtuple("Agent", ["race", "name"])):

  def __new__(cls, race, name=None):
    return super(Agent, cls).__new__(cls, race, name or "<unknown>")


Bot = collections.namedtuple("Bot", ["race", "difficulty"])


REALTIME_GAME_LOOP_SECONDS = 1 / 22.4
EPSILON = 1e-5


class SC2Env(environment.Base):
  """一个星际争霸2的环境。

  操作和观察规范的实现细节在其中
  lib / features.py
  """

  def __init__(self,  # pylint: disable=invalid-name
               _only_use_kwargs=None,
               map_name=None,
               players=None,
               agent_race=None,  # deprecated
               bot_race=None,  # deprecated
               difficulty=None,  # deprecated
               screen_size_px=None,  # deprecated
               minimap_size_px=None,  # deprecated
               agent_interface_format=None,
               discount=1.,
               discount_zero_after_timeout=False,
               visualize=False,
               step_mul=None,
               realtime=False,
               save_replay_episodes=0,
               replay_dir=None,
               replay_prefix=None,
               game_steps_per_episode=None,
               score_index=None,
               score_multiplier=None,
               random_seed=None,
               disable_fog=False,
               ensure_available_actions=True):
    """创建一个SC2环境。
    你必须通过一个你想玩的决议。你可以发送
    特征层分辨率或rgb分辨率或两者兼而有之。如果你们两个都送的话
    还必须选择使用哪个作为您的操作空间。不管你是谁
    选择您必须同时发送屏幕和小地图分辨率。
    对于这4种分辨率，要么指定大小，要么同时指定宽度和
    高度。如果指定了size，那么width和height都将接受该值。

    Args:
      _only_use_kwargs: 不要传递args，只传递kwargs。
      map_name: SC2映射的名称。运行bin/map_list以获得完整的列表
        已知的地图。或者，传递一个Map实例。来看看
        在地图/ README文档。更多关于可用地图的信息，md。
      players: 指定玩家的智能体和机器人实例列表。
      agent_race: 弃用。使用玩家代替。
      bot_race: 弃用。使用玩家代替。
      difficulty: 弃用。使用玩家代替。
      screen_size_px: 弃用。使用agent_interface_formats代替。
      minimap_size_px: 弃用。使用agent_interface_formats代替。
      agent_interface_format: 包含一个代理接口格式的序列
      每个代理，匹配球员列表中指定的代理的顺序。
      或用于所有代理的单个AgentInterfaceFormat。
      discount: 作为观察的一部分返回。
      discount_zero_after_timeout: 如果返回为真，则折扣为零
      在“game_steps_per_episode”超时之后。观察的一部分。
      visualize:他必须弹出一个显示相机和功能的窗口
      层。如果没有访问窗口管理器，这将无法工作。
      step_mul: 每个智能体步骤(动作/观察)有多少个游戏步骤。没有一个
      表示使用地图默认值。
      realtime: 每个代理步骤(动作/观察)有多少个游戏步骤。是否使用实时模式。在此模式下进行游戏模拟
        自动前进(以每秒22.4个游戏负载)而不是
        被手动了。每个游戏循环的数量都在增加
        调用step()不一定匹配指定的step_mul。的
        环境将尝试尊重step_mul，返回观测值
        用尽可能近的间隔。游戏循环将被跳过
        如果它们不能足够快地检索和处理
        表示使用地图默认值。
      save_replay_episodes: 保存这许多集之后的重播。默认为0
      意思是不要保存回放。
      replay_dir: 保存重播的目录。与save_replay_episodes所需。
      replay_prefix: 保存重播时使用的可选前缀。
      game_steps_per_episode: 游戏步骤每集，独立的
      step_mul。0表示没有极限。None表示使用map默认值。
      score_index: -1表示使用赢/输奖励，>=0是指数进入
      score_cumulative以0为课程分数。没有意味着使用
      地图上违约。
      score_multiplier: 分数乘以多少。用于否定。
      random_seed: 初始化游戏时使用的随机数种子。这
      让您运行可重复的游戏/测试。
      disable_fog: 是否禁用战争迷雾。
      ensure_available_actions:时是否抛出异常
      不可用的操作被传递给step()。

    Raises:
      ValueError: 如果agent_race、bot_race或难度无效。
      ValueError: 如果太多玩家需要地图。
      ValueError: 如果太多，如果决议没有正确指定。玩家需要一张地图。
      DeprecationWarning: 如果发送了screen_size_px或minimap_size_px。
      DeprecationWarning: 如果发送了agent_race、bot_race或难度。
    """
    if _only_use_kwargs:
      raise ValueError("All arguments must be passed as keyword arguments.")

    if screen_size_px or minimap_size_px:
      raise DeprecationWarning(
          "screen_size_px and minimap_size_px are deprecated. Use the feature "
          "or rgb variants instead. Make sure to check your observations too "
          "since they also switched from screen/minimap to feature and rgb "
          "variants.")

    if agent_race or bot_race or difficulty:
      raise DeprecationWarning(
          "Explicit agent and bot races are deprecated. Pass an array of "
          "sc2_env.Bot and sc2_env.Agent instances instead.")

    map_inst = maps.get(map_name)
    self._map_name = map_name

    if not players:
      players = list()
      players.append(Agent(Race.random))

      if not map_inst.players or map_inst.players >= 2:
        players.append(Bot(Race.random, Difficulty.very_easy))

    for p in players:
      if not isinstance(p, (Agent, Bot)):
        raise ValueError(
            "Expected players to be of type Agent or Bot. Got: %s." % p)

    num_players = len(players)
    self._num_agents = sum(1 for p in players if isinstance(p, Agent))
    self._players = players

    if not 1 <= num_players <= 2 or not self._num_agents:
      raise ValueError(
          "Only 1 or 2 players with at least one agent is "
          "supported at the moment.")

    if save_replay_episodes and not replay_dir:
      raise ValueError("Missing replay_dir")

    if map_inst.players and num_players > map_inst.players:
      raise ValueError(
          "Map only supports %s players, but trying to join with %s" % (
              map_inst.players, num_players))

    self._discount = discount
    self._step_mul = step_mul or map_inst.step_mul
    self._realtime = realtime
    self._last_step_time = None
    self._save_replay_episodes = save_replay_episodes
    self._replay_dir = replay_dir
    self._replay_prefix = replay_prefix
    self._random_seed = random_seed
    self._disable_fog = disable_fog
    self._ensure_available_actions = ensure_available_actions
    self._discount_zero_after_timeout = discount_zero_after_timeout

    if score_index is None:
      self._score_index = map_inst.score_index
    else:
      self._score_index = score_index
    if score_multiplier is None:
      self._score_multiplier = map_inst.score_multiplier
    else:
      self._score_multiplier = score_multiplier

    self._episode_length = game_steps_per_episode
    if self._episode_length is None:
      self._episode_length = map_inst.game_steps_per_episode

    self._run_config = run_configs.get()
    self._parallel = run_parallel.RunParallel()  # Needed for multiplayer.

    if agent_interface_format is None:
      raise ValueError("Please specify agent_interface_format.")

    if isinstance(agent_interface_format, AgentInterfaceFormat):
      agent_interface_format = [agent_interface_format] * self._num_agents

    if len(agent_interface_format) != self._num_agents:
      raise ValueError(
          "The number of entries in agent_interface_format should "
          "correspond 1-1 with the number of agents.")

    interfaces = []
    for i, interface_format in enumerate(agent_interface_format):
      require_raw = visualize and (i == 0)
      interfaces.append(self._get_interface(interface_format, require_raw))

    if self._num_agents == 1:
      self._launch_sp(map_inst, interfaces[0])
    else:
      self._launch_mp(map_inst, interfaces)

    self._finalize(agent_interface_format, interfaces, visualize)

  def _finalize(self, agent_interface_formats, interfaces, visualize):
    game_info = self._parallel.run(c.game_info for c in self._controllers)
    if not self._map_name:
      self._map_name = game_info[0].map_name

    for g, interface in zip(game_info, interfaces):
      if g.options.render != interface.render:
        logging.warning(
            "Actual interface options don't match requested options:\n"
            "Requested:\n%s\n\nActual:\n%s", interface, g.options)

    self._features = [
        features.features_from_game_info(
            game_info=g,
            use_feature_units=agent_interface_format.use_feature_units,
            use_raw_units=agent_interface_format.use_raw_units,
            use_unit_counts=agent_interface_format.use_unit_counts,
            use_camera_position=agent_interface_format.use_camera_position,
            action_space=agent_interface_format.action_space,
            hide_specific_actions=agent_interface_format.hide_specific_actions)
        for g, agent_interface_format in zip(game_info, agent_interface_formats)
    ]

    if visualize:
      static_data = self._controllers[0].data()
      self._renderer_human = renderer_human.RendererHuman()
      self._renderer_human.init(game_info[0], static_data)
    else:
      self._renderer_human = None

    self._metrics = metrics.Metrics(self._map_name)
    self._metrics.increment_instance()

    self._last_score = None
    self._total_steps = 0
    self._episode_steps = 0
    self._episode_count = 0
    self._obs = [None] * len(interfaces)
    self._agent_obs = [None] * len(interfaces)
    self._state = environment.StepType.LAST  # Want to jump to `reset`.
    logging.info("Environment is ready on map: %s", self._map_name)

  @staticmethod
  def _get_interface(agent_interface_format, require_raw):
    interface = sc_pb.InterfaceOptions(
        raw=(agent_interface_format.use_feature_units or
             agent_interface_format.use_unit_counts or
             agent_interface_format.use_raw_units or
             require_raw),
        score=True)

    if agent_interface_format.feature_dimensions:
      interface.feature_layer.width = (
          agent_interface_format.camera_width_world_units)
      agent_interface_format.feature_dimensions.screen.assign_to(
          interface.feature_layer.resolution)
      agent_interface_format.feature_dimensions.minimap.assign_to(
          interface.feature_layer.minimap_resolution)

    if agent_interface_format.rgb_dimensions:
      agent_interface_format.rgb_dimensions.screen.assign_to(
          interface.render.resolution)
      agent_interface_format.rgb_dimensions.minimap.assign_to(
          interface.render.minimap_resolution)

    return interface

  def _launch_sp(self, map_inst, interface):
    self._sc2_procs = [self._run_config.start(
        want_rgb=interface.HasField("render"))]
    self._controllers = [p.controller for p in self._sc2_procs]

    # 创建游戏。
    create = sc_pb.RequestCreateGame(
        local_map=sc_pb.LocalMap(
            map_path=map_inst.path, map_data=map_inst.data(self._run_config)),
        disable_fog=self._disable_fog,
        realtime=self._realtime)
    agent = Agent(Race.random)
    for p in self._players:
      if isinstance(p, Agent):
        create.player_setup.add(type=sc_pb.Participant)
        agent = p
      else:
        create.player_setup.add(type=sc_pb.Computer, race=p.race,
                                difficulty=p.difficulty)
    if self._random_seed is not None:
      create.random_seed = self._random_seed
    self._controllers[0].create_game(create)

    join = sc_pb.RequestJoinGame(
        options=interface, race=agent.race, player_name=agent.name)
    self._controllers[0].join_game(join)

  def _launch_mp(self, map_inst, interfaces):
    # 为奇怪的多人游戏实现保留一大堆端口。
    self._ports = portspicker.pick_unused_ports(self._num_agents * 2)
    logging.info("Ports used for multiplayer: %s", self._ports)

    # 实际启动游戏进程。
    self._sc2_procs = [
        self._run_config.start(extra_ports=self._ports,
                               want_rgb=interface.HasField("render"))
        for interface in interfaces]
    self._controllers = [p.controller for p in self._sc2_procs]

    # 保存地图，这样他们就可以访问它。从SC2开始就不要并行执行
    #不尊重windows上的tmpdir，导致竞态条件:
    # https://github.com/Blizzard/s2client-proto/issues/102
    for c in self._controllers:
      c.save_map(map_inst.path, map_inst.data(self._run_config))

    # 创建游戏。将第一个实例设置为主机。
    create = sc_pb.RequestCreateGame(
        local_map=sc_pb.LocalMap(
            map_path=map_inst.path),
        disable_fog=self._disable_fog,
        realtime=self._realtime)
    if self._random_seed is not None:
      create.random_seed = self._random_seed
    for p in self._players:
      if isinstance(p, Agent):
        create.player_setup.add(type=sc_pb.Participant)
      else:
        create.player_setup.add(type=sc_pb.Computer, race=p.race,
                                difficulty=p.difficulty)
    self._controllers[0].create_game(create)

    # 创建连接请求。
    agent_players = (p for p in self._players if isinstance(p, Agent))
    join_reqs = []
    for agent_index, p in enumerate(agent_players):
      ports = self._ports[:]
      join = sc_pb.RequestJoinGame(options=interfaces[agent_index])
      join.shared_port = 0  # unused
      join.server_ports.game_port = ports.pop(0)
      join.server_ports.base_port = ports.pop(0)
      for _ in range(self._num_agents - 1):
        join.client_ports.add(game_port=ports.pop(0),
                              base_port=ports.pop(0))

      join.race = p.race
      join.player_name = p.name
      join_reqs.append(join)

    # 创建加入游戏。这必须并行运行，因为连接是阻塞连接请求的。
    # 调用等待所有客户端都加入的游戏。
    self._parallel.run((c.join_game, join)
                       for c, join in zip(self._controllers, join_reqs))

    # Save them for restart.
    self._create_req = create
    self._join_reqs = join_reqs

  def observation_spec(self):
    """查看完整规范的特性."""
    return tuple(f.observation_spec() for f in self._features)

  def action_spec(self):
    """查看FLook的功能，以获得完整的规格说明."""
    return tuple(f.action_spec() for f in self._features)

  def _restart(self):
    if len(self._controllers) == 1:
      self._controllers[0].restart()
    else:
      self._parallel.run(c.leave for c in self._controllers)
      self._controllers[0].create_game(self._create_req)
      self._parallel.run((c.join_game, j)
                         for c, j in zip(self._controllers, self._join_reqs))

  @sw.decorate
  def reset(self):
    """开始新的一集."""
    self._episode_steps = 0
    if self._episode_count:
      # No need to restart for the first episode.
      self._restart()

    self._episode_count += 1
    logging.info("Starting episode: %s", self._episode_count)
    self._metrics.increment_episode()

    self._last_score = [0] * self._num_agents
    self._state = environment.StepType.FIRST
    if self._realtime:
      self._last_step_time = time.time()
      self._target_step = 0

    return self._observe()

  @sw.decorate("step_env")
  def step(self, actions, step_mul=None):
    """应用行动，让世界前进，并回报观察。

    Args:
      actions: 符合操作规范的操作列表，每个智能体一个。
      step_mul: 如果指定了，使用这个而不是环境的默认值。

    Returns:
      TimeStep命名元组的元组，每个智能体一个。
    """
    if self._state == environment.StepType.LAST:
      return self.reset()

    skip = not self._ensure_available_actions
    self._parallel.run(
        (c.act, f.transform_action(o.observation, a, skip_available=skip))
        for c, f, o, a in zip(
            self._controllers, self._features, self._obs, actions))

    self._state = environment.StepType.MID
    return self._step(step_mul)

  def _step(self, step_mul=None):
    step_mul = step_mul or self._step_mul
    if step_mul <= 0:
      raise ValueError("step_mul should be positive, got {}".format(step_mul))

    if not self._realtime:
      with self._metrics.measure_step_time(step_mul):
        self._parallel.run((c.step, step_mul)
                           for c in self._controllers)
    else:
      self._target_step = self._episode_steps + step_mul
      next_step_time = self._last_step_time + (
          step_mul * REALTIME_GAME_LOOP_SECONDS)

      wait_time = next_step_time - time.time()
      if wait_time > 0.0:
        time.sleep(wait_time)

      # 注意，这里使用的是目标next_step_time，而不是实际时间
      #时间。这是为了让我们提前看到SC2的游戏时钟
      # REALTIME_GAME_LOOP_SECONDS是递增的，而不是下滑
      #往返延误。
      self._last_step_time = next_step_time

    return self._observe()

  def _get_observations(self):
    with self._metrics.measure_observation_time():
      self._obs = self._parallel.run(c.observe for c in self._controllers)
      self._agent_obs = [f.transform_obs(o)
                         for f, o in zip(self._features, self._obs)]

  def _observe(self):
    if not self._realtime:
      self._get_observations()
    else:
      needed_to_wait = False
      while True:
        self._get_observations()

        # 检查游戏是否有足够的进展。
        # 如果还没有，那就等着它来吧。
        game_loop = self._agent_obs[0].game_loop[0]
        if game_loop < self._target_step:
          if not needed_to_wait:
            needed_to_wait = True
            logging.info(
                "Target step is %s, game loop is %s, waiting...",
                self._target_step,
                game_loop)

          time.sleep(REALTIME_GAME_LOOP_SECONDS)
        else:
          # 我们现在超出了目标。
          if needed_to_wait:
            self._last_step_time = time.time()
            logging.info("...game loop is now %s. Continuing.", game_loop)
          break

    #TODO(tewalds):我们应该如何处理两个以上的智能体以及在什么情况下
    #对于某些特工来说，这一集会提前结束吗?
    outcome = [0] * self._num_agents
    discount = self._discount
    episode_complete = any(o.player_result for o in self._obs)

    #在实时，我们不可靠地接收玩家的结果，但我们做到了
    #有时点击“结束”状态。当这种情况发生时，我们终止
    #集。
    # TODO(b/115466611): player_results应该以实时模式返回
    if self._realtime and self._controllers[0].status == protocol.Status.ended:
      logging.info("Protocol status is ended. Episode is complete.")
      episode_complete = True

    if self._realtime and len(self._obs) > 1:
      # Realtime似乎不会给我们一个玩家结果当一个玩家
      #被消除。因此出现了一些临时的黑客攻击(只能起作用)
      #当我们在这个环境中有两个代理时)…
      # TODO(b/115466611): player_results应该以实时模式返回
      p1 = self._obs[0].observation.score.score_details
      p2 = self._obs[1].observation.score.score_details
      if p1.killed_value_structures > p2.total_value_structures - EPSILON:
        logging.info("The episode appears to be complete, p1 killed p2.")
        episode_complete = True
        outcome[0] = 1.0
        outcome[1] = -1.0
      elif p2.killed_value_structures > p1.total_value_structures - EPSILON:
        logging.info("The episode appears to be complete, p2 killed p1.")
        episode_complete = True
        outcome[0] = -1.0
        outcome[1] = 1.0

    if episode_complete:
      self._state = environment.StepType.LAST
      discount = 0
      for i, o in enumerate(self._obs):
        player_id = o.observation.player_common.player_id
        for result in o.player_result:
          if result.player_id == player_id:
            outcome[i] = possible_results.get(result.result, 0)

    if self._score_index >= 0:  # 游戏得分，而不是输赢奖励。
      cur_score = [o["score_cumulative"][self._score_index]
                   for o in self._agent_obs]
      if self._episode_steps == 0:  # 第一个奖励总是0。
        reward = [0] * self._num_agents
      else:
        reward = [cur - last for cur, last in zip(cur_score, self._last_score)]
      self._last_score = cur_score
    else:
      reward = outcome

    if self._renderer_human:
      self._renderer_human.render(self._obs[0])
      cmd = self._renderer_human.get_actions(
          self._run_config, self._controllers[0])
      if cmd == renderer_human.ActionCmd.STEP:
        pass
      elif cmd == renderer_human.ActionCmd.RESTART:
        self._state = environment.StepType.LAST
      elif cmd == renderer_human.ActionCmd.QUIT:
        raise KeyboardInterrupt("Quit?")

    self._total_steps += self._agent_obs[0].game_loop[0] - self._episode_steps
    self._episode_steps = self._agent_obs[0].game_loop[0]
    if self._episode_length > 0 and self._episode_steps >= self._episode_length:
      self._state = environment.StepType.LAST
      if self._discount_zero_after_timeout:
        discount = 0.0

    if self._state == environment.StepType.LAST:
      if (self._save_replay_episodes > 0 and
          self._episode_count % self._save_replay_episodes == 0):
        self.save_replay(self._replay_dir, self._replay_prefix)
      logging.info(("Episode %s finished after %s game steps. "
                    "Outcome: %s, reward: %s, score: %s"),
                   self._episode_count, self._episode_steps, outcome, reward,
                   [o["score_cumulative"][0] for o in self._agent_obs])

    def zero_on_first_step(value):
      return 0.0 if self._state == environment.StepType.FIRST else value
    return tuple(environment.TimeStep(
        step_type=self._state,
        reward=zero_on_first_step(r * self._score_multiplier),
        discount=zero_on_first_step(discount),
        observation=o) for r, o in zip(reward, self._agent_obs))

  def send_chat_messages(self, messages):
    """用于将消息记录到重播中。"""
    self._parallel.run(
        (c.chat, message) for c, message in zip(self._controllers, messages))

  def save_replay(self, replay_dir, prefix=None):
    if prefix is None:
      prefix = self._map_name
    replay_path = self._run_config.save_replay(
        self._controllers[0].save_replay(), replay_dir, prefix)
    logging.info("Wrote replay to: %s", replay_path)
    return replay_path

  def close(self):
    logging.info("Environment Close")
    if hasattr(self, "_metrics") and self._metrics:
      self._metrics.close()
      self._metrics = None
    if hasattr(self, "_renderer_human") and self._renderer_human:
      self._renderer_human.close()
      self._renderer_human = None

    # 不要使用parallel，因为它可能会被异常破坏。
    if hasattr(self, "_controllers") and self._controllers:
      for c in self._controllers:
        c.quit()
      self._controllers = None
    if hasattr(self, "_sc2_procs") and self._sc2_procs:
      for p in self._sc2_procs:
        p.close()
      self._sc2_procs = None

    if hasattr(self, "_ports") and self._ports:
      portspicker.return_ports(self._ports)
      self._ports = None
