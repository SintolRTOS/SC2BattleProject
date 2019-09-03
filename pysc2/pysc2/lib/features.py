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
"""将SC2观察原型中的特征层呈现为numpy数组."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from absl import logging

import enum
import numpy as np
import six
from pysc2.lib import actions
from pysc2.lib import colors
from pysc2.lib import named_array
from pysc2.lib import point
from pysc2.lib import static_data
from pysc2.lib import stopwatch
from pysc2.lib import transform

from s2clientprotocol import raw_pb2 as sc_raw
from s2clientprotocol import sc2api_pb2 as sc_pb

sw = stopwatch.sw


class FeatureType(enum.Enum):
  SCALAR = 1
  CATEGORICAL = 2


class PlayerRelative(enum.IntEnum):
  """将SC2观察原型中的特征层呈现为numpy数组."""
  NONE = 0
  SELF = 1
  ALLY = 2
  NEUTRAL = 3
  ENEMY = 4


class Visibility(enum.IntEnum):
  """“可见性”特性层的值."""
  HIDDEN = 0
  SEEN = 1
  VISIBLE = 2


class Effects(enum.IntEnum):
  """“效果”特性层的值."""
  # pylint: disable=invalid-name
  PsiStorm = 1
  GuardianShield = 2
  TemporalFieldGrowing = 3
  TemporalField = 4
  ThermalLance = 5
  ScannerSweep = 6
  NukeDot = 7
  LiberatorDefenderZoneSetup = 8
  LiberatorDefenderZone = 9
  BlindingCloud = 10
  CorrosiveBile = 11
  LurkerSpines = 12
  # pylint: enable=invalid-name


class ScoreCumulative(enum.IntEnum):
  """进入“score_cumulative”观察的索引."""
  score = 0
  idle_production_time = 1
  idle_worker_time = 2
  total_value_units = 3
  total_value_structures = 4
  killed_value_units = 5
  killed_value_structures = 6
  collected_minerals = 7
  collected_vespene = 8
  collection_rate_minerals = 9
  collection_rate_vespene = 10
  spent_minerals = 11
  spent_vespene = 12


class ScoreByCategory(enum.IntEnum):
  """“score_by_category”观察的第一个维度的索引."""
  food_used = 0
  killed_minerals = 1
  killed_vespene = 2
  lost_minerals = 3
  lost_vespene = 4
  friendly_fire_minerals = 5
  friendly_fire_vespene = 6
  used_minerals = 7
  used_vespene = 8
  total_used_minerals = 9
  total_used_vespene = 10


class ScoreCategories(enum.IntEnum):
  """“score_by_category”观察的第二个维度的索引."""
  none = 0
  army = 1
  economy = 2
  technology = 3
  upgrade = 4


class ScoreByVital(enum.IntEnum):
  """“score_by_vital”观测的第一个维度的索引."""
  total_damage_dealt = 0
  total_damage_taken = 1
  total_healed = 2


class ScoreVitals(enum.IntEnum):
  """“score_by_vital”观测的第二个维度的索引."""
  life = 0
  shields = 1
  energy = 2


class Player(enum.IntEnum):
  """进入“玩家”观察的索引."""
  player_id = 0
  minerals = 1
  vespene = 2
  food_used = 3
  food_cap = 4
  food_army = 5
  food_workers = 6
  idle_worker_count = 7
  army_count = 8
  warp_gate_count = 9
  larva_count = 10


class UnitLayer(enum.IntEnum):
  """观察到的单位层的指数."""
  unit_type = 0
  player_relative = 1
  health = 2
  shields = 3
  energy = 4
  transport_slots_taken = 5
  build_progress = 6


class UnitCounts(enum.IntEnum):
  """观察到的单位层的指数."""
  unit_type = 0
  count = 1


class FeatureUnit(enum.IntEnum):
  """“特征单元”观测的索引."""
  unit_type = 0
  alliance = 1
  health = 2
  shield = 3
  energy = 4
  cargo_space_taken = 5
  build_progress = 6
  health_ratio = 7
  shield_ratio = 8
  energy_ratio = 9
  display_type = 10
  owner = 11
  x = 12
  y = 13
  facing = 14
  radius = 15
  cloak = 16
  is_selected = 17
  is_blip = 18
  is_powered = 19
  mineral_contents = 20
  vespene_contents = 21
  cargo_space_max = 22
  assigned_harvesters = 23
  ideal_harvesters = 24
  weapon_cooldown = 25
  order_length = 26  # If zero, the unit is idle.
  tag = 27  # Unique identifier for a unit (only populated for raw units).


class Feature(collections.namedtuple(
    "Feature", ["index", "name", "layer_set", "full_name", "scale", "type",
                "palette", "clip"])):
  """定义特性层的属性.

  Attributes:
    index: 将该层的索引放入一组层中.
    name: 集合中层的名称.
    layer_set: 在观察原型中查看哪一组特征层.
    full_name: 包括可视化的全称.
    scale: 此层的最大值(+1)，用于缩放值.
    type: 标量与分类的特征类型.
    palette: 用于呈现的调色板.
    clip: 是否剪辑着色值.
  """
  __slots__ = ()

  dtypes = {
      1: np.uint8,
      8: np.uint8,
      16: np.uint16,
      32: np.int32,
  }

  def unpack(self, obs):
    """返回此功能的正确形状的numpy数组."""
    planes = getattr(obs.feature_layer_data, self.layer_set)
    plane = getattr(planes, self.name)
    return self.unpack_layer(plane)

  @staticmethod
  @sw.decorate
  def unpack_layer(plane):
    """返回给定特征层字节的正确形状的numpy数组."""
    size = point.Point.build(plane.size)
    if size == (0, 0):
      # 这个SC2版本中没有实现的新层.
      return None
    data = np.frombuffer(plane.data, dtype=Feature.dtypes[plane.bits_per_pixel])
    if plane.bits_per_pixel == 1:
      data = np.unpackbits(data)
      if data.shape[0] != size.x * size.y:
        # 如果正确的长度不是8的倍数，就会发生这种情况
        # 到字符串末尾的一些填充位，这些填充位是不正确的
        # 解释为数据.
        data = data[:size.x * size.y]
    return data.reshape(size.y, size.x)

  @staticmethod
  @sw.decorate
  def unpack_rgb_image(plane):
    """返回给定图像字节的正确形状的numpy数组."""
    assert plane.bits_per_pixel == 24, "{} != 24".format(plane.bits_per_pixel)
    size = point.Point.build(plane.size)
    data = np.frombuffer(plane.data, dtype=np.uint8)
    return data.reshape(size.y, size.x, 3)

  @sw.decorate
  def color(self, plane):
    if self.clip:
      plane = np.clip(plane, 0, self.scale - 1)
    return self.palette[plane]


class ScreenFeatures(collections.namedtuple("ScreenFeatures", [
    "height_map", "visibility_map", "creep", "power", "player_id",
    "player_relative", "unit_type", "selected", "unit_hit_points",
    "unit_hit_points_ratio", "unit_energy", "unit_energy_ratio", "unit_shields",
    "unit_shields_ratio", "unit_density", "unit_density_aa", "effects"])):
  """The set of screen feature layers."""
  __slots__ = ()

  def __new__(cls, **kwargs):
    feats = {}
    for name, (scale, type_, palette, clip) in six.iteritems(kwargs):
      feats[name] = Feature(
          index=ScreenFeatures._fields.index(name),
          name=name,
          layer_set="renders",
          full_name="screen " + name,
          scale=scale,
          type=type_,
          palette=palette(scale) if callable(palette) else palette,
          clip=clip)
    return super(ScreenFeatures, cls).__new__(cls, **feats)  # pytype:禁用=少的参数


class MinimapFeatures(collections.namedtuple("MinimapFeatures", [
    "height_map", "visibility_map", "creep", "camera", "player_id",
    "player_relative", "selected"])):
  """The set of minimap feature layers."""
  __slots__ = ()

  def __new__(cls, **kwargs):
    feats = {}
    for name, (scale, type_, palette) in six.iteritems(kwargs):
      feats[name] = Feature(
          index=MinimapFeatures._fields.index(name),
          name=name,
          layer_set="minimap_renders",
          full_name="minimap " + name,
          scale=scale,
          type=type_,
          palette=palette(scale) if callable(palette) else palette,
          clip=False)
    return super(MinimapFeatures, cls).__new__(cls, **feats)  # pytype:禁用=少的参数


SCREEN_FEATURES = ScreenFeatures(
    height_map=(256, FeatureType.SCALAR, colors.winter, False),
    visibility_map=(4, FeatureType.CATEGORICAL,
                    colors.VISIBILITY_PALETTE, False),
    creep=(2, FeatureType.CATEGORICAL, colors.CREEP_PALETTE, False),
    power=(2, FeatureType.CATEGORICAL, colors.POWER_PALETTE, False),
    player_id=(17, FeatureType.CATEGORICAL,
               colors.PLAYER_ABSOLUTE_PALETTE, False),
    player_relative=(5, FeatureType.CATEGORICAL,
                     colors.PLAYER_RELATIVE_PALETTE, False),
    unit_type=(max(static_data.UNIT_TYPES) + 1, FeatureType.CATEGORICAL,
               colors.unit_type, False),
    selected=(2, FeatureType.CATEGORICAL, colors.SELECTED_PALETTE, False),
    unit_hit_points=(1600, FeatureType.SCALAR, colors.hot, True),
    unit_hit_points_ratio=(256, FeatureType.SCALAR, colors.hot, False),
    unit_energy=(1000, FeatureType.SCALAR, colors.hot, True),
    unit_energy_ratio=(256, FeatureType.SCALAR, colors.hot, False),
    unit_shields=(1000, FeatureType.SCALAR, colors.hot, True),
    unit_shields_ratio=(256, FeatureType.SCALAR, colors.hot, False),
    unit_density=(16, FeatureType.SCALAR, colors.hot, True),
    unit_density_aa=(256, FeatureType.SCALAR, colors.hot, False),
    effects=(16, FeatureType.CATEGORICAL, colors.effects, False),
)

MINIMAP_FEATURES = MinimapFeatures(
    height_map=(256, FeatureType.SCALAR, colors.winter),
    visibility_map=(4, FeatureType.CATEGORICAL, colors.VISIBILITY_PALETTE),
    creep=(2, FeatureType.CATEGORICAL, colors.CREEP_PALETTE),
    camera=(2, FeatureType.CATEGORICAL, colors.CAMERA_PALETTE),
    player_id=(17, FeatureType.CATEGORICAL, colors.PLAYER_ABSOLUTE_PALETTE),
    player_relative=(5, FeatureType.CATEGORICAL,
                     colors.PLAYER_RELATIVE_PALETTE),
    selected=(2, FeatureType.CATEGORICAL, colors.winter),
)


def _to_point(dims):
  """转换(宽度，高度)或大小->点."""
  assert dims

  if isinstance(dims, (tuple, list)):
    if len(dims) != 2:
      raise ValueError(
          "A two element tuple or list is expected here, got {}.".format(dims))
    else:
      width = int(dims[0])
      height = int(dims[1])
      if width <= 0 or height <= 0:
        raise ValueError("Must specify +ve dims, got {}.".format(dims))
      else:
        return point.Point(width, height)
  else:
    size = int(dims)
    if size <= 0:
      raise ValueError(
          "Must specify a +ve value for size, got {}.".format(dims))
    else:
      return point.Point(size, size)


class Dimensions(object):
  """屏幕和最小ap尺寸配置.

    必须指定screen和minimap。大小必须为正。
    屏幕大小必须大于或等于两个维度上的最小ap大小。

  Args:
    screen: 一个(宽，高)整型元组或一个整型用于两者.
    minimap: 一个(宽，高)整型元组或一个整型用于两者.
  """

  def __init__(self, screen=None, minimap=None):
    if not screen or not minimap:
      raise ValueError(
          "screen and minimap must both be set, screen={}, minimap={}".format(
              screen, minimap))

    self._screen = _to_point(screen)
    self._minimap = _to_point(minimap)

    if self._screen.x < self._minimap.x or self._screen.y < self._minimap.y:
      raise ValueError(
          "Screen (%s) can't be smaller than the minimap (%s)." % (
              self._screen, self._minimap))

  @property
  def screen(self):
    return self._screen

  @property
  def minimap(self):
    return self._minimap

  def __repr__(self):
    return "Dimensions(screen={}, minimap={})".format(self.screen, self.minimap)


class AgentInterfaceFormat(object):
  """特定于特定智能体的观察和操作接口格式."""

  def __init__(
      self,
      feature_dimensions=None,
      rgb_dimensions=None,
      action_space=None,
      camera_width_world_units=None,
      use_feature_units=False,
      use_raw_units=False,
      use_unit_counts=False,
      use_camera_position=False,
      hide_specific_actions=True):
    """Initializer.

    Args:
      feature_dimensions: 功能层的维度。这或
          必须设置rgb_dimensions(或两者都设置)t.
      rgb_dimensions: RGB“维度”。或者是这个，或者是feature_dimensions
          必须设置(或两者都设置).
      action_space: 如果同时传递特性和rgb大小，那么也必须这样做
          指定要将哪个操作用作ActionSpace枚举.
      camera_width_world_units: 屏幕的宽度(以世界为单位)。如果你的
            feature_dimensions。screen=(64, 48)， camera_width = 24，则每个
            px表示x和y的24 / 64 = 0.375个世界单位。
            然后它将表示一个大小为(24,0.375 * 48)=(24,18)的相机
            世界上单位.
      use_feature_units: 是否在观测中包含特征单元数据.
      use_raw_units: 是否在观测中包含原始单元数据。这
            与feature_units不同，因为它包含了
            屏幕和隐藏单元，因为单元的位置是给定的
            世界单位的术语，而不是屏幕单位.
      use_unit_counts: 是否包括unit_counts观察值。残疾的
          默认值，因为它提供了可见区域之外的信息.
      use_camera_position:是否包括相机的位置(在世界上单位).
      hide_specific_actions: 有些动作(如取消)有很多
            特定的版本(取消这个建筑，取消那个咒语)可以
            用更一般的形式表示。如果一个特定的行为是
            可用，一般也会可用。如果你设置
            ' hide_specific_actions '为False，具体版本也将为
            可用，但如果它是真的，特定的将被隐藏。
            类似地，当转换返回时，将返回特定的操作
            作为一般动作。虽然这可以简化操作空间
            导致重放中的某些操作无法精确表示
            只有一般动作.

    Raises:
      ValueError: 如果参数不一致.
    """

    if not feature_dimensions and not rgb_dimensions:
      raise ValueError("Must set either the feature layer or rgb dimensions.")

    if action_space:
      if not isinstance(action_space, actions.ActionSpace):
        raise ValueError("action_space must be of type ActionSpace.")

      if ((action_space == actions.ActionSpace.FEATURES and
           not feature_dimensions) or
          (action_space == actions.ActionSpace.RGB and
           not rgb_dimensions)):
        raise ValueError(
            "Action space must match the observations, action space={}, "
            "feature_dimensions={}, rgb_dimensions={}".format(
                action_space, feature_dimensions, rgb_dimensions))
    else:
      if feature_dimensions and rgb_dimensions:
        raise ValueError(
            "You must specify the action space if you have both screen and "
            "rgb observations.")
      elif feature_dimensions:
        action_space = actions.ActionSpace.FEATURES
      else:
        action_space = actions.ActionSpace.RGB

    self._feature_dimensions = feature_dimensions
    self._rgb_dimensions = rgb_dimensions
    self._action_space = action_space
    self._camera_width_world_units = camera_width_world_units or 24
    self._use_feature_units = use_feature_units
    self._use_raw_units = use_raw_units
    self._use_unit_counts = use_unit_counts
    self._use_camera_position = use_camera_position
    self._hide_specific_actions = hide_specific_actions

    if action_space == actions.ActionSpace.FEATURES:
      self._action_dimensions = feature_dimensions
    else:
      self._action_dimensions = rgb_dimensions

  @property
  def feature_dimensions(self):
    return self._feature_dimensions

  @property
  def rgb_dimensions(self):
    return self._rgb_dimensions

  @property
  def action_space(self):
    return self._action_space

  @property
  def camera_width_world_units(self):
    return self._camera_width_world_units

  @property
  def use_feature_units(self):
    return self._use_feature_units

  @property
  def use_raw_units(self):
    return self._use_raw_units

  @property
  def use_unit_counts(self):
    return self._use_unit_counts

  @property
  def use_camera_position(self):
    return self._use_camera_position

  @property
  def hide_specific_actions(self):
    return self._hide_specific_actions

  @property
  def action_dimensions(self):
    return self._action_dimensions


def parse_agent_interface_format(
    feature_screen=None,
    feature_minimap=None,
    rgb_screen=None,
    rgb_minimap=None,
    action_space=None,
    camera_width_world_units=None,
    use_feature_units=False,
    use_raw_units=False,
    use_unit_counts=False,
    use_camera_position=False):
  """从关键字args创建一个AgentInterfaceFormat对象.

      使用字典或命令行参数进行配置时非常方便。
        注意，feature_*和rgb_*属性定义了各自的空间
        观察尺寸及接受:
        * None或0来禁用空间观察。
        *一个单独的int值，用于具有该边长度的正方形观察。
        *一个(int, int)元组用于矩形(宽，高)观察.

  Args:
    feature_screen: 如果指定，则必须使用 feature_minimap be.
    feature_minimap: 如果指定，则必须使用 feature_screen be.
    rgb_screen: 如果指定，则必须使用 rgb_minimap be.
    rgb_minimap: 如果指定，则必须使用 rgb_screen be.
    action_space: ["FEATURES", "RGB"].
    camera_width_world_units: An int.
    use_feature_units: 布尔值，默认为False.
    use_raw_units: 布尔值，默认为False.
    use_unit_counts: 布尔值，默认为False.
    use_camera_position: 布尔值，默认为False.

  Returns:
    一个“AgentInterfaceFormat”对象.

  Raises:
    ValueError: 如果指定了无效参数.
  """
  if feature_screen or feature_minimap:
    feature_dimensions = Dimensions(
        screen=feature_screen,
        minimap=feature_minimap)
  else:
    feature_dimensions = None

  if rgb_screen or rgb_minimap:
    rgb_dimensions = Dimensions(
        screen=rgb_screen,
        minimap=rgb_minimap)
  else:
    rgb_dimensions = None

  return AgentInterfaceFormat(
      feature_dimensions=feature_dimensions,
      rgb_dimensions=rgb_dimensions,
      action_space=(action_space and actions.ActionSpace[action_space.upper()]),
      camera_width_world_units=camera_width_world_units,
      use_feature_units=use_feature_units,
      use_raw_units=use_raw_units,
      use_unit_counts=use_unit_counts,
      use_camera_position=use_camera_position
  )


def features_from_game_info(
    game_info,
    use_feature_units=False,
    use_raw_units=False,
    action_space=None,
    hide_specific_actions=True,
    use_unit_counts=False,
    use_camera_position=False):
  """使用从游戏信息中提取的数据构造一个特征对象.

  Args:
    game_info: sc pb.ResponseGameInfo '从游戏中获得.
    use_feature_units: 是否包含特征单元观察.
    use_raw_units: 是否在观测中包含原始单元数据。这
        与feature_units不同，因为它包含了
        屏幕和隐藏单元，因为单元的位置是给定的
        世界单位的术语，而不是屏幕单位.
    action_space: 如果同时传递特性和rgb大小，那么也必须这样做
        指定要将哪个操作用作ActionSpace枚举.
    hide_specific_actions: 有些动作(如取消)有很多
        特定的版本(取消这个建筑，取消那个咒语)可以
        用更一般的形式表示。如果一个特定的行为是
        可用，一般也会可用。如果你设置
        ' hide_specific_actions '为False，具体版本也将为
        可用，但如果它是真的，特定的将被隐藏。
        类似地，当转换返回时，将返回特定的操作
        作为一般动作。虽然这可以简化操作空间
        导致重放中的某些操作无法精确表示
        只有一般动作.
    use_unit_counts: 是否包括unit_counts观察值。残疾的
        默认值，因为它提供了可见区域之外的信息.
    use_camera_position: 是否包括相机的位置(在世界上单位).

  Returns:
    与指定参数匹配的特性对象.

  """

  if game_info.options.HasField("feature_layer"):
    fl_opts = game_info.options.feature_layer
    feature_dimensions = Dimensions(
        screen=(fl_opts.resolution.x, fl_opts.resolution.y),
        minimap=(fl_opts.minimap_resolution.x, fl_opts.minimap_resolution.y))
  else:
    feature_dimensions = None

  if game_info.options.HasField("render"):
    rgb_opts = game_info.options.render
    rgb_dimensions = Dimensions(
        screen=(rgb_opts.resolution.x, rgb_opts.resolution.y),
        minimap=(rgb_opts.minimap_resolution.x, rgb_opts.minimap_resolution.y))
  else:
    rgb_dimensions = None

  map_size = game_info.start_raw.map_size
  camera_width_world_units = game_info.options.feature_layer.width

  return Features(
      agent_interface_format=AgentInterfaceFormat(
          feature_dimensions=feature_dimensions,
          rgb_dimensions=rgb_dimensions,
          use_feature_units=use_feature_units,
          use_raw_units=use_raw_units,
          use_unit_counts=use_unit_counts,
          use_camera_position=use_camera_position,
          camera_width_world_units=camera_width_world_units,
          action_space=action_space,
          hide_specific_actions=hide_specific_actions),
      map_size=map_size)


def _init_valid_functions(action_dimensions):
  """初始化validfunction并设置回调."""
  sizes = {
      "screen": tuple(int(i) for i in action_dimensions.screen),
      "screen2": tuple(int(i) for i in action_dimensions.screen),
      "minimap": tuple(int(i) for i in action_dimensions.minimap),
  }

  types = actions.Arguments(*[
      actions.ArgumentType.spec(t.id, t.name, sizes.get(t.name, t.sizes))
      for t in actions.TYPES])

  functions = actions.Functions([
      actions.Function.spec(f.id, f.name, tuple(types[t.id] for t in f.args))
      for f in actions.FUNCTIONS])

  return actions.ValidActions(types, functions)


class Features(object):
  """将SC2观察原型中的特征层呈现为numpy数组.

    这里有如何渲染星际争霸环境的实现细节。
    它在代理动作/观察格式和星际争霸之间转换
    操作/观察格式，代理作者不应该看到。的
    《星际争霸前传》包含的信息比他们应该获得的更多。
    这是外部的环境，所以它也可以用于其他
    上下文(如受监督的数据集管道).
  """

  def __init__(self, agent_interface_format=None, map_size=None):
    """初始化与指定接口格式匹配的功能部件实例.

    Args:
      agent_interface_format: 参见“AgentInterfaceFormat”文档.
      map_size: 以世界单位表示的地图大小，这是feature_units所需要的.

    Raises:
      ValueError: 如果没有指定智能体接口格式.
      ValueError: 如果在use_feature_units或use_camera_position是.
    """
    if not agent_interface_format:
      raise ValueError("Please specify agent_interface_format")

    self._agent_interface_format = agent_interface_format
    aif = self._agent_interface_format

    if (aif.use_feature_units
        or aif.use_camera_position
        or aif.use_raw_units):
      self.init_camera(
          aif.feature_dimensions,
          map_size,
          aif.camera_width_world_units)

    self._valid_functions = _init_valid_functions(
        aif.action_dimensions)

  def init_camera(
      self, feature_dimensions, map_size, camera_width_world_units):
    """初始化相机(特别是对于feature_units).

    这在构造函数中调用，并且可以在调用之后重复调用
    “功能”是构建的，因为它处理重新缩放坐标而不是
    改变环境/行为规范.

    Args:
      feature_dimensions: 参见“AgentInterfaceFormat”文档.
      map_size: 地图的大小以世界为单位.
      camera_width_world_units: 参见“AgentInterfaceFormat”文档.

    Raises:
      ValueError: 如果map_size或camera_width_world_units是falsey(其中
                  应该主要发生在构造函数调用时).
    """
    if not map_size or not camera_width_world_units:
      raise ValueError(
          "Either pass the game_info with raw enabled, or map_size and "
          "camera_width_world_units in order to use feature_units or camera"
          "position.")
    map_size = point.Point.build(map_size)
    self._world_to_world_tl = transform.Linear(point.Point(1, -1),
                                               point.Point(0, map_size.y))
    self._world_tl_to_world_camera_rel = transform.Linear(offset=-map_size / 4)
    world_camera_rel_to_feature_screen = transform.Linear(
        feature_dimensions.screen / camera_width_world_units,
        feature_dimensions.screen / 2)
    self._world_to_feature_screen_px = transform.Chain(
        self._world_to_world_tl,
        self._world_tl_to_world_camera_rel,
        world_camera_rel_to_feature_screen,
        transform.PixelToCoord())

  def _update_camera(self, camera_center):
    """Update the camera transform based on the new camera center."""
    self._world_tl_to_world_camera_rel.offset = (
        -self._world_to_world_tl.fwd_pt(camera_center) *
        self._world_tl_to_world_camera_rel.scale)

  def observation_spec(self):
    """基于新的相机中心更新相机转换.

    值得注意的是，类似于图像的观察位于y、x/行、列中
    顺序不同于x y顺序的动作。这是
    由于约定冲突，并便于打印图像.

    Returns:
      观测的dict将它们的张量形状命名为。形状为0
      长度不同，例如有效操作的数量取决于哪个操作
      你选择的单位。
    """
    obs_spec = named_array.NamedDict({
        "action_result": (0,),  #看到错误。原型:ActionResult.
        "alerts": (0,),  # 看到sc2api。原型:警报.
        "available_actions": (0,),
        "build_queue": (0, len(UnitLayer)),  # pytype:禁用= wrong-arg-types
        "cargo": (0, len(UnitLayer)),  # pytype:禁用= wrong-arg-types
        "cargo_slots_available": (1,),
        "control_groups": (10, 2),
        "game_loop": (1,),
        "last_actions": (0,),
        "multi_select": (0, len(UnitLayer)),  # pytype:禁用= wrong-arg-types
        "player": (len(Player),),  # pytype:禁用= wrong-arg-types
        "score_cumulative": (len(ScoreCumulative),),  # pytype:禁用= wrong-arg-types
        "score_by_category": (len(ScoreByCategory), len(ScoreCategories)),  # pytype:禁用= wrong-arg-types
        "score_by_vital": (len(ScoreByVital), len(ScoreVitals)),  # pytype:禁用= wrong-arg-types
        "single_select": (0, len(UnitLayer)),  # Only (n, 7) for n in (0, 1).  # pytype: 禁用= wrong-arg-types
    })

    aif = self._agent_interface_format

    if aif.feature_dimensions:
      obs_spec["feature_screen"] = (len(SCREEN_FEATURES),
                                    aif.feature_dimensions.screen.y,
                                    aif.feature_dimensions.screen.x)

      obs_spec["feature_minimap"] = (len(MINIMAP_FEATURES),
                                     aif.feature_dimensions.minimap.y,
                                     aif.feature_dimensions.minimap.x)
    if aif.rgb_dimensions:
      obs_spec["rgb_screen"] = (aif.rgb_dimensions.screen.y,
                                aif.rgb_dimensions.screen.x,
                                3)
      obs_spec["rgb_minimap"] = (aif.rgb_dimensions.minimap.y,
                                 aif.rgb_dimensions.minimap.x,
                                 3)
    if aif.use_feature_units:
      obs_spec["feature_units"] = (0, len(FeatureUnit))  # 禁用= wrong-arg-types

    if aif.use_raw_units:
      obs_spec["raw_units"] = (0, len(FeatureUnit))

    if aif.use_unit_counts:
      obs_spec["unit_counts"] = (0, len(UnitCounts))

    if aif.use_camera_position:
      obs_spec["camera_position"] = (2,)
    return obs_spec

  def action_spec(self):
    """The action space pretty complicated and fills the ValidFunctions."""
    return self._valid_functions

  @sw.decorate
  def transform_obs(self, obs):
    """Render some SC2 observations into something an agent can handle."""
    empty = np.array([], dtype=np.int32).reshape((0, 7))
    out = named_array.NamedDict({  # 填一些有时是空的。
        "single_select": empty,
        "multi_select": empty,
        "build_queue": empty,
        "cargo": empty,
        "cargo_slots_available": np.array([0], dtype=np.int32),
    })

    def or_zeros(layer, size):
      if layer is not None:
        return layer.astype(np.int32, copy=False)
      else:
        return np.zeros((size.y, size.x), dtype=np.int32)

    aif = self._agent_interface_format

    if aif.feature_dimensions:
      out["feature_screen"] = named_array.NamedNumpyArray(
          np.stack(or_zeros(f.unpack(obs.observation),
                            aif.feature_dimensions.screen)
                   for f in SCREEN_FEATURES),
          names=[ScreenFeatures, None, None])
      out["feature_minimap"] = named_array.NamedNumpyArray(
          np.stack(or_zeros(f.unpack(obs.observation),
                            aif.feature_dimensions.minimap)
                   for f in MINIMAP_FEATURES),
          names=[MinimapFeatures, None, None])

    if aif.rgb_dimensions:
      out["rgb_screen"] = Feature.unpack_rgb_image(
          obs.observation.render_data.map).astype(np.int32)
      out["rgb_minimap"] = Feature.unpack_rgb_image(
          obs.observation.render_data.minimap).astype(np.int32)

    out["last_actions"] = np.array(
        [self.reverse_action(a).function for a in obs.actions],
        dtype=np.int32)

    out["action_result"] = np.array([o.result for o in obs.action_errors],
                                    dtype=np.int32)

    out["alerts"] = np.array(obs.observation.alerts, dtype=np.int32)

    out["game_loop"] = np.array([obs.observation.game_loop], dtype=np.int32)

    score_details = obs.observation.score.score_details
    out["score_cumulative"] = named_array.NamedNumpyArray([
        obs.observation.score.score,
        score_details.idle_production_time,
        score_details.idle_worker_time,
        score_details.total_value_units,
        score_details.total_value_structures,
        score_details.killed_value_units,
        score_details.killed_value_structures,
        score_details.collected_minerals,
        score_details.collected_vespene,
        score_details.collection_rate_minerals,
        score_details.collection_rate_vespene,
        score_details.spent_minerals,
        score_details.spent_vespene,
    ], names=ScoreCumulative, dtype=np.int32)

    def get_score_details(key, details, categories):
      row = getattr(details, key.name)
      return [getattr(row, category.name) for category in categories]

    out["score_by_category"] = named_array.NamedNumpyArray([
        get_score_details(key, score_details, ScoreCategories)
        for key in ScoreByCategory
    ], names=[ScoreByCategory, ScoreCategories], dtype=np.int32)

    out["score_by_vital"] = named_array.NamedNumpyArray([
        get_score_details(key, score_details, ScoreVitals)
        for key in ScoreByVital
    ], names=[ScoreByVital, ScoreVitals], dtype=np.int32)

    player = obs.observation.player_common
    out["player"] = named_array.NamedNumpyArray([
        player.player_id,
        player.minerals,
        player.vespene,
        player.food_used,
        player.food_cap,
        player.food_army,
        player.food_workers,
        player.idle_worker_count,
        player.army_count,
        player.warp_gate_count,
        player.larva_count,
    ], names=Player, dtype=np.int32)

    def unit_vec(u):
      return np.array((
          u.unit_type,
          u.player_relative,
          u.health,
          u.shields,
          u.energy,
          u.transport_slots_taken,
          int(u.build_progress * 100),  # 离散化
      ), dtype=np.int32)

    ui = obs.observation.ui_data

    with sw("ui"):
      groups = np.zeros((10, 2), dtype=np.int32)
      for g in ui.groups:
        groups[g.control_group_index, :] = (g.leader_unit_type, g.count)
      out["control_groups"] = groups

      if ui.single:
        out["single_select"] = named_array.NamedNumpyArray(
            [unit_vec(ui.single.unit)], [None, UnitLayer])

      if ui.multi and ui.multi.units:
        out["multi_select"] = named_array.NamedNumpyArray(
            [unit_vec(u) for u in ui.multi.units], [None, UnitLayer])

      if ui.cargo and ui.cargo.passengers:
        out["single_select"] = named_array.NamedNumpyArray(
            [unit_vec(ui.single.unit)], [None, UnitLayer])
        out["cargo"] = named_array.NamedNumpyArray(
            [unit_vec(u) for u in ui.cargo.passengers], [None, UnitLayer])
        out["cargo_slots_available"] = np.array([ui.cargo.slots_available],
                                                dtype=np.int32)

      if ui.production and ui.production.build_queue:
        out["single_select"] = named_array.NamedNumpyArray(
            [unit_vec(ui.production.unit)], [None, UnitLayer])
        out["build_queue"] = named_array.NamedNumpyArray(
            [unit_vec(u) for u in ui.production.build_queue],
            [None, UnitLayer])

    def full_unit_vec(u, pos_transform, is_raw=False):
      screen_pos = pos_transform.fwd_pt(
          point.Point.build(u.pos))
      screen_radius = pos_transform.fwd_dist(u.radius)
      return np.array((
          # 匹配单元vec订单
          u.unit_type,
          u.alliance,  # Self = 1, Ally = 2, Neutral = 3, Enemy = 4
          u.health,
          u.shield,
          u.energy,
          u.cargo_space_taken,
          int(u.build_progress * 100),  # 离散化

          # Resume API order
          int(u.health / u.health_max * 255) if u.health_max > 0 else 0,
          int(u.shield / u.shield_max * 255) if u.shield_max > 0 else 0,
          int(u.energy / u.energy_max * 255) if u.energy_max > 0 else 0,
          u.display_type,  # Visible = 1, Snapshot = 2, Hidden = 3
          u.owner,  # 1-15, 16 = neutral
          screen_pos.x,
          screen_pos.y,
          u.facing,
          screen_radius,
          u.cloak,  # Cloaked = 1, CloakedDetected = 2, NotCloaked = 3
          u.is_selected,
          u.is_blip,
          u.is_powered,
          u.mineral_contents,
          u.vespene_contents,

          # 不适合敌人或中立
          u.cargo_space_max,
          u.assigned_harvesters,
          u.ideal_harvesters,
          u.weapon_cooldown,
          len(u.orders),
          u.tag if is_raw else 0
      ), dtype=np.int32)

    raw = obs.observation.raw_data

    if aif.use_feature_units:
      with sw("feature_units"):
        # 更新相机位置，这样我们可以计算世界屏幕pos
        self._update_camera(point.Point.build(raw.player.camera))
        feature_units = []
        for u in raw.units:
          if u.is_on_screen and u.display_type != sc_raw.Hidden:
            feature_units.append(
                full_unit_vec(u, self._world_to_feature_screen_px))
        out["feature_units"] = named_array.NamedNumpyArray(
            feature_units, [None, FeatureUnit], dtype=np.int32)

    if aif.use_raw_units:
      with sw("raw_units"):
        raw_units = [full_unit_vec(u, self._world_to_world_tl, is_raw=True)
                     for u in raw.units]
        out["raw_units"] = named_array.NamedNumpyArray(
            raw_units, [None, FeatureUnit], dtype=np.int32)

    if aif.use_unit_counts:
      with sw("unit_counts"):
        unit_counts = collections.defaultdict(int)
        for u in raw.units:
          if u.alliance == sc_raw.Self:
            unit_counts[u.unit_type] += 1
        out["unit_counts"] = named_array.NamedNumpyArray(
            sorted(unit_counts.items()), [None, UnitCounts], dtype=np.int32)

    if aif.use_camera_position:
      camera_position = self._world_to_world_tl.fwd_pt(
          point.Point.build(raw.player.camera))
      out["camera_position"] = np.array((camera_position.x, camera_position.y),
                                        dtype=np.int32)

    out["available_actions"] = np.array(self.available_actions(obs.observation),
                                        dtype=np.int32)

    return out

  @sw.decorate
  def available_actions(self, obs):
    """返回可用操作id的列表."""
    available_actions = set()
    hide_specific_actions = self._agent_interface_format.hide_specific_actions
    for i, func in six.iteritems(actions.FUNCTIONS_AVAILABLE):
      if func.avail_fn(obs):
        available_actions.add(i)
    for a in obs.abilities:
      if a.ability_id not in actions.ABILITY_IDS:
        logging.warning("Unknown ability %s seen as available.", a.ability_id)
        continue
      for func in actions.ABILITY_IDS[a.ability_id]:
        if func.function_type in actions.POINT_REQUIRED_FUNCS[a.requires_point]:
          if func.general_id == 0 or not hide_specific_actions:
            available_actions.add(func.id)
          if func.general_id != 0:  #始终提供通用操作.
            for general_func in actions.ABILITY_IDS[func.general_id]:
              if general_func.function_type is func.function_type:
                #只有正确的类型。不想暴露一般动作
                #到小地图，如果只有屏幕版本可用.
                available_actions.add(general_func.id)
                break
    return list(available_actions)

  @sw.decorate
  def transform_action(self, obs, func_call, skip_available=False):
    """将代理样式的操作转换为SC2可以使用的操作.

    Args:
      obs: a `sc_pb.Observation` 前一帧.
      func_call: 将“FunctionCall”转换为“sc_pb.Action”.
      skip_available: 如果为真，则假设操作可用。这应该只
        用于测试，或者如果您希望执行没有执行的操作
        在最后一次观察中有效.

    Returns:
      对应的“sc pb.Action”.

    Raises:
      ValueError: 如果操作没有通过验证.
    """
    func_id = func_call.function
    try:
      func = actions.FUNCTIONS[func_id]
    except KeyError:
      raise ValueError("Invalid function id: %s." % func_id)

    # Available?
    if not (skip_available or func_id in self.available_actions(obs)):
      raise ValueError("Function %s/%s is currently not available" % (
          func_id, func.name))

    # Right number of args?
    if len(func_call.arguments) != len(func.args):
      raise ValueError(
          "Wrong number of arguments for function: %s, got: %s" % (
              func, func_call.arguments))

    # Args are valid?
    aif = self._agent_interface_format
    for t, arg in zip(func.args, func_call.arguments):
      if t.name in ("screen", "screen2"):
        sizes = aif.action_dimensions.screen
      elif t.name == "minimap":
        sizes = aif.action_dimensions.minimap
      else:
        sizes = t.sizes

      if len(sizes) != len(arg):
        raise ValueError(
            "Wrong number of values for argument of %s, got: %s" % (
                func, func_call.arguments))

      for s, a in zip(sizes, arg):
        if not 0 <= a < s:
          raise ValueError("Argument is out of range for %s, got: %s" % (
              func, func_call.arguments))

    # Convert them to python types.
    kwargs = {type_.name: type_.fn(a)
              for type_, a in zip(func.args, func_call.arguments)}

    # 将它们转换为python类型.
    sc2_action = sc_pb.Action()
    kwargs["action"] = sc2_action
    kwargs["action_space"] = aif.action_space
    if func.ability_id:
      kwargs["ability_id"] = func.ability_id
    actions.FUNCTIONS[func_id].function_type(**kwargs)
    return sc2_action

  @sw.decorate
  def reverse_action(self, action):
    """将sc2样式的操作转换为代理样式的操作.

    这应该是transform_action的倒数.

    Args:
      action: a `sc_pb.Action` 被转换.

    Returns:
      一个相应的“actions.FunctionCall”.

    Raises:
      ValueError: 如果它不知道如何转换这个动作.
    """
    FUNCTIONS = actions.FUNCTIONS  # pylint:禁用=无效名称

    aif = self._agent_interface_format

    def func_call_ability(ability_id, cmd_type, *args):
      """获取特定能力id和操作类型的函数id."""
      if ability_id not in actions.ABILITY_IDS:
        logging.warning("Unknown ability_id: %s. This is probably dance or "
                        "cheer, or some unknown new or map specific ability. "
                        "Treating it as a no-op.", ability_id)
        return FUNCTIONS.no_op()

      if aif.hide_specific_actions:
        general_id = next(iter(actions.ABILITY_IDS[ability_id])).general_id
        if general_id:
          ability_id = general_id

      for func in actions.ABILITY_IDS[ability_id]:
        if func.function_type is cmd_type:
          return FUNCTIONS[func.id](*args)
      raise ValueError("Unknown ability_id: %s, type: %s. Likely a bug." % (
          ability_id, cmd_type.__name__))

    if action.HasField("action_ui"):
      act_ui = action.action_ui
      if act_ui.HasField("multi_panel"):
        return FUNCTIONS.select_unit(act_ui.multi_panel.type - 1,
                                     act_ui.multi_panel.unit_index)
      if act_ui.HasField("control_group"):
        return FUNCTIONS.select_control_group(
            act_ui.control_group.action - 1,
            act_ui.control_group.control_group_index)
      if act_ui.HasField("select_idle_worker"):
        return FUNCTIONS.select_idle_worker(act_ui.select_idle_worker.type - 1)
      if act_ui.HasField("select_army"):
        return FUNCTIONS.select_army(act_ui.select_army.selection_add)
      if act_ui.HasField("select_warp_gates"):
        return FUNCTIONS.select_warp_gates(
            act_ui.select_warp_gates.selection_add)
      if act_ui.HasField("select_larva"):
        return FUNCTIONS.select_larva()
      if act_ui.HasField("cargo_panel"):
        return FUNCTIONS.unload(act_ui.cargo_panel.unit_index)
      if act_ui.HasField("production_panel"):
        return FUNCTIONS.build_queue(act_ui.production_panel.unit_index)
      if act_ui.HasField("toggle_autocast"):
        return func_call_ability(act_ui.toggle_autocast.ability_id,
                                 actions.autocast)

    if (action.HasField("action_feature_layer") or
        action.HasField("action_render")):
      act_sp = actions.spatial(action, aif.action_space)
      if act_sp.HasField("camera_move"):
        coord = point.Point.build(act_sp.camera_move.center_minimap)
        return FUNCTIONS.move_camera(coord)
      if act_sp.HasField("unit_selection_point"):
        select_point = act_sp.unit_selection_point
        coord = point.Point.build(select_point.selection_screen_coord)
        return FUNCTIONS.select_point(select_point.type - 1, coord)
      if act_sp.HasField("unit_selection_rect"):
        select_rect = act_sp.unit_selection_rect
        #TODO(tewalds):看了一些回放后，我们应该决定是否
        #这已经足够好了。也许我们需要模拟多个动作
        #将选择矩形合并到一个更大的矩形中.
        tl = point.Point.build(select_rect.selection_screen_coord[0].p0)
        br = point.Point.build(select_rect.selection_screen_coord[0].p1)
        return FUNCTIONS.select_rect(select_rect.selection_add, tl, br)
      if act_sp.HasField("unit_command"):
        cmd = act_sp.unit_command
        queue = int(cmd.queue_command)
        if cmd.HasField("target_screen_coord"):
          coord = point.Point.build(cmd.target_screen_coord)
          return func_call_ability(cmd.ability_id, actions.cmd_screen,
                                   queue, coord)
        elif cmd.HasField("target_minimap_coord"):
          coord = point.Point.build(cmd.target_minimap_coord)
          return func_call_ability(cmd.ability_id, actions.cmd_minimap,
                                   queue, coord)
        else:
          return func_call_ability(cmd.ability_id, actions.cmd_quick, queue)

    if action.HasField("action_raw") or action.HasField("action_render"):
      raise ValueError("Unknown action:\n%s" % action)

    return FUNCTIONS.no_op()
