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
"""Python 强化学习环境API."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections

import enum
import six


class TimeStep(collections.namedtuple(
    'TimeStep', ['step_type', 'reward', 'discount', 'observation'])):
  """每次调用环境上的“step”和“reset”时返回.

  “时间步长”包含环境在…的每一步发出的数据
  交互。“TimeStep”包含“step_type”、“observation”和“an”
  相关的“奖励”和“折扣”.

  序列中的第一个“TimeStep”将具有“StepType.FIRST”。最后一个
  ' TimeStep '将包含' StepType.LAST '。所有其他的时间步都是按顺序排列的
  “StepType.MID.

  Attributes:
    step_type: 一个“StepType”枚举值.
    reward: 标量，如果' step_type '是' StepType '，则为0。’，即在
    序列的开始
    discount: 折扣值的范围为'[0,1]'，如果' step_type '则为0
    是“StepType。FIRST '，即在序列的开始。
    观察:一个数字数组，或数组的dict、列表或元组.
  """
  __slots__ = ()

  def first(self):
    return self.step_type is StepType.FIRST

  def mid(self):
    return self.step_type is StepType.MID

  def last(self):
    return self.step_type is StepType.LAST


class StepType(enum.IntEnum):
  """定义序列中“时间步长”的状态."""
  # 表示序列中的第一个“时间步长”.
  FIRST = 0
  # 表示序列中不是第一个或最后一个的任何“时间步长”.
  MID = 1
  # 表示序列中的最后一个“时间步长”.
  LAST = 2


@six.add_metaclass(abc.ABCMeta)
class Base(object):  # pytype:禁用= ignored-abstractmethod
  """Python RL环境的抽象基类."""

  @abc.abstractmethod
  def reset(self):
    """启动一个新序列并返回该序列的第一个“时间步长”.

    Returns:
      包含一个名为“TimeStep”的元组:
        step_type:' FIRST '的' StepType '.
        reward: 0.
        discount: 0.
        observation: 一个数字数组，或数组的dict、列表或元组
        对应于“observation_spec ()“.
    """

  @abc.abstractmethod
  def step(self, action):
    """根据操作更新环境并返回一个“TimeStep”.

    如果环境返回一个带有“StepType”的“TimeStep”。最后的
    在前面的步骤中，对“step”的调用将启动一个新的序列和“action”
    将被忽略。

    如果在环境之后调用，此方法还将启动一个新序列
    已构造，尚未调用“重新启动”。在这种情况下
    “行动”将被忽略。

    Args:
      action: 与之对应的数字数组或dict、列表或数组元组
      “action_spec ()”。

    Returns:
      包含一个名为“TimeStep”的元组:
        step_type: “StepType”值.
        reward: 在这个时候奖励.
        discount: [0,1]范围内的折扣.
        observation: 一个数字数组，或数组的dict、列表或元组
        对应于“observation_spec ()”.
    """

  @abc.abstractmethod
  def observation_spec(self):
    """定义环境提供的观察结果。

    Returns:
      规范的元组(每个代理一个)，其中每个规范都是形状的dict
      元组。
    """

  @abc.abstractmethod
  def action_spec(self):
    """定义应该提供给“step”的操作。

    Returns:
      规范的元组(每个代理一个)，其中每个规范都是
      定义动作的形状。
    """

  def close(self):
    """释放环境使用的任何资源。

    为外部流程支持的环境实现此方法。
    这种方法可以直接使用

    ```python
    env = Env(...)
    # Use env.
    env.close()
    ```

    or via a context manager

    ```python
    with Env(...) as env:
      # Use env.
    ```
    """
    pass

  def __enter__(self):
    """允许环境在with-statement上下文中使用。"""
    return self

  def __exit__(self, unused_exception_type, unused_exc_value, unused_traceback):
    """允许环境在with-statement上下文中使用。"""
    self.close()

  def __del__(self):
    self.close()

