# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
import gym
import numpy as np

from smarts.contrib.pymarl import PyMARLHiWayEnv


class ListHiWayEnv(PyMARLHiWayEnv):
    """A specialized PyMARLHiWayEnv environment that provides all information as arrays. """

    metadata = {"render.modes": ["human"]}

    def __init__(self, config):
        super(ListHiWayEnv, self).__init__(config)

    def step(self, agent_actions):
        """ Returns observations, rewards, dones, infos. """
        agent_actions = np.array(agent_actions)
        _, _, infos = super().step(agent_actions)
        n_rewards = infos.pop("rewards_list")
        n_dones = infos.pop("dones_list")
        return (
            np.asarray(self.get_obs()),
            np.asarray(n_rewards),
            np.asarray(n_dones),
            infos,
        )
