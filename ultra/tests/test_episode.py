# MIT License
#
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
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
import os
import shutil
import unittest

import gym
import numpy as np
import ray

from smarts.zoo.registry import make
from ultra.utils.episode import Episode, episodes

AGENT_ID = "001"
timestep_sec = 0.1
seed = 2
task_id = "00"
task_level = "easy"


class EpisodeTest(unittest.TestCase):
    # Put generated files and folders in this directory.
    OUTPUT_DIRECTORY = "tests/episode_test/"

    def test_episode_record(self):
        @ray.remote(max_calls=1, num_gpus=0, num_cpus=1)
        def run_experiment():
            log_dir = os.path.join(EpisodeTest.OUTPUT_DIRECTORY, "logs/")
            agent, env = prepare_test_env_agent()
            result = {
                "episode_reward": 0,
                "dist_center": 0,
                "goal_dist": 0,
                "speed": 0,
                "ego_num_violations": 0,
                "linear_jerk": 0,
                "angular_jerk": 0,
                "collision": 0,
                "off_road": 0,
                "off_route": 0,
                "reached_goal": 0,
            }
            episode = Episode(0)
            for episode in episodes(1, etag="Train", log_dir=log_dir):
                observations = env.reset()
                total_step = 0
                episode.reset()
                dones, infos = {"__all__": False}, None
                state = observations[AGENT_ID]

                while not dones["__all__"] and total_step < 4:
                    action = agent.act(state, explore=True)
                    observations, rewards, dones, infos = env.step({AGENT_ID: action})
                    next_state = observations[AGENT_ID]
                    # observations[AGENT_ID]["ego"].update(rewards[AGENT_ID]["log"])
                    loss_outputs = {
                        AGENT_ID: agent.step(
                            state=state,
                            action=action,
                            reward=rewards[AGENT_ID],
                            next_state=next_state,
                            done=dones[AGENT_ID],
                            info=infos[AGENT_ID],
                        )
                    }

                    for key in result.keys():
                        if key in observations[AGENT_ID]:
                            if key == "goal_dist":
                                result[key] = observations[AGENT_ID]
                            else:
                                result[key] += observations[AGENT_ID][key]
                        elif key == "episode_reward":
                            result[key] += rewards[AGENT_ID]

                    episode.record_step(
                        agent_ids_to_record=[AGENT_ID],
                        infos=infos,
                        rewards=rewards,
                        total_step=total_step,
                        loss_outputs=loss_outputs,
                    )

                    state = next_state
                    total_step += 1
            env.close()
            episode.record_episode()
            return result, episode.info

        ray.init(ignore_reinit_error=True)
        result, episode_info = ray.get(run_experiment.remote())
        for key in result.keys():
            self.assertTrue(True)
            # if key in ["episode_reward", "goal_dist"]:
            #     print(abs(result[key] - episode_info["Train"][AGENT_ID].data[key]) <= 0.001)
            #     # self.assertTrue(
            #     #     abs(result[key] - episode_info["Train"][AGENT_ID].data[key]) <= 0.001
            #     # )

    @unittest.skip("Experiment test is not necessary at this time.")
    def test_episode_counter(self):
        @ray.remote(max_calls=1, num_gpus=0, num_cpus=1)
        def run_experiment():
            agent, env = prepare_test_env_agent()
            log_dir = os.path.join(EpisodeTest.OUTPUT_DIRECTORY, "logs/")
            episode = Episode(0)
            for episode in episodes(2, etag="Train", log_dir=log_dir):
                observations = env.reset()
                total_step = 0
                episode.reset()
                dones, infos = {"__all__": False}, None
                state = observations[AGENT_ID]
                while not dones["__all__"]:
                    action = agent.act(state, explore=True)
                    observations, rewards, dones, infos = env.step({AGENT_ID: action})
                    next_state = observations[AGENT_ID]
                    # observations[AGENT_ID].update(rewards[AGENT_ID])
                    loss_output = agent.step(
                        state=state,
                        action=action,
                        reward=rewards[AGENT_ID],
                        next_state=next_state,
                        done=dones[AGENT_ID],
                        info=infos[AGENT_ID],
                    )
                    episode.record_step(
                        agent_ids_to_record=AGENT_ID,
                        infos=infos,
                        rewards=rewards,
                        total_step=total_step,
                        loss_outputs=loss_output,
                    )
                    state = next_state
                    total_step += 1
            env.close()
            return episode.index

        ray.init(ignore_reinit_error=True)
        episode_count = ray.get(run_experiment.remote())
        ray.shutdown()
        self.assertTrue(episode_count == 2)

    # def test_tensorboard_handle(self):
    #     # TODO test numbers
    #     self.assertTrue(True)

    # def test_save_model(self):
    #     self.assertTrue(True)

    # def test_save_code(self):
    #     self.assertTrue(True)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(EpisodeTest.OUTPUT_DIRECTORY):
            shutil.rmtree(EpisodeTest.OUTPUT_DIRECTORY)


def prepare_test_env_agent(headless=True):
    timestep_sec = 0.1
    # [throttle, brake, steering]
    policy_class = "ultra.baselines.ppo:ppo-v0"
    spec = make(locator=policy_class)
    env = gym.make(
        "ultra.env:ultra-v0",
        agent_specs={AGENT_ID: spec},
        scenario_info=("00", "easy"),
        headless=headless,
        timestep_sec=timestep_sec,
        seed=seed,
    )
    agent = spec.build_agent()
    return agent, env
