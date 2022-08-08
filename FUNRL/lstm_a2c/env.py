import gym
import torch
import numpy as np
from copy import deepcopy
from utils import pre_process
from torch.multiprocessing import Process

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EnvWorker(Process):
    def __init__(self, env_name, render, child_conn, args, agent_spec):
        super(EnvWorker, self).__init__()
        #self.env = gym.make(env_name)
        self.env = gym.make(
            "smarts.env:hiway-v0",
            scenarios=args.scenarios,
            agent_specs={"SingleAgent": agent_spec},
            headless=args.headless,
            sumo_headless=True,


        )
        self.render = render
        self.child_conn = child_conn
        self.init_state()

    def init_state(self):
        state = self.env.reset()
        
        state, _, _, _ = self.env.step({'SingleAgent': [0,0,0]})
        #state = pre_process(state)
        self.history = state
        #self.history = np.moveaxis(state, -1, 0)

    def run(self):
        super(EnvWorker, self).run()

        episode = 0
        steps = 0
        score = 0
        collisions = 1 # -> Threshold for collisions; if veh has crashed once crashed ==True
        crashed = False

        while True:
            if self.render:
                self.env.render()

            action = self.child_conn.recv()
            next_state, reward, done, info = self.env.step({'SingleAgent':action + 1}) #HiwayEnv requires input to step function as dict
            #env.step({'SingleAgent':action})[3]['SingleAgent']['env_obs'].events.collisions-> info (collisions) of step function
            if collisions <=  len(info['SingleAgent']['env_obs'].events.collisions): #if length of list of collisions >1 crashed is true
                crashed = True
                collisions = info['SingleAgent']['env_obs'].events.collisions

            #next_state = pre_process(next_state)
            #self.history = np.moveaxis(next_state, -1, 0)

            steps += 1
            score += reward

            self.child_conn.send([deepcopy(self.history), reward, crashed, done])

            if done and crashed:
                # print('{} episode | score: {:2f} | steps: {}'.format(
                #     episode, score, steps
                # ))
                episode += 1
                steps = 0
                score = 0
                crashed = False
                collisions = 5
                self.init_state()

            if crashed:
                crashed = False
                self.init_state()
