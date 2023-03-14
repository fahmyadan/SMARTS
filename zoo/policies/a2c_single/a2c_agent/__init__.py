from .agent import LaneAgent
from smarts.zoo.registry import register
from smarts.zoo.agent_spec import AgentSpec

from .representations import observation_adapter, reward_adapter, a2c_agent_interface

def entrypoint():

    lane_rl_agent_spec = AgentSpec(
                        interface=a2c_agent_interface, 
                        agent_builder= lambda: LaneAgent(), 
                        observation_adapter=observation_adapter, 
                        reward_adapter= reward_adapter,

    )

    return lane_rl_agent_spec

register(locator= 'a2c-agent-v0', entry_point= entrypoint)