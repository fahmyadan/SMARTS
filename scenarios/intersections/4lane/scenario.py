import os
from pathlib import Path

from numpy import require

from smarts.sstudio.genscenario import gen_scenario
from smarts.sstudio.types import (
    EndlessMission,
    Flow,
    JunctionEdgeIDResolver,
    Mission,
    RandomRoute,
    Route,
    Scenario,
    Traffic,
    TrafficActor,
    Via,
    SocialAgentActor,
)

ego_missions = [
    Mission(
        route=Route(begin=("edge-south-SN", 1, 10), end=("edge-west-EW", 1, "max")),
        via=(Via("edge-south-SN",lane_index=1, lane_offset=50, required_speed=5.0),
             Via("edge-south-SN",lane_index=1, lane_offset=2, required_speed=1.0),
             Via("edge-west-EW",lane_index=1, lane_offset=100, required_speed=10.0)), 
    ),
    Mission(route=Route(begin=("edge-north-NS", 0, 10), end=(("edge-south-NS", 0,'max'))),
            via=(Via("edge-north-NS", lane_index=0, lane_offset=50, required_speed=5.0), 
                 Via("edge-north-NS", lane_index=0, lane_offset=2, required_speed=1.0),
                 Via("edge-south-NS", lane_index=0,lane_offset=100, required_speed=10.0))),

    Mission(route=Route(begin=("edge-west-WE", 0, 10), end=(("edge-north-SN", 0,'max'))),
            via=(Via("edge-west-WE", lane_index=0, lane_offset=50, required_speed=5.0), 
                 Via("edge-west-WE", lane_index=0, lane_offset=2, required_speed=1.0),
                 Via("edge-north-SN", lane_index=0,lane_offset=100, required_speed=10.0))),

    Mission(route=Route(begin=("edge-east-EW", 0, 10), end=(("edge-west-EW", 0,'max'))),
            via=(Via("edge-east-EW", lane_index=0, lane_offset=50, required_speed=5.0), 
                 Via("edge-east-EW", lane_index=0, lane_offset=2, required_speed=1.0),
                 Via("edge-west-EW", lane_index=0,lane_offset=100, required_speed=10.0))),
    
]

# zoo_a2c_agent_actor = SocialAgentActor(
#     name="zoo-a2c-agent-actor",
#     agent_locator="a2c_agent:a2c-agent-v0",
# )

# social_agent_missions = {
#     "a2c-key": ( [ zoo_a2c_agent_actor,
#         ],
#         [
#             Mission(
#                 Route(begin=("edge-south-SN", 1, 30), end=("edge-west-EW", 1, "max"))
#             )
#         ],
#     ),
# }

scenario = Scenario(
    traffic={
        "S2N": Traffic(
            flows=[
                Flow(
                    route=Route(begin=("edge-south-SN", 1, 20), end=("edge-north-SN", 1, "max")),
                    rate=2000,
                    actors={TrafficActor(name="car"): 1.0},
                ),
                Flow(
                    route=Route(begin=("edge-south-SN", 0, 20), end=("edge-east-WE", 1, "max")),
                    rate=2000,
                    actors={TrafficActor(name="car"): 1.0},
                ),

                Flow(
                route=Route(begin=("edge-south-SN", 0, 0), end=("edge-west-EW", 1, "max")),
                rate=2000,
                actors={TrafficActor(name="car"): 1.0},
            ),
                Flow(
                route=Route(begin=("edge-north-NS", 0, 0), end=("edge-west-EW", 0, "max")),
                rate=2000,
                actors={TrafficActor(name="car"): 1.0},
            ),
                Flow(
            route=Route(begin=("edge-north-NS", 1, 0), end=("edge-south-NS", 1, "max")),
            rate=2000,
            actors={TrafficActor(name="car"): 1.0},
        )
        ]
        ),
    
    },
    ego_missions=ego_missions,
    # social_agent_missions= social_agent_missions
)

gen_scenario(
    scenario=scenario,
    output_dir=Path(__file__).parent,
)
