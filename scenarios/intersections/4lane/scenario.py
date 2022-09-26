import os
from pathlib import Path

from smarts.sstudio.genscenario import gen_scenario
from smarts.sstudio.types import (
    PositionalZone,
    SocialAgentActor,
    Bubble,
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
)

#ego_missions = [
#    Mission(
#        route=RandomRoute()
#    )]
traffic_actor = TrafficActor(name="bus", vehicle_type='bus')
social_actor = SocialAgentActor(name='social_keep_lane',agent_locator="zoo.policies:keep-lane-agent-v0" )

bubble_agent = Bubble(zone=PositionalZone(pos=(0, 0), size=(50, 50)), actor= social_actor , margin= 2)

scenario = Scenario(
    traffic={
        "basic": Traffic(
            flows=[
                Flow(
                    route=RandomRoute(),
                    rate=3600,
                    actors={traffic_actor: 1.0},
                )
            for i in range(10)],

        )
    },
    bubbles= [bubble_agent],
    social_agent_missions= {'all_social': ([social_actor], [Mission(route=RandomRoute())])}
    #ego_missions=ego_missions,
)

gen_scenario(
    scenario=scenario,
    output_dir=Path(__file__).parent,
)
