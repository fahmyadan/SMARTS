import os
from pathlib import Path

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
)

ego_missions = [
    Mission(
        route=Route(begin=('edge-south-SN', 0, 5), end=('edge-west-EW', 0, 'max')), 

    ), 
    Mission(
        route=Route(begin=('edge-north-NS', 0, 0), end=('edge-east-WE', 0, 'max'))
    )]

scenario = Scenario(
    traffic={
        "basic": Traffic(
            flows=[
                Flow(
                    route=Route(begin=('edge-south-SN', 0 , i), end=('edge-north-SN', 0, 'max')),
                    rate=3600,
                    actors={TrafficActor(name="car"): 1.0},
                )
            for i in range(10, 50,3)]
        )
    },
    ego_missions=ego_missions,
)

gen_scenario(
    scenario=scenario,
    output_dir=Path(__file__).parent,
)
