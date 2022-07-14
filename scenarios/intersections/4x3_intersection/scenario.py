import os
import random
from pathlib import Path

from smarts.sstudio.genscenario import gen_scenario
from smarts.core import seed
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
    SocialAgentActor, Distribution

)
import sumolib
seed(42)

"""net = sumolib.net.readNet('map.net.xml')
edges_list = net.getEdges()
all_edges = []
south_edges =[]
north_edges=[]
west_edges=[]
east_edges=[]

for j in edges_list:
    edge_id = repr(j)[10:20]
    all_edges.append(edge_id)

for i in all_edges:

    if 'S' in i[5]:
        south_edges.append(i)
    if 'N' in i[5]:
        north_edges.append(i)
    if 'E' in i[5]:
        east_edges.append(i)
    if 'W' in i[5]:
        west_edges.append(i)

north_entry =[]
south_entry =[]
east_entry =[]
west_entry =[]

for i in north_edges:
    if '3' in i[9]:
        north_entry.append(i)
for i in south_edges:
    if '0' in i[9]:
        south_entry.append(i)
for i in east_edges:
    if '4' in i[8]:
        east_entry.append(i)
for i in west_edges:
    if '0' in i[8]:
        west_entry.append(i)

all_entry = south_entry+west_entry+east_entry+north_entry
print(all_entry)

south_exit = [i[0:9] + '3' for i in south_entry]
west_exit = [i[0:8]+ '4' + i[9] for i in west_entry]
east_exit = [i[0:8] + '0' + i[9] for i in east_entry]
north_exit = [i[0:9] + '0' for i in north_entry]

all_exit = south_exit + west_exit + east_exit + north_exit
"""
NUM_TRAFFIC_FLOWS = 100
N_AGENTS = 5

ego_missions = [
    Mission(
        route=RandomRoute()
    ) for _ in range(N_AGENTS)]


traffic = Traffic(
    flows=[
            Flow(
                route=RandomRoute(),
                rate=60 * 2,

                actors={
                        TrafficActor(
                            name="car",
                            vehicle_type=random.choices(
                                [
                                    "passenger",
                                    "bus",


                                ],
                                weights=[5, 1],
                                k=1,
                            )[0],
                        ): 1
                    },
            )
            for _ in range(NUM_TRAFFIC_FLOWS)
    ])

laner_actor = SocialAgentActor(
    name="keep-lane-agent",
    agent_locator="zoo.policies:keep-lane-agent-v0",
)

gen_scenario(
    Scenario(
        traffic={"basic": traffic},
        ego_missions=ego_missions,
        social_agent_missions={
            "all": ([laner_actor], [Mission(route=RandomRoute())])
        },

    ),
    output_dir=Path(__file__).parent,
)

