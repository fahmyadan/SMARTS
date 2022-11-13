import traci
import sys, os

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare env var SUMO_HOME")

# class SumoInterface():
#     def __int__(self):
#         self.sumo_binary = ""
#         self.edge_NS_1 =
#         self.edge_NS_1 =