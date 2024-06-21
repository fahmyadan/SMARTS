from smarts.core.utils.sumo import traci

traci.init(port=45761, numRetries=5, label="data_retrieval")
conn = traci.getConnection("data_retrieval")
conn.setOrder(1)

while True:
    print("check", traci.vehicle.getIDList())
    traci.simulationStep()