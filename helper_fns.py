import numpy as np
import matplotlib.pyplot as plt
import os
import win32com.client as com
import pickle
# def ask_vissim(sc_id,sg_id,phase_to_be_set, color, time,sim_object):
#         # sg_id is zero based
#         # sc_id is 1 based
#         # for time, set this phase to this color
#         # eat up this much of time
#         # cant eat up the time, simulation will stop
#         sc=Vissim.Net.SignalControllers.ItemByKey(sc_id)
#         sgs = sc.sgs
#         sgs[sg_id].SetAttValue("SigState", "RED")
#         minimum = 6
#         for i in range(minimum):
#             sim_object.Simulation.RunSingleStep()
#         pass

def clear_previous_written_files(num_agent = 6):
    path = os.path.join(os.getcwd(), "plots")
    def clear_pickle(path):
        data = [[0], [0]]
        for p in path:
            with open(p, "wb") as f :
                pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    for i in range(1,num_agent+1):
        path1 = os.path.join(os.path.join(path,"Agent_" + str(i)),
                                     str(i)+"_critic.pickle")
        clear_pickle([path1])
        path1 = os.path.join(os.path.join(path, "Agent_" + str(i)),
                             str(i) + "_actor.pickle")
        clear_pickle([path1])



def sc_sg_to_green(sc, sg_id, Vissim):
    sgs = sc.SGs
    for i in range(len(sgs)):
        if i == sg_id:
            sgs[i].SetAttValue("SigState", "GREEN")
            # Vissim.Simulation.RunSingleStep()
        else:
            sgs[i].SetAttValue("SigState", "RED")
            # Vissim.Simulation.RunSingleStep()

def load_network(NetFileInpx = "C:\\Users\\vissim\\Desktop\\trial_vissim-net\\6jun_4may.inpx",
                 LayoutFileLayx = "C:\\Users\\vissim\\Desktop\\trial_vissim-net\\6jun_4may.layx"):
    Vissim = com.Dispatch("Vissim.Vissim-64.10")
    flag_read_additionally = False  # you can read network(elements) additionally, in this case set "flag_read_additionally" to true
    Vissim.LoadNet(NetFileInpx, flag_read_additionally)
    Vissim.LoadLayout(LayoutFileLayx)
    return Vissim

def get_qcounter_for_signal_controller(sc_id, sim_object):
    """
    currently it is hard coded
    hard code the q_counters for every sc_id
    """
    #key in the dict is signal_controller id
    sc_que_counter_dict = {
                            2:[(9,10),(11,12),(13,14),(15,16)],
                            3:[(17,18),(19,20),(21,22),(23,24)],
                            4:[(25,26),(27,28),(29,30),(31,32)],
                            1: [(1, 2), (3, 4), (5, 6), (7, 8)],
                            5:[(33,34),(35,36),(37,38),(39,40)],
                            6:[(41,42),(43,44),(45,46),(47,48)]
                           }

    return sc_que_counter_dict

def get_neighbors(agent_id,sim_object):
    """
    hard code agent neighbours here
    """
    if agent_id == 1:
        return [2,3]
    elif agent_id == 2:
        return [1,4,5]
    elif agent_id == 3:
        return [1,4]
    elif agent_id == 4:
        return [3,2,6]
    elif agent_id == 5:
        return [2,6]
    elif agent_id == 6:
        return [4,5]
    else:
        print("agent_id should be (1,2,3,..,6)")
        print("agent_id is",agent_id)

def get_occupancy(agent_id,simulation_object,time,truncate=True):
    ##use vissim object to get occupancy
    ##return occupancy as a list
    # simulation object is the com object
    """
    use com object
    com.get_occupancy(get_sc_id(agent-id))
    """
    q_dict = get_qcounter_for_signal_controller(agent_id,simulation_object)
    ques = q_dict[agent_id]
    occupancy=[]
    for i in range(len(ques)):
        q1 = simulation_object.Net.QueueCounters.ItemByKey(ques[i][0])
        q2 = simulation_object.Net.QueueCounters.ItemByKey(ques[i][1])
        q_len1 = q1.AttValue("QLen(Current," + str(time) + ")")
        q_len2 = q2.AttValue("QLen(Current," + str(time) + ")")
        m = max(q_len1,q_len2)/200 ## bringing all numbers in a particular range
        if truncate:
            occupancy.append(np.around(m, decimals=2))
        else:
            occupancy.append(m)
    return occupancy

def get_relative_occupancy(agent_id,simulation_object,max_len=150):
    abs_occupancy = get_occupancy(agent_id,simulation_object)
    abs_occupancy /= max_len
    """
    write code to round each element to one decimal point,
    or as needed
    """
    return abs_occupancy

def plot(xaxis,yaxis,xlabel,ylabel,path_to_save,name,t):
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    plt.plot(xaxis,yaxis)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(os.path.join(path_to_save,str(t)+str(name)+".png"))
    plt.close()