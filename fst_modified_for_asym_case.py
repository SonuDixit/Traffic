import numpy as np
from helper_fns import plot,load_network,sc_sg_to_green,get_occupancy
import os
import argparse

def performance_eval(Vissim,delay_stop_tot,speed_avg,trav_tm_tot,stops_tot,
                     delay_stop_avg, stops_avg,x_axis):
    """

    :param Vissim:
    :param delay_stop_tot:
    :param speed_avg:
    :param trav_tm_tot:
    :param stops_tot:
    :param delay_stop_avg:
    :param stops_avg:
    :param x_axis:
    :return:
    DelayStopTot=Total standstill time of all vehicles that are in the network or have already arrived
    DelayStopAvg = Average standstill time per vehicle
    SpeedAvg = Average speed [km/h] or [mph] Total distance DistTot / Total travel time TravTmTot
    """
    NetPerformance = Vissim.Net.VehicleNetworkPerformanceMeasurement
    path_to_save_eval_plot = os.path.join(os.getcwd(), "eval_plots")
    # dst = NetPerformance.AttValue("DelayStopTot(Current,1,All)")
    dsa = NetPerformance.AttValue("DelayStopAvg(Current,1,All)")
    spa = NetPerformance.AttValue("SpeedAvg(Current,1,All)")
    # ttt = NetPerformance.AttValue("TravTmTot(Current,1,All)")
    # stt = NetPerformance.AttValue("StopsTot(Current,1,All)")
    sta = NetPerformance.AttValue("StopsAvg(Current,1,All)")

    delay_stop_avg.append(dsa)
    # delay_stop_tot.append(dst)
    speed_avg.append(spa)
    # stops_tot.append(stt)
    stops_avg.append(sta)
    # trav_tm_tot.append(ttt)
    plot(x_axis,delay_stop_avg,xlabel="time",ylabel="delay_stop_avg",
         path_to_save=path_to_save_eval_plot,name="delay_stop_avg",t=x_axis[-1])
    # plot(x_axis, delay_stop_tot, xlabel="time", ylabel="delay_stop_total",
    #      path_to_save=path_to_save_eval_plot, name="delay_stop_tot", t=x_axis[-1])
    plot(x_axis, speed_avg, xlabel="time", ylabel="speed_avg",
         path_to_save=path_to_save_eval_plot, name="speed_avg", t=x_axis[-1])
    # plot(x_axis, stops_tot, xlabel="time", ylabel="stops_tot",
    #      path_to_save=path_to_save_eval_plot, name="stops_tot", t=x_axis[-1])
    plot(x_axis, stops_avg, xlabel="time", ylabel="stops_avg",
         path_to_save=path_to_save_eval_plot, name="stops_avg", t=x_axis[-1])
    # plot(x_axis, trav_tm_tot, xlabel="time", ylabel="trav_tot_time",
    #      path_to_save=path_to_save_eval_plot, name="trav_tot_time", t=x_axis[-1])

def write_to_file_for_analysis(id, time, vissim_object, action_taken):
    state = get_occupancy(id,vissim_object,time)
    state = np.sum(state) - state[-1]
    file = os.path.join(os.getcwd(),"plots","Agent_"+str(id))
    if not os.path.exists(file):
        os.makedirs(file)
    file += "fst.txt"
    with open(file,"a+") as f:
        f.write(str(state) + ", "+str(action_taken)+"\n")

if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument('-time', '--fst_time', help='time for fst',type=int,default=30, required=False)
    # parser.add_argument('-Tst', '--Test_mode', help='test_mode', required=False)
    args = vars(parser.parse_args())  ####args is now a dictionary
    fst_time = args['fst_time']

    Vissim = load_network(NetFileInpx="C:\\Users\\vissim\Desktop\\trial_vissim-net\\network_asymmetric_\\6jun_asym.inpx",
                          LayoutFileLayx="C:\\Users\\vissim\Desktop\\trial_vissim-net\\network_asymmetric_\\6jun_asym.layx")
    num_signals = 6
    signal_ids = [i+1 for i in range(num_signals)]
    """make a dictionary of signal_controller, sg, fst_time
    """
    dict_sc_sg_phase_time={1:{0:22, 1: 18, 2: 10, 3: 10},
                           2:{0:10, 1: 10, 2: 22, 3: 18},
                           3:{0:10, 1: 10, 2: 22, 3: 18},
                           4:{0:10, 1: 10, 2: 22, 3: 18},
                           5:{0:10, 1: 10, 2: 22, 3: 18},
                           6:{0:22, 1: 18, 2: 10, 3: 10}
                           }
    green_times = [0 for _ in signal_ids]
    prev_phase = [0, 1, 2, 3, 1, 2]
    actions = [dict_sc_sg_phase_time[i][prev_phase[i-1]] for i in signal_ids]

    scs = [Vissim.Net.SignalControllers.ItemByKey(i) for i in signal_ids]
    change_time = np.asarray(green_times) + np.asarray(actions)
    total_time = 100000


    for i in range(total_time):
        Vissim.Simulation.RunSingleStep()
        sim_time = Vissim.Simulation.SimulationSecond

        sc_indexes_to_change = list(np.where(change_time <= sim_time)[0])
        if sc_indexes_to_change != []:
            print("simulation time: ", sim_time)
        for index in sc_indexes_to_change:
            """
            data collection is supposed to be here
            """
            print("signal_controller " + str(index + 1) + " is being changed")
            sc_sg_to_green(scs[index], (prev_phase[index] + 1) % 4,Vissim)
            prev_phase[index] += 1
            prev_phase[index] %= 4
            Vissim.Simulation.RunSingleStep()
            change_time[index] += dict_sc_sg_phase_time[index+1][prev_phase[index]]
            write_to_file_for_analysis(id = index+1, time = sim_time,vissim_object =Vissim, action_taken= dict_sc_sg_phase_time[index+1][prev_phase[index]])
        """
        plotting evaluation parameters
        """
    print("out of for loop, completed FST simulation fst_time was ", str(fst_time))