import numpy as np
from helper_fns import plot,load_network,sc_sg_to_green, get_occupancy
import os
import argparse
import random
# random.seed(7)

def change_vehicle_input_rate(sim_object, time):
    ##change after every 3000 seconds
    d = [500, 1000, 1500, 2000, 2500, 3000, 3500]
    for k in range(1,11):
        inc_den = d[random.randint(0, len(d)-1)]
        sim_object.Net.VehicleInputs.ItemByKey(k).SetAttValue('Volume(1)', inc_den)
        with open("vehicle_incoming_density_fst.txt","a+") as f:
            f.write(str(time)+": "+str(k)+", "+str(inc_den)+"\n")
def write_performance_file(dsa, spa, sta, start,fst_time):
    path_to_save_eval_plot = os.path.join(os.getcwd(), "eval_plots_fst",str(fst_time), "perform_data.txt")
    if start==True:
        with open(path_to_save_eval_plot, "w") as f:
            f.write("delay_stop_avg, speedAvg, StopsAvg \n")
    with open(path_to_save_eval_plot,"a+") as f:
        f.write(str(dsa)+", "+str(spa)+", "+str(sta)+"\n")

def performance_eval(Vissim,delay_stop_tot,speed_avg,trav_tm_tot,stops_tot,
                     delay_stop_avg, stops_avg,x_axis, time,fst_time):
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
    path_to_save_eval_plot = os.path.join(os.getcwd(), "eval_plots_fst",str(fst_time))
    if not os.path.exists(path_to_save_eval_plot):
        os.makedirs(path_to_save_eval_plot)
    # time = str(time)
    # dst = NetPerformance.AttValue("DelayStopTot(Current,1,All)")
    dsa = NetPerformance.AttValue("DelayStopAvg(Current,"+"Last"+",All)")
    spa = NetPerformance.AttValue("SpeedAvg(Current,"+"Last"+",All)")
    # ttt = NetPerformance.AttValue("TravTmTot(Current,1,All)")
    # stt = NetPerformance.AttValue("StopsTot(Current,1,All)")
    sta = NetPerformance.AttValue("StopsAvg(Current,"+"Last"+",All)")

    start_flag = True if time == 1 else False
    write_performance_file(dsa, spa, sta, start_flag,fst_time)

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
def clear_prev_data(num_signals=6,fst_time=30):
    path = os.path.join(os.getcwd(), "plots")
    fst_time = str(fst_time)
    for i in range(1, num_signals + 1):
        path1 = os.path.join(os.path.join(path, "Agent_" + str(i)), fst_time+"fst.txt")
        open(path1,"w").close()
        path1 = os.path.join(os.path.join(path, "Agent_" + str(i)), fst_time+"fst_state.txt")
        open(path1, "w").close()

def write_to_file_for_analysis(id, time, vissim_object, action_taken,fst_time):
    fst_time = str(fst_time)
    state = get_occupancy(id, vissim_object, time)
    s = np.sum(state)
    file = os.path.join(os.getcwd(), "plots", "Agent_" + str(id))
    if not os.path.exists(file):
        os.makedirs(file)
    with open(os.path.join(file,fst_time+"fst_state.txt"),"a+") as f:
        f.write(str(time)+" :: "+str(state) + "\n")
    file =os.path.join(file,fst_time+"fst.txt")
    with open(file, "a+") as f:
        f.write(str(s) + ", " + str(action_taken) + "\n")


if __name__ == "__main__" :
    parser = argparse.ArgumentParser()

    parser.add_argument('-total_time', type=int, help='total_run_time', default=100000, required=False)
    parser.add_argument('-sim_seed', type=int, help='simulation random seed', default=15, required=False)
    parser.add_argument('-rand_seed', type=int, help='random_seed_for_input_change', default=7, required=False)
    parser.add_argument('-fst_time', help='time for fst',type=int,default=20, required=False)

    args = vars(parser.parse_args())  ####args is now a dictionary
    fst_time = int(args['fst_time'])

    # Vissim = load_network()## default is copy_network
    random.seed(int(args['rand_seed']))
    Vissim = load_network(NetFileInpx="C:\\Users\\vissim\\Desktop\\trial_vissim-net\\6jun_4may.inpx",
                          LayoutFileLayx="C:\\Users\\vissim\\Desktop\\trial_vissim-net\\6jun_4may.layx")
    print("vissim random seed:", args['sim_seed'])

    Vissim.Simulation.SetAttValue("RandSeed",int(args['sim_seed']))
    # Activate QuickMode:
    Vissim.Graphics.CurrentNetworkWindow.SetAttValue("QuickMode", 1)
    Vissim.SuspendUpdateGUI()
    # Set maximum speed:
    Vissim.Simulation.SetAttValue('UseMaxSimSpeed', True)
    Vissim.Simulation.SetAttValue("SimRes", 1)

    num_signals = 6
    clear_prev_data(num_signals, fst_time)
    open("vehicle_incoming_density_fst.txt", "w").close()

    signal_ids = [i+1 for i in range(num_signals)]
    actions = [fst_time for _ in signal_ids]
    green_times = [0 for _ in signal_ids]
    prev_phase = [0 for _ in signal_ids]
    scs = [Vissim.Net.SignalControllers.ItemByKey(i) for i in signal_ids]
    change_time = np.asarray(green_times) + np.asarray(actions)

    delay_stop_tot = []
    speed_avg = []
    trav_tm_tot = []
    stops_tot = []
    delay_stop_avg = []
    stops_avg = []
    x_axis = []

    total_time = int(args['total_time'])

    for i in range(total_time):
        Vissim.Simulation.RunSingleStep()
        sim_time = Vissim.Simulation.SimulationSecond
        sc_indexes_to_change = list(np.where(change_time <= sim_time)[0])
        if len(sc_indexes_to_change) > 0:
            print("simulation time: ", sim_time)
        for index in sc_indexes_to_change:
            # print("signal_controller " + str(index + 1) + " is being changed")
            sc_sg_to_green(scs[index], (prev_phase[index] + 1) % 4,Vissim)
            prev_phase[index] += 1
            prev_phase[index] %= 4
            # Vissim.Simulation.RunSingleStep()
            change_time[index] += fst_time
            write_to_file_for_analysis(id=index + 1, time=sim_time, vissim_object = Vissim,
                                       action_taken=fst_time, fst_time=fst_time)

        if i % 1000 == 0 and i != 0:
            print("i= ", i)
            x_axis.append(i)
            performance_eval(Vissim, delay_stop_tot, speed_avg, trav_tm_tot, stops_tot,
                             delay_stop_avg, stops_avg, x_axis,time=int(i/1000),fst_time=fst_time)
            if i % 3000 == 0:
                change_vehicle_input_rate(Vissim,i)
    print("out of for loop, completed FST simulation fst_time was ", str(fst_time))