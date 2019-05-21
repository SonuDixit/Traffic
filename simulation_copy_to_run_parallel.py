from Agent_online_buffer import Agent
import numpy as np
from helper_fns import plot
# from Env import Env
from helper_fns import plot,load_network,sc_sg_to_green,clear_previous_written_files
import os
import argparse, time
import random
# random.seed(7)

def change_vehicle_input_rate(sim_object, time):
    ##change after every 3000 seconds
    d = [500, 1000, 1500, 2000, 2500, 3000, 3500]
    for k in range(1,11):
        inc_den = d[random.randint(0, len(d)-1)]
        sim_object.Net.VehicleInputs.ItemByKey(k).SetAttValue('Volume(1)', inc_den)
        with open("vehicle_incoming_density.txt","a+") as f:
            f.write(str(time)+": "+str(k)+", "+str(inc_den)+"\n")

def write_performance_file(dsa, spa, sta, start):
    path_to_save_eval_plot = os.path.join(os.getcwd(), "eval_plots", "perform_data.txt")
    if start==True:
        with open(path_to_save_eval_plot, "w") as f:
            f.write("delay_stop_avg, speedAvg, StopsAvg \n")
    with open(path_to_save_eval_plot,"a+") as f:
        f.write(str(dsa)+", "+str(spa)+", "+str(sta)+"\n")

def performance_eval(Vissim,delay_stop_tot,speed_avg,trav_tm_tot,stops_tot,
                     delay_stop_avg, stops_avg,x_axis, time):
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
    dsa = NetPerformance.AttValue("DelayStopAvg(Current,"+"Last"+",All)")
    spa = NetPerformance.AttValue("SpeedAvg(Current,"+"Last"+",All)")
    # ttt = NetPerformance.AttValue("TravTmTot(Current,1,All)")
    # stt = NetPerformance.AttValue("StopsTot(Current,1,All)")
    sta = NetPerformance.AttValue("StopsAvg(Current,"+"Last"+",All)")

    start_flag = True if time == 1 else False
    write_performance_file(dsa, spa, sta, start_flag)

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


if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument('-lw', '--load_weight', help='want to load pre-trained weights',default="False", required=False)
    parser.add_argument('-Tst', '--Test_mode',  help='test_mode_True or False', default="False", required=False)
    parser.add_argument('-tot_time', '--total_time', type=int, help = 'total_run_time', default = 100000, required=False)
    parser.add_argument('-sim_seed', '--sim_seed', type=int, help='simulation random seed', default=15, required=False)
    parser.add_argument('--rand_seed', type=int, help = 'random_seed_for_input_change', default=7, required= False)
    args = vars(parser.parse_args())  ####args is now a dictionary
    w_l = True if args["load_weight"] == "True" else False

    random.seed(int(args['rand_seed']))
    Vissim = load_network()  ## default is copy_network
    vis_r_seed = int(args['sim_seed'])
    print("vissim random seed:", vis_r_seed)
    Vissim.Simulation.SetAttValue("RandSeed", vis_r_seed)
    Vissim.Simulation.SetAttValue("SimRes", 1)
    # Activate QuickMode:
    Vissim.Graphics.CurrentNetworkWindow.SetAttValue("QuickMode", 1)
    Vissim.SuspendUpdateGUI()
    # Set maximum speed:
    Vissim.Simulation.SetAttValue('UseMaxSimSpeed', True)

    signal_ids = [1, 2, 3, 4, 5,6]
    prev_phase = [0, 0, 0, 0, 0, 0]
    green_times = [0, 0, 0, 0, 0, 0]

    clear_previous_written_files(num_agent=6)
    open("vehicle_incoming_density.txt", "w").close()

    Agents = [
        Agent(Vissim, id=signal_ids[i], state_size=13,
              action_size=11, num_signal=4,
              seq_length=10, weight_load = w_l,
              current_phase=prev_phase[i],
              test_mode=args["Test_mode"])
        for i in range(len(signal_ids))]

    ###for today, want to train only one agent, all others in test mode
    # Agents[0].test_mode = False
    # Agents[2].test_mode = False
    # Agents[3].test_mode = False
    # Agents[5].test_mode = False
    Vissim.Simulation.RunSingleStep()
    actions = [agent.select_action(sim_time=1) * 5 + 20 for agent in Agents] # intial action
    print("initial_action", actions)
    scs = [Vissim.Net.SignalControllers.ItemByKey(i) for i in signal_ids]
    change_time = np.asarray(green_times) + np.asarray(actions)
    total_time = args["total_time"]
    # 10 times here is one simulation time.
    # check simulation->simulation parameter in vissim
    # these are evaluation_parameters
    delay_stop_tot = []
    speed_avg = []
    trav_tm_tot = []
    stops_tot = []
    delay_stop_avg = []
    stops_avg = []
    x_axis = []
    change_time_file = str(time.strftime("%Y%m%d-%H%M%S")) + "change_time.txt"
    for i in range(total_time):
        Vissim.Simulation.RunSingleStep()
        sim_time = Vissim.Simulation.SimulationSecond
        # print("simulation time: ", sim_time)
        sc_indexes_to_change = list(np.where(change_time <= sim_time)[0])
        # Vissim.Simulation.RunSingleStep()
        for index in sc_indexes_to_change:
            """
            data collection is supposed to be here
            """
            print("simulation time: ", sim_time)
            print("signal_controller " + str(index + 1) + " is being changed for phase = ",
                  (prev_phase[index] + 1) % 4)
            sc_sg_to_green(scs[index], (prev_phase[index] + 1) % 4, Vissim)
            prev_phase[index] += 1
            prev_phase[index] %= 4
            Agents[index].current_phase = prev_phase[index]

            Agents[index].act_and_train_if_reqd(sim_time)
            # print("debug it", Agents[index].next_signal_change_time)
            f = open(os.path.join(os.getcwd(),"debug",change_time_file),"a+")
            # print("simulation time: " +  str(sim_time)+"\n")
            f.write("before " + str(change_time) + "\n")
            change_time[index] += Agents[index].next_signal_change_time * 5 + 20
            print("green duration is:" + str(change_time[index] - sim_time) + "\n")
            f.write("after " + str(change_time) + "\n")
            f.write("green duration is:" + str(change_time[index] - sim_time) + "\n")
            f.close()

        """
        plotting evaluation parameters
        """
        if i % 1000 == 0 and i != 0:
            print("i= ", i)
            x_axis.append(i)
            performance_eval(Vissim, delay_stop_tot, speed_avg, trav_tm_tot, stops_tot,
                             delay_stop_avg, stops_avg, x_axis,time=int(i/1000))

            if i%3000 == 0:
                change_vehicle_input_rate(Vissim,i)

    print("out of for loop, completed training, tot time was", total_time)