import win32com.client as com
import os
from ac_net_updated_exp import Shared_actor_critic as Actor_Critic_model
from helper_fns import get_neighbors, get_occupancy
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import pickle
import time

np.random.seed(seed=123)

class Buffer:
    def __init__(self,state_dim, action_dim,max_size=500):
        self.states = np.zeros((max_size,state_dim))
        self.actions = np.zeros((max_size, action_dim))
        self.rews = np.zeros((max_size,1))
        self.next_states = np.zeros((max_size, state_dim))
        self.pointer = 0
        self.max_size = max_size
    def put(self,s,a,r,next):
        self.states[self.pointer] = s
        self.actions[self.pointer] = a
        self.rews[self.pointer] = r
        self.next_states[self.pointer] = next
        self.pointer +=1
        self.pointer %= self.max_size

    def sample(self,batch_size=32):
        indexes = np.random.randint(self.max_size,size=batch_size)
        return self.states[indexes,:],self.actions[indexes,:],self.rews[indexes,:], self.next_states[indexes, :]

    def sample_on_policy(self, batch_size=16, all = False):
        if all :
            indexes = [i for i in range(self.pointer)]
            return self.states[indexes, :], self.actions[indexes, :], self.rews[indexes, :], self.next_states[indexes, :]
        indexes = np.random.randint(self.pointer, size=batch_size)
        return self.states[indexes, :], self.actions[indexes, :], self.rews[indexes, :], self.next_states[indexes, :]

    def preprocess_on_policy(self,val_next_state, disc=0.98):
        self.rews[self.pointer-1] += disc * val_next_state
        for i in range(self.pointer - 2, -1, -1):
            self.rews[i] += disc * self.rews[i + 1]


class Agent:
    def __init__(self, sim_object, id, state_size, action_size, num_signal=4,
                 seq_length=10,
                 weight_load=False,
                 test_mode=False,
                 current_phase=0,
                 distance_from_neighbors=False):
        # id is the sc_id
        self.simulation = sim_object
        self.id = id
        self.state_size = state_size
        if id in [2, 4]:
            self.state_size = 17  ##remove this, should be done from main simulation file
        self.action_size = action_size
        self.num_signal = num_signal  # number of lanes, number of signals at the junction
        self.seq_length = 10  # atleast 2, for reward calculation to work properly #seq_length #not required for non_recurrent model
        self.current_phase = current_phase  ###being changed and set from main file itself.
        self.actions = [x for x in range(20, 71, 5)]
        self.action_count = [0 for x in range(0, len(self.actions))]
        self.neighbors = get_neighbors(id, self.simulation)
        self.num_actions_taken = 1  # change this initialization to 0, update it in act method
        # self.epislon = 0.99
        # self.discount_factor = 0.90
        self.Actor_Critic = Actor_Critic_model(state_size=self.state_size,
                                               action_size=self.action_size,
                                               )
        self.exp_replay = Buffer(state_dim=self.state_size, action_dim=self.action_size,max_size=100)
        self.rewards = []  ## this is used to plot time vs reward
        self.next_signal_change_time = 0
        self.action_que = deque(list(np.random.randint(0, self.action_size, self.seq_length)))
        self.save_weight_path = os.path.join(os.path.join(os.getcwd(), "weights"), "Agent_" + str(self.id))
        self.debug_path = os.path.join(os.path.join(os.getcwd(), "debug"), "Agent_" + str(self.id),
                                       str(time.strftime("%Y%m%d-%H%M%S")))

        self.test_mode = True if test_mode == "True" else False

        if self.test_mode:
            self.Actor_Critic.load_saved_model(self.save_weight_path)
            # self.epislon = 0.05
            print("Agent " + str(self.id) + " in **TEST** mode")

        elif weight_load:
            print("loading previous saved models for ", self.id)
            self.Actor_Critic.load_saved_model(self.save_weight_path)

        self.cric_loss_list = []
        self.ac_loss_list = []
        self.prev_cost = 0
        self.min_actions = 64
        self.num_actions_taken_policy = 1 ## action count according to current policy

    def get_neighbors_occupancy(self, sim_time):
        neighbors_occupancy = []
        # print("neighbors are ", self.neighbors)
        for neigh in self.neighbors:
            neighbors_occupancy.extend(get_occupancy(neigh, self.simulation, sim_time))
        return neighbors_occupancy

    def get_current_state(self, sim_time, neighbours_dist=False):
        own_occupancy = get_occupancy(self.id, self.simulation, sim_time)
        neighbors_occupancy = self.get_neighbors_occupancy(sim_time)
        ## phase change is being done through main file itself
        #        print(neighbors_occupancy)
        if neighbours_dist:
            return own_occupancy.extend(neighbors_occupancy).extend(self.distance_from_neighbors).append(
                self.current_phase)
        else:
            #            print(own_occupancy)
            own_occupancy.extend(neighbors_occupancy)
            own_occupancy.append(self.current_phase)  # required to append phase
            #            own_occupancy.extend(neighbors_occupancy).append(self.current_phase)
            #            print("returning ",own_occupancy )
            return own_occupancy

    def select_action(self,sim_time):
        self.curr_state = np.array(self.get_current_state(sim_time)).reshape(self.state_size,)
        pie = list(self.Actor_Critic.actor_predict(self.curr_state, self.debug_path))
        # print(self.id , pie)
        if self.test_mode:
            action_index = np.argmax(pie)
        else:
            # np.random.seed(123)
            # print("pie is",pie, "sum is ", sum(pie))
            if sum(pie) > 1.0:
                max_val = max(pie)
                max_ind = pie.index(max_val)
                pie = [(1 - max(pie)) / (len(pie) - 1)] * len(pie)
                pie[max_ind] = max_val
            pie2 = np.random.multinomial(n=1, pvals=pie, size=1)
            action_index = np.argmax(pie2)
        """
        sum of last row of temp is current state
        action_index can give current action
        write these to a file for analysis
        """
        self.write_to_file_for_actor(np.sum(self.curr_state[:4]), action_index * 5 + 20)  ## its own occupancy only

        self.num_actions_taken += 1

        self.action_taken = np.zeros(self.action_size,)
        self.action_taken[action_index] = 1
        self.num_actions_taken_policy += 1
        return action_index

    def act_and_train_if_reqd(self, sim_time):
        self.next_st = np.array(self.get_current_state(sim_time)).reshape(self.state_size,)  ##next_st is a list
        cost = (sum(self.next_st) - self.next_st[-1])
        reward = self.prev_cost - cost
        self.prev_cost = cost
        self.exp_replay.put(self.curr_state,self.action_taken,reward,self.next_st)
        if not self.test_mode:
            # if self.num_actions_taken % self.seq_length == 0:
            # checking episode is complete or not
            ###training of model is supposed to be done here
            if self.num_actions_taken_policy > self.min_actions:
                # convert reward to value in buffer
                # its not a episodic task, so next state value is not zero
                self.exp_replay.preprocess_on_policy(val_next_state=self.Actor_Critic.critic.predict(self.next_st))
                for _ in range(4):
                    s,a,r,next = self.exp_replay.sample_on_policy()
                    ac_loss = self.Actor_Critic.actor_ppo_fit_online(s, a, r, next)
                    self.ac_loss_list.append(ac_loss)

                s, a, r, next = self.exp_replay.sample_on_policy(all=True)
                cric_loss = self.Actor_Critic.critic_fit(s,r)
                self.cric_loss_list.append(cric_loss)
                print(str(self.id) + " trained")

                self.Actor_Critic.old_actor.set_weights(self.Actor_Critic.Actor.get_weights())
                self.exp_replay = Buffer(state_dim=self.state_size, action_dim=self.action_size,max_size=100)
                self.num_actions_taken_policy = 0

        self.next_signal_change_time = self.select_action(sim_time=sim_time)  # returns index, this var is being used in main file
        self.action_que.append(self.next_signal_change_time)
        self.action_que.popleft()

        if not self.test_mode:
            if self.num_actions_taken % 20 == 0:
                self.Actor_Critic.save_weights(path=self.save_weight_path)

            if self.num_actions_taken % 25 == 0:
                self.plot_and_save_data()

    # def update_epislon(self):
    #     ### update epislon, linear any decay
    #     ## it needs number of action steps taken till now
    #     #        self.num_actions_taken
    #     #        self.epislon = 0.8 ##update epislon using number of actions taken till now from this state
    #     if self.num_actions_taken < 375:
    #         if self.num_actions_taken % 10 == 0:
    #             self.epislon = self.epislon * np.exp(-self.num_actions_taken / 6000)
    #     elif self.num_actions_taken < 1100:
    #         self.epislon = 0.22
    #     elif self.num_actions_taken < 1500:
    #         self.epislon = 0.16
    #     else:
    #         self.epislon = 0.12

    def plot_and_save_data(self):
        """
        plot and then save the data
        empty the lists if num_actions_taken is divisble by 5000
        """
        plot_save_path = os.path.join(os.path.join(os.getcwd(), "plots"), "Agent_" + str(self.id))
        if not os.path.exists(plot_save_path):
            os.makedirs(plot_save_path)
        x = [i for i in range(len(self.cric_loss_list))]
        plt.plot(x, self.cric_loss_list)
        # naming the x axis
        plt.xlabel('time_steps')
        # naming the y axis
        plt.ylabel('critic_network_loss')
        # giving a title to my graph
        plt.title('cric_loss MSE versus time ' + str(self.num_actions_taken) + " actions taken")
        plt.savefig(os.path.join(plot_save_path, str(self.num_actions_taken) + "critic.png"))
        print("critic plot saved for " + str(self.num_actions_taken) + " actions.")
        plt.close()

        x = [i for i in range(len(self.ac_loss_list))]
        plt.plot(x, self.ac_loss_list)
        # naming the x axis
        plt.xlabel('time_steps')
        # naming the y axis
        plt.ylabel('actor_network_loss')
        # giving a title to my graph
        plt.title('actor_loss versus time ' + str(self.num_actions_taken) + " actions taken")
        plt.savefig(os.path.join(plot_save_path, str(self.num_actions_taken) + "actor.png"))
        print("actor plot saved for " + str(self.num_actions_taken) + " actions.")
        plt.close()
        if self.num_actions_taken % 5000 == 0:
            data_save_path = os.path.join(os.path.join(os.getcwd(), "data"), "Agent_" + str(self.id))
            if not os.path.exists(data_save_path):
                os.makedirs(data_save_path)
            f = open(os.path.join(data_save_path, str(self.num_actions_taken) + "critic.pickle"), "wb")
            pickle.dump(self.cric_loss_list, f, pickle.HIGHEST_PROTOCOL)
            f.close()
            self.cric_loss_list = []

            f = open(os.path.join(data_save_path, str(self.num_actions_taken) + "actor.pickle"), "wb")
            pickle.dump(self.ac_loss_list, f, pickle.HIGHEST_PROTOCOL)
            f.close()
            self.ac_loss_list = []
            self.cric_loss_list = []

    def write_to_file_for_visualization(self, state_arr, cost_list, next_st):
        # print("input arr shape is",state_arr.shape)
        a = [cost_list[i] for i in range(len(cost_list))]
        predict_critic_list = self.Actor_Critic.critic_predict_ret_as_list(state_arr)
        actual_critic_list = list(self.Actor_Critic.cal_discounted_rew(np.asarray(a), next_st))
        # print("length of actual",len(actual_critic_list))
        # print("length of predicted",len(predict_critic_list))

        path = os.path.join(os.path.join(os.getcwd(), "plots"), "Agent_" + str(self.id))
        if not os.path.exists(path):
            os.makedirs(path)
        data_file = os.path.join(path, str(self.id) + "_critic.pickle")
        with open(data_file, "rb") as f:
            data = pickle.load(f)
        data[0].append(actual_critic_list[-1])
        data[1].append(predict_critic_list[-1])
        with open(data_file, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        # print("critic data saved at ", data_file)

    def write_to_file_for_actor(self, state_data, action_data):
        path = os.path.join(os.path.join(os.getcwd(), "plots"), "Agent_" + str(self.id))
        if not os.path.exists(path):
            os.makedirs(path)
        data_file = os.path.join(path, str(self.id) + "_actor.pickle")
        with open(data_file, "rb") as f:
            data = pickle.load(f)
        data[0].append(state_data)
        data[1].append(action_data)
        with open(data_file, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        # print("actor data saved at ", data_file)

    def print_status(self):
        print("sc_id =" + str(self.id) + "number of actions_taken = " + str(self.num_actions_taken))
        print("current action = ", self.next_signal_change_time * 5 + 20)
        print("\n")


if __name__ == "__main__":
    sim = 5
    Vissim = com.Dispatch("Vissim.Vissim-64.10")
    flag_read_additionally = True  # you can read network(elements) additionally, in this case set "flag_read_additionally" to true
    Filename = "C:\\Users\\vissim\\Desktop\\trial_vissim-net\\2_4juncton_network.inpx"
    Vissim.LoadNet(Filename, flag_read_additionally)
    Agents = Agent(sim, id=1, state_size=12, action_size=11, num_signal=4)
    Agents.train()

    """
    this code is to plot epislon decay
    """
    # epislon = []
    # sim = None
    # a = Agent(sim, id=1, state_size=12, action_size=11, num_signal=4)
    # t=800
    # for i in range(t):
    #     a.num_actions_taken += 1
    #     a.update_epislon()
    #     epislon.append(a.epislon)
    # plt.plot([i for i in range(t)], epislon)
    # plt.show()