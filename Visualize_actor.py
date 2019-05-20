import matplotlib.pyplot as plt
import pickle
class Visualize_actor:
    def __init__(self,filename, time_to_refresh = 20):
        self.datafile = filename
        self.time_to_refresh = time_to_refresh
        pass
    def draw(self,fig_size=(8,8)):
        """
        plot the data from filename
        refresh after every time_to_refresh
        :return:
        """
        with open(self.datafile,"rb") as f:
            data = pickle.load(f)
        # considering data as a list of 2 lists, True and predicted
        # print(type(data))
        # print(data[0])
        print("total_num_vals = ", len(data[0]))
        print("state values are:")

        print(data[0])
        print("actions are:")
        print(data[1])

        data[1] = [-x/10 for x in data[1]]  ##actions
        # data[0] = [x / 1000 for x in data[0]]

        x = [ i for i in range(len(data[0]))]
        x2 = [i for i in range(len(data[1]))]
        plt.figure(figsize=fig_size)
        # plt.bar(x,data[0]/1000,'r', label="True Value")
        plt.ylabel("action_and_sum_state_occupancy")
        plt.bar(x, data[0])
        plt.bar(x2, data[1])
        # plt.bar(x2,data[1],'b', label="Predicted")
        plt.legend()
        plt.show()