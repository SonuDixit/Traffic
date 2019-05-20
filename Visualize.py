import matplotlib.pyplot as plt
import pickle
class Visualize:
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
        print("true values are:")
        print(data[0])
        print("predicted values are:")
        print(data[1])

        x = [ i for i in range(len(data[0]))]
        x2 = [i for i in range(len(data[1]))]
        plt.figure(figsize=fig_size)
        plt.plot(x,data[0],'r', label="True Value")
        plt.plot(x2,data[1],'b', label="Predicted")
        plt.legend()
        plt.show()
