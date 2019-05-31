import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import numpy as np
import os,time
from loss import proximal_policy_optimization_loss
import copy

class Shared_actor_critic:
    def __init__(self, state_size=11, action_size=20):
        self.state_size = state_size
        self.action_size = action_size
        # self.seq_len = seq_len
        self.state_shape = (-1, state_size)
        # self.reward_shape = (seq_len, 1)
        self.action_shape = (-1, action_size)
        self.Actor = self.build_ppo_actor()
        self.critic = self.build_critic()
        self.old_actor = self.build_ppo_actor()
        self.old_actor.set_weights(self.Actor.get_weights())
        self.advantage_discount = 0.75

    def build_ppo_actor(self):
        HIDDEN_SIZE = 64
        NUM_LAYERS = 2
        LR = 0.0001
        state_input = Input(shape=(self.state_size,))
        advantage = Input(shape=(1,))
        old_prediction = Input(shape=(self.action_size,))

        x = Dense(HIDDEN_SIZE, activation='tanh')(state_input)
        for _ in range(NUM_LAYERS - 1):
            x = Dense(HIDDEN_SIZE, activation='tanh')(x)

        out_actions = Dense(self.action_size, activation='softmax', name='output')(x)

        model = Model(inputs=[state_input, advantage, old_prediction], outputs=[out_actions])
        model.compile(optimizer=Adam(lr=LR),
                      loss=[proximal_policy_optimization_loss(
                      advantage=advantage,
                      old_prediction=old_prediction)])
        print("actor model summary")
        # model.summary()
        return model

    def build_critic(self):
        HIDDEN_SIZE = 64
        NUM_LAYERS = 2
        LR = 0.0004
        state_input = Input(shape=(self.state_size,))
        x = Dense(HIDDEN_SIZE, activation='tanh')(state_input)
        for _ in range(NUM_LAYERS - 1):
            x = Dense(HIDDEN_SIZE, activation='tanh')(x)
        out_value = Dense(1)(x)
        model = Model(inputs=[state_input], outputs=[out_value])
        model.compile(optimizer=Adam(lr=LR), loss='mse')
        print("critic model summary")
        # model.summary()
        return model

    def cal_discounted_rew_exp(self, rew, next_st, disc_factor=0.98):
        for i in range(rew.shape[0]):
            rew[i][0] += disc_factor * self.critic.predict(next_st[i].reshape(-1,self.state_size))
        return rew

    def cal_advantage(self, reward_array, state_array, disc_factor=0.9, gae=False):
        values_of_states = self.critic.predict(state_array)
        # values_of_states = values_of_states.reshape((values_of_states.shape[0],))
        advantage_array = reward_array - values_of_states
        return advantage_array
        # if not gae:
        #     return advantage_array
        # else:
        #     advantage_array = self.cal_discounted_rew_exp(advantage_array, disc_factor=self.advantage_discount)
        #     return advantage_array

    def critic_fit(self,state_array,val_array):
        critic_history = self.critic.fit(state_array, val_array,
                                         epochs=1, batch_size=state_array.shape[0]//4,
                                         shuffle=True)
        return critic_history.history["loss"]

    def actor_ppo_fit_online(self, state_array, action_array, val_array):
        self.state_array = state_array
        self.action_array = action_array
        self.advantage = val_array - self.critic.predict(self.state_array)

        dummy_advantage = np.zeros((self.state_array.shape[0],))
        dummy_pred = np.zeros(self.action_array.shape)
        old_pred = self.old_actor.predict([self.state_array,dummy_advantage, dummy_pred])

        act_history = self.Actor.fit([self.state_array,self.advantage,old_pred], [self.action_array],
                                     epochs=1,
                                     batch_size=self.state_array.shape[0], shuffle=True)
        # self.old_actor.set_weights(self.Actor.get_weights()) ### is being set from agent file
        return act_history.history["loss"]

    # def ppo_fit_exp(self, state_array, action_array, rew_array, next_st_arr, done=False, lr_critic=0.0001, lr_actor=0.0001):
    #     # assert rew_array.ndim == 1
    #     self.state_array = state_array
    #     self.action_array = action_array
    #     rew_array_for_critic = self.cal_discounted_rew_exp(rew_array, next_st_arr)
    #     #         if self.cr_lr != lr_critic:
    #     #             self.cr_lr = lr_critic
    #     #             opt = tf.train.AdamOptimizer(learning_rate=self.cr_lr)
    #     #             self.critic.compile(loss='mse', optimizer=opt)
    #     critic_history = self.critic.fit(self.state_array, rew_array_for_critic,
    #                                      epochs=1, batch_size=32,
    #                                      shuffle=True)
    #
    #     self.advantage = self.cal_advantage(rew_array_for_critic, self.state_array)
    #     # opt2 = tf.train.AdamOptimizer(learning_rate=lr_actor)
    #     # old_weights = self.old_actor.get_weights()
    #     # copy_weights = self.Actor.get_weights()
    #     print("action_array shape is",self.action_array.shape)
    #     # self.Actor.compile(loss=self.ppo_loss, optimizer=opt2)
    #     dummy_advantage = np.zeros((self.state_array.shape[0],))
    #     dummy_pred = np.zeros(self.action_array.shape)
    #     old_pred = self.old_actor.predict([self.state_array,dummy_advantage, dummy_pred])
    #     act_history = self.Actor.fit([self.state_array,self.advantage,old_pred], [self.action_array], epochs=1, batch_size=1, shuffle=True)
    #     self.old_actor.set_weights(self.Actor.get_weights())
    #     return critic_history.history["loss"], act_history.history["loss"]

    def critic_predict_ret_as_list(self, state_array):
        # this fn is used for debugging, visualization
        # state_array = state_array.reshape(self.state_shape)
        return list(self.critic.predict(state_array).reshape((state_array.shape[0],)))

    def actor_predict(self, state_array, debug_path):
        actor_path = os.path.join(debug_path,"actor.txt")
        if not os.path.exists(debug_path):
            os.makedirs(debug_path)

        dummy_advantage = np.zeros((state_array.shape[0],))
        dummy_pred = np.zeros((state_array.shape[0],self.action_size))
        t = self.Actor.predict([state_array.reshape(-1,self.state_size),dummy_advantage, dummy_pred])
        # t = t.reshape(-1, self.action_size)
        with open(actor_path,"a+") as f:
            f.write("state input to actor network is:" + str(state_array) + "\n")
            f.write("actor predicts" + str(t[0])+"\n")
            f.write("predicted action is "+str(np.argmax(t[0]) * 5 + 20)+ "\n")
        return t[-1, :]

    def save_weights(self, path):
        # make directory for actor and critic
        # save the weights
        actor_path = os.path.join(path, "actor")
        critic_path = os.path.join(path, "critic")
        if not os.path.exists(actor_path):
            os.makedirs(actor_path)
        if not os.path.exists(critic_path):
            os.makedirs(critic_path)
        self.Actor.save(os.path.join(actor_path, "actor.h5"))  ## save full model in hdf5 file
        self.critic.save(os.path.join(critic_path, "critic.h5")) ## save full model
        print("weights saved")

    def load_saved_model(self, path):
        actor_path = os.path.join(path, "actor")
        critic_path = os.path.join(path, "critic")
        if not os.path.exists(actor_path):
            print("actor weight_file_doesn't exist, cant load")
        else:
            try:
                self.Actor = load_model(os.path.join(actor_path, "actor.h5")) ## fails bacause of custom loss function
                print("actor model restored")
            except:
                self.Actor.load_weights(os.path.join(actor_path, "actor.h5"))
                print("actor weights loaded")
            self.Actor.summary()
            self.old_actor.set_weights(self.Actor.get_weights())
            print("old_actor weights equal to Actor")

        if not os.path.exists(critic_path):
            print("critic weight file does not exist")
        else:
            try :
                self.critic = load_model(os.path.join(critic_path, "critic.h5")) ## fails bacause of custom loss function
                print("Critic model restored")
            except :
                self.critic.load_weights(os.path.join(critic_path, "critic.h5"))
                print("critic weights loaded")
            self.critic.summary()


if __name__ == "__main__":
    next_st = np.array([1,2,3,4])
    # A_C_net.ppo_fit(state_data,act_arr,rew_arr,next_st)

    """
    checked
    passing zeros is a bad choice
    """
    # print("returned vals are",A_C_net.actor_predict(in_data2))
    '''A_C_net = Shared_actor_critic(10, 5, 4)
    x = np.asarray([1, 2, 3])
    print(A_C_net.cal_discounted_rew(x))'''