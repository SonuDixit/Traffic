import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import numpy as np
import os,time
from loss import proximal_policy_optimization_loss, actor_critic_loss
import copy

class actor_critic:
    def __init__(self, state_size=11, action_size=20 ):
        self.state_size = state_size
        self.action_size = action_size
        self.state_shape = (-1, state_size)
        self.action_shape = (-1, action_size)
        self.Actor = self.build_actor()
        self.critic = self.build_critic()
    def build_actor(self):
        HIDDEN_SIZE = 64
        NUM_LAYERS = 2
        LR = 0.0001
        state_input = Input(shape=(self.state_size,))
        value = Input(shape=(1,))

        x = Dense(HIDDEN_SIZE, activation='tanh')(state_input)
        for _ in range(NUM_LAYERS - 1):
            x = Dense(HIDDEN_SIZE, activation='tanh')(x)

        out_actions = Dense(self.action_size, activation='softmax', name='output')(x)

        model = Model(inputs=[state_input, value], outputs=[out_actions])
        model.compile(optimizer=Adam(lr=LR),
                      loss=[actor_critic_loss(
                      value = value)]
                      )
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

    def critic_fit(self,state_array,val_array):
        self.state_array = state_array
        critic_history = self.critic.fit(self.state_array, val_array,
                                         epochs=1, batch_size=16,
                                         shuffle=True)
        return critic_history.history["loss"]
    def actor_fit(self, state_array, action_array, val_array):
        self.state_array = state_array
        self.action_array = action_array

        act_history = self.Actor.fit([self.state_array,val_array], [self.action_array],
                                     epochs=1,
                                     batch_size=16, shuffle=True)
        return act_history.history["loss"]

    def fit(self, state_array, action_array, val_array, next_st_arr):
        ## get value array from reward array
        a_loss = self.actor_fit(state_array, action_array, val_array)
        c_loss = self.critic_fit(state_array, val_array)
        return a_loss, c_loss

    def actor_predict(self, state_array):
        dummy_value = np.zeros((state_array.shape[0],))
        t = self.Actor.predict([state_array.reshape(-1,self.state_size),dummy_value])
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

class advantage_actor_critic(actor_critic):
    def __init__(self, state_size = 11, action_size = 20):
        super().__init__(state_size, action_size)

    def cal_advantage(self, state_array, val_array):
        advantage = val_array - self.critic.predict(state_array)
        return advantage
    def fit(self, state_array, action_array, val_array, next_st_arr):
        ## get value array from reward array
        advantage = self.cal_advantage(state_array, val_array) ## or calculate in some other way
        a_loss = self.actor_fit(state_array, action_array, advantage)
        c_loss = self.critic_fit(state_array, val_array)
        return a_loss, c_loss

class ppo_advantage_actor_critic(advantage_actor_critic):
    def __init__(self, state_size=11, action_size=20):
        super().__init__(state_size, action_size)
        self.Actor = self.build_actor()
        self.old_actor = self.build_actor()
        self.old_actor.set_weights(self.actor.get_weights())

    def build_actor(self):
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

    def actor_fit(self, state_array, action_array, val_array):
        self.state_array = state_array
        self.action_array = action_array
        advantage = self.cal_advantage(state_array, val_array)

        dummy_advantage = np.zeros((self.state_array.shape[0],))
        dummy_pred = np.zeros(self.action_array.shape)
        old_pred = self.old_actor.predict([self.state_array,dummy_advantage, dummy_pred])

        act_history = self.Actor.fit([self.state_array, advantage, old_pred], [self.action_array],
                                     epochs=1,
                                     batch_size=16, shuffle=True)
        return act_history.history["loss"]

    def actor_predict(self, state_array):
        dummy_advantage = np.zeros((state_array.shape[0],))
        dummy_pred = np.zeros((state_array.shape[0],self.action_size))
        t = self.Actor.predict([state_array.reshape(-1,self.state_size),dummy_advantage, dummy_pred])
        return t[-1, :]

    def fit(self, state_array, action_array, value_array, next_st_arr):
        for _ in range(state_array.shape[0] // 16):
            indexes = np.random.randint(0,state_array.shape[0],16)
            self.actor_fit(state_array[indexes,:], action_array[indexes, :], value_array[indexes,:])
        self.old_actor.set_weights(self.Actor.get_weights())
        self.critic_fit(state_array, value_array)

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