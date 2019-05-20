#ref https://github.com/OctThe16th/PPO-Keras/blob/master/Main.py
import tensorflow
import tensorflow.keras.backend as K


def proximal_policy_optimization_loss(advantage, old_prediction):
    LOSS_CLIPPING = 0.2
    ENTROPY_LOSS = 2e-1
    # print("shape of advantage ", K.int_shape(advantage))
    def loss(y_true, y_pred):
        prob = y_true * y_pred
        old_prob = y_true * old_prediction
        r = prob/(old_prob + 1e-10)
        # print("shape of r", K.int_shape(r))
        return -K.mean(K.minimum(r * advantage, K.clip(r, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantage) + ENTROPY_LOSS * (-prob * K.log(prob + 1e-10)))
    return loss

def actor_critic_loss(advantage):
    ENTROPY_LOSS = 1e-3
    def loss(y_true,y_pred):
        prob = y_true * y_pred
        return -K.mean(K.log(prob + 1e-10) * y_true * advantage + ENTROPY_LOSS * (-prob * K.log(prob + 1e-10)))
    return loss

def proximal_policy_optimization_loss_continuous(advantage, old_prediction):
    def loss(y_true, y_pred):
        var = K.square(NOISE)
        pi = 3.1415926
        denom = K.sqrt(2 * pi * var)
        prob_num = K.exp(- K.square(y_true - y_pred) / (2 * var))
        old_prob_num = K.exp(- K.square(y_true - old_prediction) / (2 * var))

        prob = prob_num/denom
        old_prob = old_prob_num/denom
        r = prob/(old_prob + 1e-10)

        return -K.mean(K.minimum(r * advantage, K.clip(r, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantage))
    return loss