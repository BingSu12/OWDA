from keras.layers import Dense  #, Merge
from keras.models import Sequential
from keras.regularizers import l2
from keras.layers import Lambda
import keras.backend as K


def create_model(input_dim, reg_par, outdim_size):
    """
    Builds the model
    The structure of the model can get easily substituted with a more efficient and powerful network like CNN
    """
    model = Sequential()

    model.add(Dense(1024, input_shape=(input_dim,), activation='relu'))  #, kernel_regularizer=l2(reg_par)
    #model.add(BatchNormalization()) 
    model.add(Dense(1024, activation='relu'))  #, kernel_regularizer=l2(reg_par)
    model.add(Dense(1024, activation='relu'))  #, kernel_regularizer=l2(reg_par)
    model.add(Dense(outdim_size, activation='linear'))  #, kernel_regularizer=l2(reg_par), activity_regularizer=l2(1)
    model.add(Lambda(lambda x: K.l2_normalize(x,axis=1)))

    return model