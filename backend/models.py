import tensorflow as tf
from tensorflow.keras import optimizers, callbacks
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dropout, Dense
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import mean_squared_log_error

def f_regr():

    model_in = Input(shape = (6369,), dtype = 'float32', sparse = True)
    out = Dense(256, activation = 'relu')(model_in)
    out = Dropout(0.2)(out)
    out = Dense(128, activation = 'relu')(out)
    out = Dense(64, activation = 'relu')(out)
    out = Dense(32, activation = 'relu')(out)
    out = Dense(16, activation = 'relu')(out)
    model_out = Dense(1)(out)
    model = Model(model_in, model_out)
    model.compile(loss = 'mean_squared_error', optimizer = Adam(0.001))
    
    return model