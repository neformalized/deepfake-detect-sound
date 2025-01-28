from tensorflow.keras import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv1D, MaxPooling1D
from tensorflow.keras.optimizers import Adam

def create(input_shape, output_shape):
    
    model = Sequential()
    
    ###
    
    model.add(Input(shape=(input_shape, 1)))
    
    model.add(Conv1D(2, (16), activation='tanh', padding='valid'))
    model.add(MaxPooling1D(pool_size=(2)))
    
    model.add(Conv1D(4, (16), activation='tanh', padding='valid'))
    model.add(MaxPooling1D(pool_size=(2)))
    
    model.add(Conv1D(8, (16), activation='tanh', padding='valid'))
    model.add(MaxPooling1D(pool_size=(2)))
    
    model.add(Flatten())
    model.add(Dense(128, activation='tanh'))
    
    model.add(Dense(output_shape, activation='sigmoid'))
    
    ###
    
    optimizer = Adam(learning_rate = 0.001)
    model.compile(optimizer = optimizer, loss = "binary_crossentropy")
    
    ###
    
    model.summary()
    
    ###
    
    return model
#