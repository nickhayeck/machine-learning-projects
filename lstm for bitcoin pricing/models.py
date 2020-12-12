import pandas as pd
import keras
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def loadData_mlp(path):
    raw = pd.read_csv(path)
    #remove NaNs
    raw = raw[raw.isnull()['btc_trade_volume']==False]
    x = raw.values[1:,2:]
    y = raw.values[:,1]
    #turn y into up/down data
    y = y[1:]-y[:-1] < np.zeros(len(y)-1)
    #ensure numpy arrays & scale data
    x = np.asarray(x).astype('float32')
    y = np.asarray(y).astype('float32')
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    #ensure no NaNs are being introduced by input data
    assert not np.any(np.isnan(x))
    assert not np.any(np.isnan(y))

    return x, y

def loadData_lstm(path):
    lookback_n = 3
    raw = pd.read_csv(path)
    raw['btc_trade_volume'] = raw['btc_trade_volume'].fillna(0)
    x = raw.values[177:2500-1,1:]
    for i in range(1,lookback_n):
        x = np.concatenate((x,raw.shift(i).values[177:2500-1,1:]), axis=1)
    y = raw.shift(-1).values[177:2500-1,1]

    #ensure numpy arrays
    x = np.asarray(x).astype('float32').reshape(x.shape[0],1,x.shape[1])
    y = np.asarray(y).astype('float32')
    #ensure no NaNs are being introduced by input data
    assert not np.any(np.isnan(x))
    assert not np.any(np.isnan(y))
    #perform train/test split
    n = int(x.shape[0]*0.01//1)
    return x,y

############
## MODELS ##
############

### MULTILAYER PERCEPTRON ###
def mlp(x_train, y_train, args=None):
    # Define model architecture
    model = keras.Sequential([  keras.layers.Input(shape=(22)),
                                keras.layers.Dense(128),
                                keras.layers.Dense(128),
                                keras.layers.Dense(1),
                                keras.layers.Activation(keras.activations.sigmoid)
                                ])
    # Optimizer
    opt = keras.optimizers.Adam(learning_rate=args['LR'])
    # Compilation
    model.compile(optimizer=opt,
              	  loss=keras.losses.BinaryCrossentropy(),
              	  metrics=['accuracy'])
    #Training
    history = model.fit(x=x_train,
                        y=y_train,
                        epochs=args['epochs'],
                        validation_split=0.1)
    return model, history


### Random Forest ###
def randomForest(x,y, args=None):
    classifier = RandomForestClassifier(n_estimators=100)
    accuracy = 0
    v=10
    for i in range(v):
        x_train, x_test, y_train, y_test = train_test_split(
            x, y,test_size=0.15
        )
        model = classifier.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        print("Random Forest accuracy on trial",i+1,":",accuracy_score(y_test, y_pred))
        accuracy += accuracy_score(y_test, y_pred)
    print("Average Random Forest accuracy:", accuracy/v)
    return 5


### LSTM ###
def lstm(x,y,args=None):
    # Define model architecture
    model = keras.Sequential([  keras.layers.LSTM(50),
                                keras.layers.Dense(1),
                                keras.layers.Activation(keras.activations.relu)
                                ])
    # Optimizer
    opt = keras.optimizers.Adam(learning_rate=args['LR'])
    # Compilation
    model.compile(optimizer=opt,
              	  loss=keras.losses.MeanAbsoluteError())
    #Training
    history = model.fit(x=x,
                        y=y,
                        epochs=args['epochs'])
    return model, history







if __name__ == '__main__':
    mlp_args = {'LR' : 0.001, 'epochs': 100}
    lstm_args = {'LR' : 0.05, 'epochs': 500}

    #mlp
    x,y = loadData_mlp('data.csv')
    mlp_model,mlp_hist = mlp(x,y,args=mlp_args)
    plt.plot(mlp_hist.history['loss'])
    plt.show()

    #randomForest
    randomForest(x,y)

    #lstm
    x,y = loadData_lstm('data.csv')
    lstm_model,lstm_hist = lstm(x,y, args=lstm_args)
    y_pred = lstm_model.predict(x)
    plt.plot(y, label='actual')
    plt.plot(y_pred, color='green', label='predicted')
    plt.show()
    plt.plot(lstm_hist.history['loss'])
    plt.show()
