import tensorflow as tf
from keras import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from pandas import read_csv
import csv

def load_data():
    d = read_csv('Breast_cancer_data.csv')
    data_array = d.values
    return data_array

def get_training_data(data_array):
    X_train = data_array[:400, :5]
    y_train = data_array[:400,5:6]
    return X_train,y_train

def get_testing_data(data_array):
    X_test = data_array[400:,:5]
    y_test = data_array[400:,5:6]
    return X_test, y_test

def get_yhat(predictions):
    yhat = np.zeros_like(predictions)
    for i in range(len(predictions)):
        if predictions[i] >= 0.05:
            yhat[i] = 1
        else:
            yhat[i] = 0
    return yhat

def get_percent_accuracy(yhat,target):
    count = 0
    for i in range(len(yhat)):
        print("Y_hat is: ", yhat[i], "and target y is: ", target[i])
        if yhat[i] == target[i]:
            count = count+1
    return count/len(yhat)


if __name__ == "__main__":
    data_set = load_data()
    X_train, y_train = get_training_data(data_set)
    norm_l = tf.keras.layers.Normalization(axis=-1)
    norm_l.adapt(X_train)
    X_train_norm = norm_l(X_train)
    X_norm_t = np.tile(X_train_norm, (1000,1))
    Yt = np.tile(y_train,(1000,1))
    print(X_norm_t.shape, Yt.shape)
    tf.random.set_seed(1234)
    model = Sequential([
        tf.keras.Input(shape=(5,)),
        Dense(3, activation='sigmoid', name='layer1'),
        Dense(1, activation = 'sigmoid', name='layer2')
    ])
    model.summary()
    L1_num_params = 5*3 + 3
    L2_num_params = 3*1 + 1
    print("L1 params = ", L1_num_params, " and L2 params = ", L2_num_params)
    W1, b1 = model.get_layer("layer1").get_weights()
    W2, b2 = model .get_layer("layer2").get_weights()
    print("Shape of W1 is: ", W1.shape, W1, " and b1 is : ", b1.shape, b1)
    print("Shape of W2 is: ", W2.shape, W2,  "and b2 is : ", b2.shape, b2)
    model.compile(
        loss = tf.keras.losses.BinaryCrossentropy(),
        optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01),
    )
    model.fit(
        X_norm_t,Yt,
        epochs = 10,
    )
    W1, b1 = model.get_layer("layer1").get_weights()
    W2, b2 = model.get_layer("layer2").get_weights()
    print("Shape of new W1 is: ", W1.shape, W1, " and b1 is : ", b1.shape, b1)
    print("Shape of new W2 is: ", W2.shape, W2, "and b2 is : ", b2.shape, b2)
    model.get_layer("layer1").set_weights([W1,b1])
    model.get_layer("layer2").set_weights([W2,b2])
    X_test,y_test = get_testing_data(data_set)
    X_test_norm = norm_l(X_test)
    predictions = model.predict(X_test_norm)
    yhat = get_yhat(predictions)
    print("Percentage accuracy is ", get_percent_accuracy(yhat,y_test))



