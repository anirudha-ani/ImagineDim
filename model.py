import tensorflow as tf
import gensim
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape
from preprocessing import get_shapenet_data
import matplotlib.pyplot as plt

from DataPreprocess.ShapeNet.shapenet import loadVerifiedShapenetData

def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
#   plt.plot(history.history['val_loss'], label='val_loss')
#   plt.ylim([10, 70])
  plt.xlabel('Epoch')
  plt.ylabel('Error [MPG]')
  plt.legend()
  plt.grid(True)
  plt.show()

# def predict(word, prediction_model, embedding_model):

def main():
    print("Data preprocessing start")
    # embedding_model = gensim.models.KeyedVectors.load_word2vec_format('Dataset/GoogleNews/GoogleNews-vectors-negative300.bin', binary=True)
    train_x, train_y, test_x, test_y, train_english_word, test_english_word = get_shapenet_data()
    # train_y = train_y
    # print(train_x[0:3])
    # print(train_y[0:3])
    train_x = train_x[0:1000, :]
    train_y = train_y[0:1000]

   
    print("Data preprocessing done")
    prediction_model = Sequential([
        # Dense(512, activation=tf.nn.relu),
        # Dense(512, activation=tf.nn.relu),
        Dense(512, activation=tf.nn.relu), 
        Dense(300, activation=tf.nn.relu),
        Dense(1)
    ])

    prediction_model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(0.00001))

    history = prediction_model.fit(
        train_x,
        train_y,
        # validation_split=.01,
        epochs=5000,
        verbose=1
    )

    plot_loss(history)

    # print("word = ", test_english_word[0], "prediction = ", prediction_model.predict(np.expand_dims(test_x[0], axis=0)), " actual = ", test_y[0])
    # print("word = ", test_english_word[1], "prediction = ", prediction_model.predict(np.expand_dims(test_x[1], axis=0)), " actual = ", test_y[1])
    # print("word = ", test_english_word[2], "prediction = ", prediction_model.predict(np.expand_dims(test_x[2], axis=0)), " actual = ", test_y[2])
    # print("word = ", test_english_word[3], "prediction = ", prediction_model.predict(np.expand_dims(test_x[3], axis=0)), " actual = ", test_y[3])

    test_prediction = prediction_model.predict(test_x)
    percentage_error_over_test = []

    # for index in range(len(test_english_word)):
    for index in range(10):
        print("word = ", test_english_word[index], " prediction = ",test_prediction[index], " actual = ", test_y[index])
        percentage_error= abs((test_y[index] - test_prediction[index][0]) / test_y[index] * 100.)
        percentage_error_over_test.append(percentage_error)
        print("pecentage error = ", percentage_error)
    print("Average percentage error = ", np.average(percentage_error_over_test)) 

    # miss_counter_over_test = []
    # for index in range(len(test_english_word)):
    #     miss_counter = 0 
    #     for index2 in range(len(test_english_word)):
    #         if index == index2:
    #             continue

    #         if test_y[index2] < test_y[index] and test_prediction[index2][0] > test_prediction[index][0]:
    #             miss_counter+=1
    #         elif test_y[index2] > test_y[index] and test_prediction[index2][0] < test_prediction[index][0]:
    #             miss_counter +=1
    #     print("miss counter for word = ", test_english_word[index], " = " ,miss_counter)
    #     miss_counter_over_test.append(miss_counter)
    # print("Total test = ",len(test_english_word), "Average miss counter = ", np.average(miss_counter_over_test)) 
   
   
    train_prediction = prediction_model.predict(train_x)
    percentage_error_over_train = []
    for index in range(10):
        print("word = ", train_english_word[index], " prediction = ",train_prediction[index], " actual = ", train_y[index])
        percentage_error= abs((train_y[index] - train_prediction[index][0]) / train_y[index] * 100.)
        percentage_error_over_train.append(percentage_error)
        print("pecentage error = ", percentage_error)
    print("Average train percentage error = ", np.average(percentage_error_over_train)) 
if __name__ == '__main__':
    main()