import matplotlib.pyplot as plt
import numpy as np
import pandas as pa
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.models import Sequential
from keras.models import model_from_json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from time import gmtime, strftime
from keras.callbacks import CSVLogger
from keras.callbacks import Callback
from keras import backend as K

# **********(_ Define Variables _)**********
# >> Pattern of name model
# >> model_[LSTM units]_[Dropout percent]_[test_size]_[Validation]_[Epochs]_[Batch Size]

_lstm_units = 10
_dropout = .2
_test_sizze = .5
_validation = 0.15
_epochs = 3
_batch_size = 1
_now = strftime("%Y-%m-%d_%H-%M-%S_", gmtime())
_model_name = "model_" + _now + str(_lstm_units) + "_" + str(_dropout) + "_" + str(_test_sizze) + "_" + str(
    _validation) + "_" + str(_epochs) + "_" + str(
    _batch_size)


# **********(_ Define on_hot function _)**********
def one_hot(_c_name):
    """
        Define the encoder function.
        For example, if we've 3 classes, A B C
        for input [A B C] the output is as follow:
        _c_name =  [1 0 0
                    0 1 0
                    0 0 1]
    """
    _n_class = len(_c_name)
    _unique_labels = len(np.unique(_c_name))
    _output = np.zeros((_n_class, _unique_labels))
    _output[np.arange(_n_class), _c_name] = 1
    return _output


# **********(_ Define Callback _)**********
class LogThirdLayerOutput(Callback):
    def on_epoch_end(self, epoch, logs=None):
        layer_output = K.function([model.layers[0].input],
                                  [model.layers[2].output])(self.validation_data)[0]
        print(layer_output)

    def on_batch_end(self, batch, logs={}):
        # print(logs.get('loss'))
        layer_output = K.function([model.layers[0].input],
                                  [model.layers[2].output])(self.validation_data)[0]

        print(batch)

        # def get_layer(model,x):


# from keras import backend as K
#
#     get_3rd_layer_output = K.function([model.layers[0].input],
#                                       [model.layers[2].output])
#     layer_output = get_3rd_layer_output([x])[0]
#     print(layer_output.shape)
#     return layer_output
_raw_data = pa.read_csv("D:\\Thesis\\DeepCNV\\DataSet\\fake.csv", low_memory=False)
X = _raw_data[_raw_data.columns[0:_raw_data.shape[1] - 1]].values
_c_name = _raw_data[_raw_data.columns[_raw_data.shape[1] - 1]]
_encoder = LabelEncoder()
_encoder.fit(_c_name)
_c_name = _encoder.transform(_c_name)
Y = one_hot(_c_name)

# >> Split data set into train and test ...
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=_test_sizze, random_state=np.random.seed(7),
                                                    shuffle=False)

# >> Reshape train and test ...
train_X = np.reshape(train_X, (train_X.shape[0], 1, train_X.shape[1]))
test_X = np.reshape(test_X, (test_X.shape[0], 1, test_X.shape[1]))

print(train_X)
# print(test_X)

# **********(_ Define stack mode _)**********

model = Sequential()
# >> Since we know the shape of our Data we can input the time step and feature data
# >> The number of time step sequence are dealt with in the fit function
model.add(LSTM(_lstm_units, return_sequences=False, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dropout(_dropout))

# **********(_ Number of features on the output _)**********

model.add(Dense(train_Y.shape[1], activation='softmax'))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
csv_logger = CSVLogger(_model_name + '_log.txt', append=True, separator=';')

history = model.fit(train_X, train_Y, validation_split=_validation,
                    epochs=_epochs, batch_size=_batch_size,
                    callbacks=[csv_logger, LogThirdLayerOutput()],
                    verbose=0)

# **********(_ serialize model to JSON _)**********

model_json = model.to_json()
with open(_model_name + ".json", "w") as json_file:
    json_file.write(model_json)

_file = open(_model_name + "_result.txt", "w")

# **********(_ Calculate the accuracy of training _)**********

metrics = model.evaluate(train_X, train_Y)
print('> Training data results: ')
_file.write('> Training data results: \n')
for i in range(len(model.metrics_names)):
    _st = str(model.metrics_names[i]) + ": " + str(metrics[i])
    _file.write(_st + '\n')
    print(_st)

# **********(_ Save the weights as h5 format _)**********

model.save_weights(_model_name + ".h5")
print("> Saved model to disk")

# **********(_ Model testing _)**********

# >> Load json and create model...
json_file = open(_model_name + '.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# >> Load weights into new model...
loaded_model.load_weights(_model_name + ".h5")
print("> Loaded model from disk")

# >> Evaluate loaded model on test data...
loaded_model.compile(loss='mean_squared_error', optimizer='rmsprop',
                     metrics=['accuracy'])
score = loaded_model.evaluate(test_X, test_Y, verbose=0)
_file.write("\n> Accracy:\n %s: %.2f%% \n" % (loaded_model.metrics_names[1], score[1] * 100))
print("> %s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))
_file.close()

# **********(_ Show results _)**********

# >> List all data in history...
_fig1 = plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.grid(True, linestyle='-', linewidth=.5)
# plt.show()
plt.savefig(_model_name + '_acc.eps', format='eps', dpi=1000)
plt.close(_fig1)
# >> Summarize history for loss...
_fig2 = plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Mean Squared Error')
plt.ylabel('Loss')
plt.grid(True, linestyle='-', linewidth=.5)
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper right')
# plt.show()
plt.savefig(_model_name + '_loss.eps', format='eps', dpi=1000)
plt.close(_fig2)

# **********(_ Summarize history for accuracy _)**********

model.summary()
