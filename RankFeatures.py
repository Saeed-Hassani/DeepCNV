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


def one_hot(_c_name):
    """
        Define the encoder function.
        For example, if we've 3 classes like A B C
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


_raw_data = pa.read_csv("D:\\Thesis\\DeepCNV\\DataSet\\datatest.csv", low_memory=False)
# print(_raw_data)
X = _raw_data[_raw_data.columns[0:_raw_data.shape[1] - 1]].values
print(">> X) ", X)
_class_name = _raw_data[_raw_data.columns[_raw_data.shape[1] - 1]]
_encoder = LabelEncoder()
_encoder.fit(_class_name)
_class_name = _encoder.transform(_class_name)
# print(">> ",_class_name)

Y = one_hot(_class_name)
print(">> Y) ", Y)
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.5, random_state=np.random.seed(7), shuffle=False)

print(">> train_X) ", train_X)
print(">> test_X) ", test_X)
print(">> train_Y) ", train_Y)
print(">> test_Y) ", test_Y)

train_X = np.reshape(train_X, (train_X.shape[0], 1, train_X.shape[1]))
test_X = np.reshape(test_X, (test_X.shape[0], 1, test_X.shape[1]))
print(">> train_X) ", train_X)
print(">> test_X) ", test_X)

model = Sequential()
model.add(LSTM(200, return_sequences=False, input_shape=
(train_X.shape[1], train_X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(train_Y.shape[1], activation='softmax'))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
history = model.fit(train_X, train_Y, validation_split=.10,
                    epochs=500, batch_size=50)
print(">> test_X", test_X)
print(">> Predict...")
print(model.predict(train_X, batch_size=50))
# ====ذخیره مدل به صورت JSON========
# serialize model to JSON
model_json = model.to_json()
with open("model_v3.json", "w") as json_file:
    json_file.write(model_json)

# ======= محاسبه دقت آموزش ========
metrics = model.evaluate(train_X, train_Y)
print('> Training data results: ')
for i in range(len(model.metrics_names)):
    print(str(model.metrics_names[i]) + ": " + str(metrics[i]))

# ====ذخیره وزن ها به صورت h5 ===========
model.save_weights("model_v3.h5")
print("> Saved model to disk")

# ==== تست برنامه =============
# load json and create model
json_file = open('model_v3.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model_v3.h5")
print("> Loaded model from disk")
# evaluate loaded model on test data
loaded_model.compile(loss='mean_squared_error', optimizer='rmsprop',
                     metrics=['accuracy'])
score = loaded_model.evaluate(test_X, test_Y, verbose=0)
print("> %s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))

# ==== نمایش نتایج ======
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Mean Squared Error')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

model.summary()

# >> Get input matrix and create matrix D and T

# >> Main Loop
# >> While |T| > 0

# >> Train the network with LSTM with D matrix

# >> For each j in J, compute the probability using of the 12 equation

# >> Obtained a rank list, J = {j1, j2, ..., jd} ascending

# >> Remove jl from T and delete the l'th feature from D matrix
