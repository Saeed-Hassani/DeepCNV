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

# "D:\\Thesis\\DeepCNV\\DataSet\\MONK-1\\monk-1-test-c.csv",
#          "D:\\Thesis\\DeepCNV\\DataSet\\MONK-1\\monk-1-test-cd.csv",
#          "D:\\Thesis\\DeepCNV\\DataSet\\MONK-1\\monk-1-test-cde.csv",
#          "D:\\Thesis\\DeepCNV\\DataSet\\MONK-1\\monk-1-test-cdea.csv",
#          "D:\\Thesis\\DeepCNV\\DataSet\\MONK-1\\monk-1-test-cdeab.csv",
#          "D:\\Thesis\\DeepCNV\\DataSet\\MONK-1\\monk-1-test-cdeabf.csv",
_path = ["D:\\Thesis\\DeepCNV\\DataSet\\MONK-1\\monk-1-test-b.csv",
         "D:\\Thesis\\DeepCNV\\DataSet\\MONK-1\\monk-1-test-bf.csv",
         "D:\\Thesis\\DeepCNV\\DataSet\\MONK-1\\monk-1-test-afb.csv",
         "D:\\Thesis\\DeepCNV\\DataSet\\MONK-1\\monk-1-test-eafb.csv",
         "D:\\Thesis\\DeepCNV\\DataSet\\MONK-1\\monk-1-test-deafb.csv",
         "D:\\Thesis\\DeepCNV\\DataSet\\MONK-1\\monk-1-test-cdeafb.csv"]
_mse = []


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


for v in _path:
    _raw_data = pa.read_csv(v, low_memory=False)
    _x = _raw_data[_raw_data.columns[0:_raw_data.shape[1] - 1]].values
    _class_name = _raw_data[_raw_data.columns[_raw_data.shape[1] - 1]]
    _encoder = LabelEncoder()
    _encoder.fit(_class_name)
    _class_name = _encoder.transform(_class_name)
    _y = one_hot(_class_name)
    # train_x, test_x, train_y, test_y = train_test_split(_x, _y, test_size=0.1, random_state=np.random.seed(7),
    #                                                     shuffle=False)
    train_x, test_x, train_y, test_y = train_test_split(_x, _y, test_size=0, random_state=np.random.seed(7),
                                                        shuffle=False)

    train_x = np.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1]))
    test_x = np.reshape(test_x, (test_x.shape[0], 1, test_x.shape[1]))

    model = Sequential()
    model.add(LSTM(50, return_sequences=False, input_shape=(train_x.shape[1], train_x.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(train_y.shape[1], activation='softmax'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    history = model.fit(train_x, train_y, validation_split=.20,
                        epochs=100, batch_size=50)
    metrics = model.evaluate(train_x, train_y)
    # for i in range(len(model.metrics_names)):
    #     print(metrics.__len__())
    print(str(model.metrics_names[0]) + ": " + str(metrics[0]))
    _mse.append(metrics[0])
print(_mse)
plt.plot(_mse)
plt.show()
