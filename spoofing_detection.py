import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Flatten, Dropout, Conv2D
from keras.models import Sequential
import matplotlib.pyplot as plt
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import itertools


# %%
# initialize
# loading train data
real_data = {}
with open(f"data\\users512.p", "rb") as f:
    tmp_dict = pickle.load(f)

    real_data.update(tmp_dict)

real_data = dict(itertools.islice(real_data.items(), 256))

spoofing_data = {}
# with open(f"data\\spoof512.p", "rb") as f:
#     tmp_dict = pickle.load(f)
#
#     spoofing_data.update(tmp_dict)

files = ["spoof0-10.p", "spoof20-30.p", "spoof30-40.p", "spoof40-50.p", "spoof50-60.p", "spoof60-70.p", "spoof70-80.p", "spoof80-90.p", "spoof90-100.p","spoof100-110.p","spoof110-120.p","spoof120-130.p","spoof140-150.p","spoof150-160.p","spoof160-170.p","spoof170-180.p","spoof180-190.p","spoof190-200.p","spoof210-220.p","spoof220-230.p","spoof230-240.p","spoof240-250.p","spoof250-260.p","spoof260-270.p"]

for file in files:
    with open(f"data\\spoof\\{file}", "rb") as f:
        tmp_dict = pickle.load(f)

    spoofing_data.update(tmp_dict)


# DATA PREPARATION
# Cut data into 5 second chunks, flatten,
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


def to_chunk(data_to_chunk: dict, label: str):
    X = []
    Y = []
    for it, person_id in enumerate(data_to_chunk.keys()):
        chunk_len = int((data_to_chunk[person_id]).shape[1] * (5 / 200))
        for position in range(0, data_to_chunk[person_id].shape[1], chunk_len):
            tmp_data = (data_to_chunk[person_id])[:, position:position + chunk_len]
            try:
                tmp_data = np.reshape(tmp_data, (chunk_len * 36,))
                X.append(tmp_data)
                Y.append(label)
            except Exception as e:
                pass
    X = np.array(X)
    Y = np.array(Y)

    return X, Y


X_spoof, Y_spoof = to_chunk(spoofing_data, "spoof")
X_real, Y_real = to_chunk(real_data, "real")

X = np.concatenate((X_spoof, X_real), axis=0)
Y = np.concatenate((Y_spoof, Y_real), axis=0)

encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

X_train, X_valid, Y_train, Y_valid = train_test_split(X, encoded_Y,
                                                      test_size=0.2,
                                                      random_state=123)

# %%






###############################
#           Training
#   k-fold cross validation
###############################

# %%

# def create_model():
model = Sequential()
# model.add(Flatten(input_dim=7740))
model.add(Dense(256, activation='sigmoid'))
model.add(Dense(128, activation='sigmoid'))
model.add(Dense(64, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, Y_train,validation_data = (X_valid,Y_valid), epochs=30, batch_size=512)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# estimator = KerasClassifier(model=create_model, epochs=100, batch_size=512)
# kfold = StratifiedKFold(n_splits=8, shuffle=True)
# results = cross_val_score(estimator, X, encoded_Y, cv=kfold)
# print("Baseline: %.2f%% (%.2f%%))" % (results.mean()*100, results.std()*100))


### ????
# y_porb = model.predict(x)
# metrics.binary_accuracy()


