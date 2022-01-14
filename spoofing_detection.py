import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Flatten
from keras.models import Sequential
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

# %%
# initialize
# loading train data
real_data = {}
with open(f"data\\users512.p", "rb") as f:
    tmp_dict = pickle.load(f)

    real_data.update(tmp_dict)

spoofing_data = {}
with open(f"data\\spoof512.p", "rb") as f:
    tmp_dict = pickle.load(f)

    spoofing_data.update(tmp_dict)

#
# loading evaluation data
# eval_data = {}
# for it in range(2):
#     with open(f"data\\test.p", 'rb') as f:
#         tmp_dict = pickle.load(f)
#
#     eval_data.update(tmp_dict)


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

###############################
#           Training
#   k-fold cross validation
###############################


def create_model():
    model = Sequential()
    model.add(Dense(521, input_dim=7740, activation='relu'))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


estimator = KerasClassifier(model=create_model, epochs=100, batch_size=512)
kfold = StratifiedKFold(n_splits=8, shuffle=True)
results = cross_val_score(estimator, X, encoded_Y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%))" % (results.mean()*100, results.std()*100))

