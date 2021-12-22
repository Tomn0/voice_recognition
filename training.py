import numpy as np
import pickle
import matplotlib.pyplot as plt
from keras.layers import Dense, Flatten
from keras.models import Sequential

# %%
# loading train data
train_data = {}
with open(f"data\\users512.p", "rb") as f:
    tmp_dict = pickle.load(f)

    train_data.update(tmp_dict)

# %%
# loading evaluation data
eval_data = {}
for it in range(2):
    with open(f"data\\test.p", 'rb') as f:
        tmp_dict = pickle.load(f)

    eval_data.update(tmp_dict)
# %%
# nie używane w MLP
ubm_data = np.empty((36, 1))

for person_id in train_data.keys():
  ubm_data = np.concatenate((ubm_data, train_data[person_id]), axis=1)

ubm_data = ubm_data[:, 1:]
print(ubm_data.shape)

# %%
plt.figure(figsize=(20, 10))
for it in range(ubm_data.shape[0]):
  plt.subplot(6, 6, it+1)
  plt.hist(ubm_data[it, :], bins=64)

plt.show()

# %%
# ANN (Artificial Neural Network) DATA PREPARATION
# Cut data into 5 second chunks, flatten,
# Create parallel person label tensor (id -> number -> onehot)
# Split data into training and validation sets for neural network training
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
X = []
Y = []
for it, person_id in enumerate(train_data.keys()):
  chunk_len = int((train_data[person_id]).shape[1]*(5/200))
  print(chunk_len*36)
  for position in range(0, train_data[person_id].shape[1], chunk_len):
    tmp_data = (train_data[person_id])[:, position:position+chunk_len]
    try:
      tmp_data = np.reshape(tmp_data, (chunk_len*36,))
      X.append(tmp_data)
      Y.append(it)
    except Exception as e:
      print(f'sample shape = {tmp_data.shape}, not appending')
    # print(tmp_data.shape)
X = np.array(X)
Y = np.array(Y)
print(X.shape)
print(Y.shape)
Y_onehot = to_categorical(Y, num_classes=512)
print(Y_onehot.shape)
X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y_onehot,
                                                      test_size=0.2,
                                                      random_state=123)

# %%
print(X_train.shape)
print(X_valid.shape)
print(Y_train.shape)
print(Y_valid.shape)
# %%
# Model1
model1 = Sequential()
model1.add(Dense(128, input_dim=7740,  activation='relu'))
model1.add(Flatten())
model1.add(Dense(64, activation='sigmoid'))
model1.add(Dense(128, activation='relu'))
model1.add(Dense(512, activation='softmax'))

model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model1.fit(X_train, Y_train,validation_data = (X_valid,Y_valid), epochs=100, batch_size=128)

# list all data in history
print(history.history.keys())
# summarize history for accuracy
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

# %%
center_vector1 = model1.predict(X_train).mean(axis=0)
# print(center_vector1.shape)