import numpy as np
import pickle
import matplotlib
import matplotlib.pyplot as plt
from keras.layers import Dense, Flatten
from keras.models import Sequential
from sklearn.metrics.pairwise import cosine_similarity

# %%
# initialize
# loading train data
train_data = {}
with open(f"data\\users512.p", "rb") as f:
    tmp_dict = pickle.load(f)

    train_data.update(tmp_dict)

#
# loading evaluation data
eval_data = {}
for it in range(2):
    with open(f"data\\test.p", 'rb') as f:
        tmp_dict = pickle.load(f)

    eval_data.update(tmp_dict)
#
# nie używane w MLP
ubm_data = np.empty((36, 1))

for person_id in train_data.keys():
  ubm_data = np.concatenate((ubm_data, train_data[person_id]), axis=1)

ubm_data = ubm_data[:, 1:]
print(ubm_data.shape)

#
plt.figure(figsize=(20, 10))
for it in range(ubm_data.shape[0]):
  plt.subplot(6, 6, it+1)
  plt.hist(ubm_data[it, :], bins=64)

plt.show()

#
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



#
print(X_train.shape)
print(X_valid.shape)
print(Y_train.shape)
print(Y_valid.shape)

# pomysły na poprawę modelu: https://machinelearningmastery.com/improve-deep-learning-performance/

# %%
# # Model1
# model1 = Sequential()
# model1.add(Flatten(input_dim=7740))
# model1.add(Dense(64, activation='relu'))
# model1.add(Dense(128, activation='relu'))
# model1.add(Dense(512, activation='sigmoid'))
# model1.add(Dense(1024, activation='relu'))
# model1.add(Dense(512, activation='softmax'))
# model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

########################
# Model1
# model1 = Sequential()
# model1.add(Flatten(input_dim=7740))
# model1.add(Dense(1024, activation='relu'))
# model1.add(Dense(512, activation='relu'))
# model1.add(Dense(256, activation='relu'))
# model1.add(Dense(512, activation='relu'))
# model1.add(Dense(128, activation='relu'))
# model1.add(Dense(256, activation='relu'))
# model1.add(Dense(128, activation='relu'))
# model1.add(Dense(256, activation='relu'))
# model1.add(Dense(128, activation='relu'))
# model1.add(Dense(512, activation='softmax'))
#
# model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

###################
model1 = Sequential()
model1.add(Flatten(input_dim=7740))
model1.add(Dense(1024, activation='relu'))
model1.add(Dense(512, activation='softsign'))
model1.add(Dense(512, activation='softsign'))
model1.add(Dense(512, activation='softmax'))

model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model1.summary()

history = model1.fit(X_train, Y_train,validation_data = (X_valid,Y_valid), epochs=100, batch_size=1024)

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

#
center_vector1 = model1.predict(X_train).mean(axis=0)
# print(center_vector1.shape)
models = {}
models['model1'] = (model1, center_vector1)


 # %%
# util functions
def create_enroll_and_test_set(eval_user, enroll_len=60, test_len=30):
    slice_len = 5  # długość slice'a (5 sekund)
    enroll_slices_num = round(enroll_len / slice_len)
    test_slices_num = round(test_len / slice_len)

    # przesuwamy index aby nie brać dwa razy tych samych danych
    index_start = 0
    inc_5s = round(slice_len * (eval_user.shape[1] / 200))  # 5 seconds increment added after each slice
    enroll = np.empty(shape=(0, 7740))
    test = np.empty(shape=(0, 7740))

    index = round(slice_len * (eval_user.shape[1] / 200))

    for _ in range(enroll_slices_num):
        new_slice = eval_user[:, index_start:index].reshape(1, -1)
        enroll = np.append(enroll, new_slice, axis=0)
        index_start = index
        index = index + inc_5s

    for _ in range(test_slices_num):
        new_slice = eval_user[:, index_start:index].reshape(1, -1)
        test = np.append(test, new_slice, axis=0)
        index_start = index
        index = index + inc_5s

    return enroll, test

def centralize(vect, centr):
  assert len(vect) == len(centr), f"{len(vect)} != {len(centr)}"

  vect = vect - centr

  return vect

def calc_confidences(model_id):
    # global model, center_vector
    confidence_TP = []
    confidence_TN = []
    model = models[f'model{model_id}'][0]
    center_vect = models[f'model{model_id}'][1]

    # generowanie embeddingu
    for userA in eval_data.keys():
        enroll_A, test_A = create_enroll_and_test_set(eval_data[userA], enroll_len=60, test_len=30)

        enroll_embedding_A = model.predict(enroll_A).mean(axis=0)
        enroll_embedding_A = centralize(enroll_embedding_A, center_vect)

        test_embedding_A = model.predict(test_A).mean(axis=0)
        test_embedding_A = centralize(test_embedding_A, center_vect)

        for userB in eval_data.keys():
            if userA == userB:
                continue

            enroll_B, test_B = create_enroll_and_test_set(eval_data[userB], enroll_len=60, test_len=30)

            enroll_embedding_B = model.predict(enroll_B).mean(axis=0)
            enroll_embedding_B = centralize(enroll_embedding_B, center_vect)

            test_embedding_B = model.predict(test_B).mean(axis=0)
            test_embedding_B = centralize(test_embedding_B, center_vect)

            # confidence_TP.append(cosine_similarity(enroll_embedding_A, test_embedding_A))
            confidence_TP.append(
                cosine_similarity(np.expand_dims(enroll_embedding_A, axis=0), np.expand_dims(test_embedding_A, axis=0)))

            # confidence_TN.append(cosine_similarity(enroll_embedding_A, test_embedding_B))
            confidence_TN.append(
                cosine_similarity(np.expand_dims(enroll_embedding_A, axis=0), np.expand_dims(test_embedding_B, axis=0)))

    confidence_TP = np.squeeze(np.array(confidence_TP))
    confidence_TN = np.squeeze(np.array(confidence_TN))

    return confidence_TP, confidence_TN


# model evaluation
confidence_TP, confidence_TN = calc_confidences(1)
# confidence_TP_2, confidence_TN_2 = calc_confidences(2)
# confidence_TP_3, confidence_TN_3 = calc_confidences(3)
# confidence_TP_4, confidence_TN_4 = calc_confidences(4)

scoring = {}
scoring['model1'] = (confidence_TN.tolist(), confidence_TP.tolist())
# scoring['model2'] = (confidence_TN_2.tolist(), confidence_TP_2.tolist())
# scoring['model3'] = (confidence_TN_3.tolist(), confidence_TP_3.tolist())
# scoring['model4'] = (confidence_TN_4.tolist(), confidence_TP_4.tolist())

far_frr = {}
far = []
frr = []

legend_score = ['score True Negative', 'score True Positive']
legend_f = ['false acceptance rate', 'false rejection rate']
plt.figure(figsize=(25, 10))
for it, model in enumerate(scoring.keys()):
  plt.subplot(2, len(scoring.keys()), it +1)
  plt.title(model)
  n_TN, bins, _ = plt.hist(scoring[model][0], bins=100, range=[-1, 1], alpha=0.5)
  n_TP, bins, _ = plt.hist(scoring[model][1], bins=100, range=[-1, 1], alpha=0.5)
  plt.grid()
  plt.xlim([-1.5, 1.5])
  plt.legend(legend_score, loc='upper right')
  plt.xlabel('log likelihood')

  # Implementacja False Rejection Rate / False Acceptance Rate
  # Proszę uzupełnić słownik far_frr analogicznie do słownika scoring
  # zamiast wartości score uzupełnić far, frr
  # Miejsce na kod
  # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  far = 1 - (np.cumsum(n_TN) / np.sum(n_TN))
  frr = np.cumsum(n_TP) / np.sum(n_TP)

  far_frr[model] = (far, frr)


  # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  plt.subplot(2, len(scoring.keys()), it + 1 + len(scoring.keys()))
  plt.plot(bins[1:], far_frr[model][0])
  plt.plot(bins[1:], far_frr[model][1])
  plt.legend(legend_f, loc='lower left')
  plt.xlabel('threshold')
  plt.ylabel('probability')
  plt.grid()
plt.show()

def DETCurve(far_frr):
    """
    Given false positive and false negative rates, produce a DET Curve.
    The false positive rate is assumed to be increasing while the false
    negative rate is assumed to be decreasing.
    """
    plt.figure(figsize=(10, 10))
    axis_min = 0
    fig,ax = plt.subplots(figsize=(10, 10))
    plt.yscale('log')
    plt.xscale('log')
    # ticks_to_use = [0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10,20,50]
    ticks_to_use = [8, 15, 20, 27, 30, 50, 100]
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xticks(ticks_to_use)
    ax.set_yticks(ticks_to_use)
    legend_det = [dir]
    for model in far_frr.keys():
      (far, frr) = far_frr[model]
      plt.plot(far*100,frr*100)
    plt.axis([8,100,8,100])
    plt.grid()
    plt.legend(far_frr.keys(), loc='upper right')
    plt.xlabel('false acceptance rate (%)')
    plt.ylabel('false rejection rate (%)')
    plt.show()

DETCurve(far_frr)


#########################
#       RESULTS
#########################

# import csv
