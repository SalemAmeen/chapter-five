from numpy import *

model = Sequential()
model.add(Dense(300,
                input_shape=(150,), 
                activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(7, activation="softmax"))

model.load_weights('/Users/salemameen/Desktop/banditsbook/python_face/faceModelbest.hdf5')


model.compile(loss='categorical_crossentropy',
              metrics=['accuracy'],
              optimizer='adam')
# Actual modelling

score, accuracy = model.evaluate(X_test_pca, labelsTest, batch_size=100, verbose=0)

print("Test fraction correct (NN-Score) = {:.2f}".format(score))
print("Test fraction correct (NN-Accuracy) = {:.2f}".format(accuracy))

####################################################             
#n_samples ,_=Y_test.shape
SamplingTesting=500
All_weights=model.get_weights()
All_weights_BUCKUP = model.get_weights()
FC_weights_3=All_weights[0]
row,col= shape(FC_weights_3)
SizeWights=row*col
OldAccuracy = accuracy
