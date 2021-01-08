print('Setting up')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from utility import *
from sklearn.model_selection import train_test_split


path = 'mySimData'
data = importData(path)

#visualizing and balancing the sim data

data = regData(data, flag=False)
# regData(data, flag=True)

imgsList, steerLists = inputData(path,data)

# training and validation data ,, 80% of data for training , 20% for validation
x_train, x_valid, y_train, y_valid = train_test_split(imgsList, steerLists, test_size=0.2, random_state=5)
print('Num of Training Imgs: ', len(x_train))
print('Num of Validation Imgs: ', len(x_valid))
#
#
#
#
#
#
#
model = genModel()
model.summary()

history = model.fit(dataGen(x_train,y_train,150,1),steps_per_epoch=300, epochs=10, validation_data=dataGen(x_valid,y_valid,150,0),
          validation_steps=200)

model.save('model.h5')
print('Model has been saved')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training', 'Validation'])
plt.ylim([0.025,0.065])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()