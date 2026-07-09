from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth=True
session=InteractiveSession(config=config)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow
import matplotlib.pyplot as plt
import numpy as numpy
classifier = Sequential()
classifier.add(Conv2D(32,(3,3),input_shape=(128,128,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Flatten())
classifier.add(Dense(units=128,activation='relu'))
classifier.add(Dense(units=3,activation='softmax'))
classifier.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
train_datagen=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
test_datagen=ImageDataGenerator(rescale=1./255)
training_set=train_datagen.flow_from_directory("./dataset/train", target_size=(128,128), batch_size=6, class_mode='categorical')
valid_set=test_datagen.flow_from_directory("./dataset/test",target_size=(128,128),batch_size=3,class_mode='categorical')
labels=(training_set.class_indices)
print(labels)
classifier.fit(training_set,steps_per_epoch=20,epochs=10,validation_data=valid_set)
history=classifier.history
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','validation'],loc='upper left')
plt.subplot(1,2,2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','validation'],loc="upper left")
plt.tight_layout()
plt.savefig('training_history.png')
plt.show()
classifier_json=classifier.to_json()
with open("model1.json","w") as json_file:
    json_file.write(classifier_json)
classifier.save_weights("my_model.weights.h5")
classifier.save("model.h5")
print("saved model to disk")
"""import Cv2 
from matplotlib import pyplot as plt 
import os 
import numpy as np  
img=cv2.imread('leaf disease.jpg')
img_resize=cv2.resize(img,(128,128))
b,g,r=cv2.split(img_resize)
vgb_img=cv2.merge([r,g,b])
plt.imshow(rgb_img)
label_map=(training_set.class_indices)
print(label_map)
img_rank4=np.expand_dims(rgb_img/255,axis=0)
pred=classifier.predict(img_rank4)
pred_class=np.argmax(pred,axis=1)[0]
h=list(label_map.keys())[pred_class]
font=cv2.FONT_HERSHEY_DUPLEX
cv2.puttext(img,h,(10,30),font,1.0,(0,0,255),1)
cv2.imshow(h,img)
cv2.waitkey(0)
cv2.imshow()
print(h)"""