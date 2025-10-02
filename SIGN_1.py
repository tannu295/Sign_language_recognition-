import matplotlib.pyplot as plt
import numpy as np
import cv2

import os
import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping


import random

seed = 4
tf.random.set_seed(seed)
np.random.seed(seed)
random.seed(seed)

data_dir=r"C:\Users\Tannu Panwar\OneDrive\Desktop\SIGN RECOGINATION\code\SIGN_DATA\asl_dataset"

import pathlib
data_dir=pathlib.Path(data_dir)
print(data_dir)


image_count=len(list(data_dir.glob('*/*.jpeg')))
print(image_count)


A=list(data_dir.glob('A/*'))
print(A[:5])



# from PIL import Image
# img= Image.open(str(A[0]))
# img.show()


SIGN_PATH={
    'A':list(data_dir.glob('A/*')),
    'B':list(data_dir.glob('B/*')),
    'C':list(data_dir.glob('C/*')),
    'D':list(data_dir.glob('D/*')),
    'E':list(data_dir.glob('E/*')),
    'F':list(data_dir.glob('F/*')),
    'G':list(data_dir.glob('G/*')),
    'H':list(data_dir.glob('H/*')),
    'I':list(data_dir.glob('I/*')),
    'J':list(data_dir.glob('J/*')),
    'K':list(data_dir.glob('K/*')),
    'L':list(data_dir.glob('L/*')),
    'M':list(data_dir.glob('M/*')),
    'N':list(data_dir.glob('N/*')),
    'O':list(data_dir.glob('O/*')),
    'P':list(data_dir.glob('P/*')),
    'Q':list(data_dir.glob('Q/*')),
    'R':list(data_dir.glob('R/*')),
    'S':list(data_dir.glob('S/*')),
    'T':list(data_dir.glob('T/*')),
    'U':list(data_dir.glob('U/*')),
    'V':list(data_dir.glob('V/*')),
    'W':list(data_dir.glob('W/*')),
    'X':list(data_dir.glob('X/*')),
    'Y':list(data_dir.glob('Y/*')),
    'Z':list(data_dir.glob('Z/*')),
}



SIGN_LABELS={
    'A':0,
    'B':1,
    'C':2,
    'D':3,
    'E':4,
    'F':5,
    'G':6,
    'H':7,
    'I':8,
    'J':9,
    'K':10,
    'L':11,
    'M':12,
    'N':13,
    'O':14,
    'P':15,
    'Q':16,
    'R':17,
    'S':18,
    'T':19,
    'U':20,
    'V':21,
    'W':22,
    'X':23,
    'Y':24,
    'Z':25,
}



print(str(SIGN_PATH['A'][0]))



img=cv2.imread(str(SIGN_PATH['A'][0]))
print(img.shape)

resize_ = cv2.resize(img,(180,180)).shape
print(resize_)

X,y =[],[]
for name,images in SIGN_PATH.items():
    for image in images:
        img=cv2.imread(str(image))
        resized_image=cv2.resize(img,(180,180))
        X.append(resized_image)
        y.append(SIGN_LABELS[name])



X=np.array(X)
print(X.shape)

y=np.array(y)
print(y.shape)


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=0)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)






X_train_scaled = X_train/255
X_test_scaled = X_test/255


num_classes=26
cnn = Sequential([
    #cnn
    layers.Conv2D(16 ,3 , padding='same' , activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32 ,3 , padding='same' , activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64 ,3 , padding='same' , activation='relu'),
    layers.MaxPooling2D(),

    #dense network
    layers.Flatten(),
    layers.Dense(120,activation='relu'),
    layers.Dense(num_classes)
])
cnn.compile(optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
            
             )


print(cnn.fit(X_train_scaled,y_train, validation_split = 0.2, epochs=15))

print(cnn.evaluate(X_test_scaled,y_test))




data_augmentation=keras.Sequential([
   
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.3),
    
])



num_classes=26
model = Sequential([
    data_augmentation,
    #cnn
    layers.Conv2D(16 ,3 , padding='same' , activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32 ,3 , padding='same' , activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64 ,3 , padding='same' , activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),

    #dense network
    layers.Flatten(),
    layers.Dense(120,activation='relu'),
    layers.Dense(num_classes)
])


model.compile(optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
            
             )

print(model.fit(X_train_scaled,y_train,validation_split = 0.2,epochs=15))

model.save("SIGN_MODEL.keras")
print("model is saved as SIGN_MODEL")
print(model.evaluate(X_test_scaled,y_test))


y_pred=model.predict(X_test)
print(y_pred[:5])

score=tf.nn.softmax(y_pred[30])
print(score)

print(np.argmax(score))

print(y_test[30])

