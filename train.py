import numpy as np
import cv2
from matplotlib import pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
# from keras.optimizers import Adam
from keras.layers import MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import adam_v2 
 

train_dir = 'train'  
test_dir = 'test'

# Điều chỉnh giá trị pixel trong khoảng 0-1 (đầu vào CNN là matrix trong khoảng [0-1])
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48,48),
        batch_size=64,              # Số lượng ảnh mỗi lần đưa vào 
        color_mode="grayscale",
        class_mode='categorical')   # sử dụng nhiều hơn 2 lớp
        
test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(48,48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')

print(train_generator)
label_train = (train_generator.class_indices)
print(label_train.items())

print(test_generator)
label_test = (test_generator.class_indices)
print(label_test.items())

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))  
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2)) 

model.add(Dense(7, activation='softmax'))

# Lớp Conv2D: param_number = output_channel_number * (input_channel_number * kernel_height * kernel_width + 1)
# Lớp Dense: param_number = output_channel_number * (input_channel_number + 1)
model.summary() 

model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.001, decay=1e-6),metrics=['accuracy'])
model_info = model.fit_generator(
        train_generator,
        steps_per_epoch=28709 // 64,     
        epochs=500, 
        validation_data=test_generator,
        validation_steps= 7178 // 64)
model.save_weights('model\model.h5')

import matplotlib.pyplot as plt    
# Vẽ biểu đồ fit model  
keys=model_info.history.keys()
print(keys)  

def show_train_history(hisData,train,test):  
    plt.plot(hisData.history[train])
    plt.plot(hisData.history[test])
    plt.title('Training History') 
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

show_train_history(model_info, 'loss', 'val_loss')
show_train_history(model_info, 'accuracy', 'val_accuracy')
from sklearn.metrics import classification_report, confusion_matrix
 
# Độ chính xác 
train_loss, train_accu = model.evaluate(train_generator)
test_loss, test_accu = model.evaluate(val_generator)
print("final train accuracy = {:.2f} , validation accuracy = {:.2f}".format(train_accu*100, test_accu*100))