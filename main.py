from flask import Flask, render_template, url_for, request, json, redirect
from werkzeug.utils import secure_filename
import os

import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
# from keras.optimizers import Adam
from keras.layers import MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import precision_score, classification_report, confusion_matrix


app = Flask(__name__)

@app.route("/")
@app.route("/home")
def  home():
    return render_template('index.html')

@app.route("/nhandien", methods=["POST"])
def  nhandien():
    UPLOAD_FOLDER = 'static/anh/'
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

    if request.method == 'POST':
        hinhanh = request.files['anh_nd']
        # return str(hinhanh)
        img_name = secure_filename(hinhanh.filename)
        # return img_name
        saved_path = os.path.join(app.config['UPLOAD_FOLDER'], img_name)
        app.logger.info("saving {}".format(saved_path))
        hinhanh.save(saved_path)

        return kiemtra(saved_path, img_name)
        # return render_template('index.html')


# @app.route("/kiemtra")
def  kiemtra(path, name):
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
    model.load_weights('model\model.h5')
    cv2.ocl.setUseOpenCL(False)


    dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral",
                5: "Sad", 6: "Surprised"}

    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    image = cv2.imread(path)
    print(path)
    image = cv2.resize(image, (500,500))

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # print(type(gray_image))

    faces=face_cascade.detectMultiScale(gray_image,scaleFactor=1.10,minNeighbors=6)
    # print(faces)
    for x,y,w,h in faces:
        cv2.rectangle(image, (x,y), (x+w, y+h), (0, 255, 0),1)
        crop_img = gray_image[y:y+h, x:x+w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(crop_img, (48, 48)), -1), 0)

        float_formatter = "{:.4f}".format
        np.set_printoptions(formatter={'float_kind':float_formatter})

        emotion_prediction = model.predict(cropped_img)
        print(emotion_prediction)
        maxindex = int(np.argmax(emotion_prediction))
        print(dict[maxindex])
        cv2.putText(image, dict[maxindex], (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 0), 2, cv2.LINE_AA)
        cv2.imshow("Face Detector", image)
        k=cv2.waitKey(2000)
    cv2.destroyAllWindows()
    cv2.imwrite("static/anh_danhandang/"+name, image)
    # return "static/anh_t/"+name
    return render_template('index.html', image="static/anh_danhandang/"+name)

if __name__ == "__main__":
    app.run(debug=True)
