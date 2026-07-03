
import os

from keras.layers import Input, Lambda, Dense, Flatten, Dropout
from keras.models import Model
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.models import load_model

import numpy as np

import os
import cv2
import matplotlib.pyplot as plt
import tkinter.filedialog
from tkinter import *
from PIL import Image
from PIL import ImageTk
from PIL import ImageOps
import time

IMAGE_SIZE = [224, 224]

train_path = "dataset/train"
test_path = "dataset/test"
val_path = "dataset/val"

x_train = []

for folder in os.listdir(train_path):
    sub_path = train_path+"/"+folder
    for img in os.listdir(sub_path):
        image_path = sub_path+"/"+img
        img_arr = cv2.imread(image_path)
        img_arr = cv2.resize(img_arr, (224, 224))
        x_train.append(img_arr)


x_test = []


for folder in os.listdir(test_path):
    sub_path = test_path+"/"+folder
    for img in os.listdir(sub_path):
        image_path = sub_path+"/"+img
        img_arr = cv2.imread(image_path)
        img_arr = cv2.resize(img_arr, (224, 224))
        x_test.append(img_arr)


x_val = []

for folder in os.listdir(val_path):
    sub_path = val_path+"/"+folder
    for img in os.listdir(sub_path):
        image_path = sub_path+"/"+img
        img_arr = cv2.imread(image_path)
        img_arr = cv2.resize(img_arr, (224, 224))
        x_val.append(img_arr)


train_x = np.array(x_train)
test_x = np.array(x_test)
val_x = np.array(x_val)

train_x.shape, test_x.shape, val_x.shape

train_x = train_x/255.0
test_x = test_x/255.0
val_x = val_x/255.0


train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)


training_set = train_datagen.flow_from_directory(train_path,
                                                 target_size=(224, 224),
                                                 batch_size=32,
                                                 class_mode='sparse')

test_set = test_datagen.flow_from_directory(test_path,
                                            target_size=(224, 224),
                                            batch_size=32,
                                            class_mode='sparse')

val_set = val_datagen.flow_from_directory(val_path,
                                          target_size=(224, 224),
                                          batch_size=32,
                                          class_mode='sparse')

print(training_set.class_indices)

train_y = training_set.classes

test_y = test_set.classes

val_y = val_set.classes

print(train_y.shape, test_y.shape, val_y.shape)
vgg = VGG19(input_shape=IMAGE_SIZE + [3],
            weights='imagenet', include_top=False)


for layer in vgg.layers:
    layer.trainable = False
x = Flatten()(vgg.output)

prediction = Dense(8, activation='softmax')(x)
model = Model(inputs=vgg.input, outputs=prediction)
model.summary()


def load_dataset():

    status_label.config(text="Loading Dataset\t\t\t\t")
    root.update_idletasks()
    time.sleep(2)
    status_label.config(text="Dataset loaded successfully")


def preprocess():
    status_label.config(text="Preprocessing\t\t\t\t")
    root.update_idletasks()
    time.sleep(2)
    status_label.config(text="Image resized successfully")


def train():
    status_label.config(text="Training...\t\t\t\t")
    root.update_idletasks()
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer="adam",
        metrics=['accuracy']
    )
    early_stop = EarlyStopping(
        monitor='val_loss', mode='min', verbose=1, patience=5)

    history = model.fit(
        train_x,
        train_y,
        validation_data=(val_x, val_y),
        epochs=20,
        callbacks=[early_stop],
        batch_size=2, shuffle=True)
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.legend()

    plt.savefig('vgg-loss-rps-1.png')
    plt.show()

    plt.plot(history.history['accuracy'], label='train acc')
    plt.plot(history.history['val_accuracy'], label='val acc')
    plt.legend()

    plt.savefig('vgg-acc-rps-1.png')
    plt.show()

    model.evaluate(test_x, test_y, batch_size=32)

    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    import numpy as np

    y_pred = model.predict(test_x)
    y_pred = np.argmax(y_pred, axis=1)

    accuracy_score(y_pred, test_y)

    print(classification_report(y_pred, test_y))

    confusion_matrix(y_pred, test_y)

    model.save("vgg-rps-final.h5")

    # plt.show()

    status_label.config(text="Trained Successfully\t\t\t\t")


def show_loss():
    global panelA
    image = cv2.imread("vgg-loss-rps-1.png")
    im2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    im2 = Image.fromarray(im2)
    resized_image = im2.resize((400, 305), Image.ANTIALIAS)
    im2 = ImageTk.PhotoImage(resized_image)
    if panelA is None:
        panelA = Label(image=im2)
        panelA.image = im2
        # panelA.pack(side="left", padx=10, pady=10)
        # panelA.grid(row=1,column=2)
        panelA.place(x=430, y=150)

    else:
        panelA.configure(image=im2)
        panelA.image = im2
    status_label.config(text="Loss Graph\t\t\t\t")


def show_acc():
    global panelA
    image = cv2.imread("vgg-acc-rps-1.png")
    im2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    im2 = Image.fromarray(im2)
    resized_image = im2.resize((400, 305), Image.ANTIALIAS)
    im2 = ImageTk.PhotoImage(resized_image)
    if panelA is None:
        panelA = Label(image=im2)
        panelA.image = im2
        # panelA.pack(side="left", padx=10, pady=10)
        # panelA.grid(row=1,column=2)
        panelA.place(x=430, y=150)

    else:
        panelA.configure(image=im2)
        panelA.image = im2
    status_label.config(text="Accuracy Graph\t\t\t\t")


def detection():
    path = tkinter.filedialog.askopenfilename()

    # ensure a file path was selected
    if len(path) > 0:
        model = load_model('keras_model_plant.h5')

        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        image = Image.open(path)
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.ANTIALIAS)

        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        data[0] = normalized_image_array

        prediction = model.predict(data)
        y_classes = prediction.argmax(axis=-1)
        accu = prediction[0][y_classes]*100

        if y_classes == 0:
            val = "Corn_(maize)___Common_rust \nUses: everything"
            print("Corn_(maize)___Common_rust_ with accuracy: ", accu)
        elif y_classes == 1:
            val = "Amaranthus"
            print("Corn_(maize)___healthy with accuracy: ", accu)
        elif y_classes == 2:
            val = "Artocarpus"
            print("Artocarpus with accuracy: ", accu)
        elif y_classes == 3:
            val = "Azadirachta"
            print("Azadirachta viridis with accuracy: ", accu)
        elif y_classes == 4:
            val = "Basella Alba viridis"
            print("Basella Alba viridis with accuracy: ", accu)
        else:
            val = "Brassica Juncea viridis"
            print("Brassica Juncea viridis with accuracy: ", accu)

        global panelA
        image = cv2.imread(path)
        im2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        im2 = Image.fromarray(im2)
        resized_image = im2.resize((400, 305), Image.ANTIALIAS)
        im2 = ImageTk.PhotoImage(resized_image)
        if panelA is None:
            panelA = Label(image=im2)
            panelA.image = im2
            # panelA.pack(side="left", padx=10, pady=10)
            # panelA.grid(row=1,column=2)
            panelA.place(x=430, y=150)

        else:
            panelA.configure(image=im2)
            panelA.image = im2
        status_label.config(text="Accuracy: "+str(accu)+"\nPredicted: "+val)


root = Tk()
root.geometry("980x850")

root.title("Medicinal Leaf Detection")
# root.configure(bg="#DAF7A6")
image = Image.open("b.jpg")
# Resize the image if desired
image = image.resize((1400, 700), Image.ANTIALIAS)
tk_image = ImageTk.PhotoImage(image)

label = Label(root, image=tk_image)
label.pack()
l1 = Label(root, text="Medicinal Leaf Detection", font=(
    "Courier,", 18, "italic", "bold"), fg="Black")
l1.place(x=450, y=25, anchor="center")

l2 = Label(root, text=">>> MENU <<<", font=(
    "Courier", 14, "italic", "bold"), fg="Black")
l2.place(x=60, y=100)

Button(root, text="Load Dataset", command=load_dataset, bg="#f9a655", font=(
    "Courier", 14, "italic", "bold"), height=1, width=15).place(x=50, y=150)

Button(root, text="Preprocessing", bg="#f9a655", font=("Courier", 14, "italic",
       "bold"), height=1, width=15, command=preprocess).place(x=50, y=250)

Button(root, text="Train", command=train, bg="#f9a655", font=(
    "Courier", 14, "italic", "bold"), height=1, width=15).place(x=50, y=350)

Button(root, text="Loss Graph", command=show_loss, bg="#f9a655", font=(
    "Courier", 14, "italic", "bold"), height=1, width=15).place(x=50, y=450)

Button(root, text="Accuracy Graph", command=show_acc, bg="#f9a655", font=(
    "Courier", 14, "italic", "bold"), height=1, width=15).place(x=50, y=550)

Button(root, text="Browse Image", command=detection, bg="#f9a655", font=(
    "Courier", 14, "italic", "bold"), height=1, width=15).place(x=50, y=650)

status_label = Label(root, font=("Courier", 14, "italic", "bold"), fg="Black")
status_label.pack()
status_label.place(x=400, y=600)

panelA = None

root.mainloop()
