# importando bibliotecas

print("_______ Importanto bibliotecas _______")

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential, Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow_hub as hub

import cv2
import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.model_selection import train_test_split

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
)
import random


class TelesNet:
    def __init__(self, model="default"):
        self.buildModel(model)

    def setData(self, data, labels, numClasses, classes=None):
        self.X = np.asarray(data)
        self.y = np.asarray(labels)
        self.y = to_categorical(self.y, numClasses)

        if numClasses == None or numClasses < 1:
            raise Exception("Please provide the number of classes.")

        self.n_classes = numClasses

        if classes:
            self.classes = classes

    def buildModel(self, model):
        print(f"Versão: {model}")

        if model == "v1":
            self.model = self.v1()
        elif model == "v2":
            self.model = self.v2()
        elif model == "v3":
            self.model = self.v3()
        elif model == "default":
            self.classes = ["papel", "pedra", "tesoura"]
            self.model = tf.keras.models.load_model("./models/v1.h5")
        else:
            raise Exception(
                f"{model} is not a option, please select a version of the model: [v1,v2,v3]"
            )

        self.model.compile(
            optimizer="adam",
            loss=keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"],
        )

    def v1(self):
        width = self.X.shape[1]
        height = self.X.shape[2]

        model = Sequential(
            [
                layers.Input(shape=(width, height, self.n_classes)),
                layers.Conv2D(16, 3, padding="same", activation="relu"),
                layers.Conv2D(32, 3, padding="same", activation="relu"),
                layers.MaxPooling2D(),
                layers.Conv2D(64, 3, padding="same", activation="relu"),
                layers.MaxPooling2D(),
                layers.Flatten(),
                layers.Dense(128, activation="relu"),
                layers.Dense(self.n_classes, activation="softmax"),
            ]
        )

        return model

    def v2(self):
        width = self.X.shape[1]
        height = self.X.shape[2]

        input_layer = Sequential(
            [
                layers.Input(shape=(width, height, self.n_classes)),
                layers.Conv2D(16, 3, padding="same", activation="relu"),
                layers.Conv2D(16, 3, padding="same", activation="relu"),
                layers.Conv2D(32, 3, padding="same", activation="relu"),
                layers.MaxPooling2D(),
                layers.Conv2D(64, 3, padding="same", activation="relu"),
                layers.MaxPooling2D(),
            ]
        )

        x1 = layers.Conv2D(64, (1, 1), padding="same", activation="relu")(
            input_layer.output
        )

        x2 = layers.Conv2D(32, (1, 1), padding="same", activation="relu")(
            input_layer.output
        )
        x2 = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x2)

        x3 = layers.Conv2D(16, (1, 1), padding="same", activation="relu")(
            input_layer.output
        )
        x3 = layers.Conv2D(32, (5, 5), padding="same", activation="relu")(x3)

        o = layers.Concatenate(axis=3)([x1, x2, x3])

        output_layer = layers.Flatten()(o)
        output_layer = layers.Dense(128, activation="relu")(output_layer)
        output_layer = layers.Dense(self.n_classes, activation="softmax")(output_layer)

        model = Model(inputs=[input_layer.input], outputs=output_layer)

        return model

    def v3(self):
        width = self.X.shape[1]
        height = self.X.shape[2]

        # Load the EfficientNet model from TensorFlow Hub
        efficientnet_url = (
            "https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1"
        )
        efficientnet_model = hub.KerasLayer(
            efficientnet_url, trainable=False, input_shape=(*(width, height), 3)
        )

        model = Sequential(
            [efficientnet_model, layers.Dense(self.n_classes, activation="softmax")]
        )

        return model

    def summary(self):
        self.model.summary()

    def validaTreino(self):
        if self.x == None or self.y == None:
            raise Exception(
                "There's no data or the data is incomplete. Please set the data using the setData() method."
            )

    def train(self, perc_treino=0.8, batch_size=32, epochs=20, save=False):
        self.validaTreino()

        # a) divide a base aleatoriamente
        x_train, x_test, y_train, y_test = train_test_split(
            self.X, self.y, train_size=perc_treino, random_state=random.randint(1, 1000)
        )

        print("_______ Quantidade de dados _______")
        print("treino -> x:", len(x_train), "y:", len(y_train))
        print("validação -> x:", len(x_test), "y:", len(y_test))

        x_train, x_val, y_train, y_val = train_test_split(
            x_train,
            y_train,
            train_size=perc_treino,
            random_state=random.randint(1, 1000),
        )

        # Data aumentation
        datagen = ImageDataGenerator(
            brightness_range=(0.7, 1.3),
        )

        if save == True:
            # Criando chekpoint callback
            checkpoint = ModelCheckpoint(
                "/content/drive/MyDrive/visao_comp/best_model.h5",
                monitor="val_accuracy",
                verbose=1,
                save_best_only=True,
                save_weights_only=True,
            )

            # b) treina
            H = self.model.fit(
                datagen.flow(x_train, y_train, batch_size=batch_size),
                epochs=epochs,
                callbacks=[checkpoint],  # plotando gráficos
                validation_data=(x_val, y_val),
            )

        else:
            # b) treina
            H = self.model.fit(
                datagen.flow(x_train, y_train, batch_size=batch_size),
                epochs=epochs,
                validation_data=(x_val, y_val),
            )

        # c) predição
        y_pred = self.model.predict(x_test)
        y_pred = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_test, axis=1)

        print("Acuracia:", accuracy_score(y_true, y_pred))
        print("Precision:", precision_score(y_true, y_pred, average="weighted"))
        print("Recall:", recall_score(y_true, y_pred, average="weighted"))

    def play(self):
        self.playing = False

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Erro ao abrir a webcam!")
            exit(1)

        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()

            # Verificar se o frame foi capturado
            if ret:
                # Exibir o frame
                cv2.imshow("Webcam", frame)

                # Sair do loop se a tecla 'q' for pressionada
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                if cv2.waitKey(1) & 0xFF == ord("c"):
                    if self.playing == True:
                        return

                    self.playing = True
                    self.pred(frame)
            else:
                print("Erro ao capturar o frame!")
                break

        # Liberar a captura da webcam
        cap.release()

        # Fechar todas as janelas abertas
        cv2.destroyAllWindows()

    def pred(self, frame):
        print("Predicting...")

        img = cv2.resize(frame, (300, 200))

        teste = np.asarray([img])

        pred = self.model.predict(teste)

        result = self.classes[np.argmax(pred)]

        print(f"result: {result}")

        self.showResult(frame, result)

        self.playing = False

    def showResult(self, frame, text):
        # Define the font and text color
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_size = 1.0
        font_color = (255, 0, 0)  # White

        # Get the text size and offset
        text_size, baseline = cv2.getTextSize(text, font, font_size, thickness=1)
        text_offset_x = 10
        text_offset_y = frame.shape[0] - baseline - 10

        # Add text to the image
        cv2.putText(
            frame,
            text,
            (text_offset_x, text_offset_y),
            font,
            font_size,
            font_color,
            thickness=1,
        )

        # Display the image
        cv2.imshow("Result", frame)
        cv2.waitKey(0)
