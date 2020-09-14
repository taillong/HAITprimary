from django.shortcuts import render
import sys

# kerasで組むと
# from keras.models import Sequential, load_model
# from keras.layers import Conv2D, MaxPool2D
# from keras.layers import Activation, Dropout, Flatten, Dense
# from keras.utils import np_utils
# import keras
# import numpy as np
# import sys
# from PIL import Image
# from keras import regularizers
# AttributeError: '_thread._local' object has no attribute 'value'を消すため
# import keras.backend.tensorflow_backend as tb
# tb._SYMBOLIC_SCOPE.value = True


# tensorflowで組むと
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras import utils 
import tensorflow as tf
import numpy as np
import sys
from PIL import Image
from tensorflow.keras import regularizers



# モデルの復元
def build_model():
    # modelの定義
    model = Sequential()

    model.add(Conv2D(filters=32, input_shape=(100, 100, 3), kernel_size=(3, 3),kernel_regularizer=regularizers.l2(0.001), strides=(1, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2,2)))


    model.add(Conv2D(filters=32, kernel_size=(4, 4), strides=(1, 1), kernel_regularizer=regularizers.l2(0.001), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.5))


    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),kernel_regularizer=regularizers.l2(0.001), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.5))


    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),kernel_regularizer=regularizers.l2(0.001), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, kernel_regularizer=regularizers.l2(0.001)))
    model.add(Activation('relu'))
    model.add(Dense(5))
    model.add(Activation('softmax'))

    # 最適化の手法を宣言
    opt = tf.keras.optimizers.Adam(lr=0.0001, decay=1e-6)

    # モデルの最適化を宣言(loss:損失関数、metrics:評価の値(今回は正答数))
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    # モデルのロード
    model = load_model("/Users/taillong/Documents/programming/python/django/sashimiapp/HAITprimary/sashimi/maguro_salmon_katsuo_buri_azi_shape100_81.h5", custom_objects={'L2':regularizers.l2(0.001)}, compile=False)

    return model

# ここでエラーが出る
# model = build_model()

# トップページ（仮）
def index(request):
    if request.method == "GET":
        return render(
            request,
            "sashimi/home.html",
        )


# file入力ページ
def five(request):
    if request.method == "GET":
        return render(
            request,
            "sashimi/five.html",
        )

def shiromi(request):
    if request.method == "GET":
        return render(
            request,
            "sashimi/shiromi.html",
        )

def ao(request):
    if request.method == "GET":
        return render(
            request,
            "sashimi/ao.html",
        )


def result(request):
    # formから画像ファイルを取得
    file = request.FILES.get("file")

    # 画像を整形
    image = Image.open(file)
    image = image.convert("RGB")
    image = image.resize((100, 100))
    data = np.asarray(image)
    X = []
    X.append(data)
    X = np.array(X)

    # result = model.predict([X])[0]


    return render(
        request,
        "sashimi/result.html",
        {"file":X}
    )
