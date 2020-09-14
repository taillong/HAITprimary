from django.shortcuts import render
import sys
from HAITprimary.settings import MODEL_FILE_PATH
from PIL import Image
import numpy as np

from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import np_utils
import keras
from keras import regularizers
# AttributeError: '_thread._local' object has no attribute 'value'を消すため

# kerasで組むと
# from keras.models import Sequential, load_model
# from keras.layers import Conv2D, MaxPool2D
# from keras.layers import Activation, Dropout, Flatten, Dense
# from keras.utils import np_utils
# import keras
# from keras import regularizers
# # AttributeError: '_thread._local' object has no attribute 'value'を消すため
# import keras.backend.tensorflow_backend as tb
# tb._SYMBOLIC_SCOPE.value = True
# import tensorflow_hub as hub


# tensorflowで組むと
# from tensorflow.keras.models import Sequential, load_model
# from tensorflow.keras.layers import Conv2D, MaxPool2D
# from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
# from tensorflow.keras import utils 
# import tensorflow as tf
# from tensorflow.keras import regularizers
# import tensorflow_hub as hub
# from tensorflow.keras import layers



# # モデルの復元
# def build_model():
#     # modelの定義
#     model = Sequential()

#     model.add(Conv2D(filters=32, input_shape=(100, 100, 3), kernel_size=(3, 3),kernel_regularizer=regularizers.l2(0.001), strides=(1, 1), padding='same'))
#     model.add(Activation('relu'))
#     model.add(MaxPool2D(pool_size=(2,2)))


#     model.add(Conv2D(filters=32, kernel_size=(4, 4), strides=(1, 1), kernel_regularizer=regularizers.l2(0.001), padding='same'))
#     model.add(Activation('relu'))
#     model.add(MaxPool2D(pool_size=(2,2)))
#     model.add(Dropout(0.5))


#     model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),kernel_regularizer=regularizers.l2(0.001), padding='same'))
#     model.add(Activation('relu'))
#     model.add(MaxPool2D(pool_size=(2,2)))
#     model.add(Dropout(0.5))


#     model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),kernel_regularizer=regularizers.l2(0.001), padding='same'))
#     model.add(Activation('relu'))
#     model.add(MaxPool2D(pool_size=(2,2)))
#     model.add(Dropout(0.25))

#     model.add(Flatten())
#     model.add(Dense(512, kernel_regularizer=regularizers.l2(0.001)))
#     model.add(Activation('relu'))
#     model.add(Dense(5))
#     model.add(Activation('softmax'))

#     # 最適化の手法を宣言
#     opt = tf.keras.optimizers.Adam(lr=0.0001, decay=1e-6)
#     # opt = keras.optimizers.Adam(lr=0.0001, decay=1e-6)

#     # モデルの最適化を宣言(loss:損失関数、metrics:評価の値(今回は正答数))
#     model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

#     # モデルのロード
#     model = load_model(MODEL_FILE_PATH, custom_objects={'L2':regularizers.l2()})

#     return model

# # ここでエラーが出る
# model = build_model()


def build_model(path):
    # modelの定義
    model = Sequential()
    model.add(Conv2D(filters=32, input_shape=(300,300,3), kernel_size=(3, 3),kernel_regularizer=regularizers.l2(0.001), strides=(1, 1), padding='same'))
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
    model.add(Dense(2))
    model.add(Activation('softmax'))
    # 最適化の手法を宣言
    opt = keras.optimizers.Adam(lr=0.0001, decay=1e-6)
    # earlystopping
    from keras.callbacks import EarlyStopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1,mode='auto')
    # モデルの最適化を宣言(loss:損失関数、metrics:評価の値(今回は正答数))
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    # モデルのロード
    # efficientnetを使った時はcustomobjectsを入れないと無理っぽい
    model = load_model(path) #, custom_objects={'KerasLayer':hub.KerasLayer})

    return model

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
    import keras.backend.tensorflow_backend as tb
    tb._SYMBOLIC_SCOPE.value = True


    # formから画像ファイルを取得
    file = request.FILES.get("file")

    # 画像を整形
    image = Image.open(file)
    image = image.convert("RGB")
    image = image.resize((300, 300))
    data = np.asarray(image)
    X = []
    X.append(data)
    X = np.array(X)

    # モデルの復元
    model = build_model(MODEL_FILE_PATH + "/azi_iwashi_downgraded_shape300_70.h5")

    result = model.predict([X])[0]


    return render(
        request,
        "sashimi/result.html",
        {"file":result}
    )
