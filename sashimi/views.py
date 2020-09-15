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


# モデルを復元する関数
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
    model = load_model(path) 
    return model

# トップページ
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

# 結果をテンプレートに渡す関数
def five_result(request):
    # 分類したいもの
    classes = ["マグロ", "サーモン", "カツオ", "ぶり", "アジ刺し身"]
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
    # model = build_model(MODEL_FILE_PATH + "/azi_iwashi_downgraded_shape300_70.h5")
    # # 予測結果をresultに
    # result = model.predict([X])[0]
    # predicted = result.argmax()
    # percentage = int(result[predicted] * 100)
    # kind = classes[predicted]
    classifier = "５種類（マグロ, サーモン, カツオ, ぶり, アジ刺し身)"
    kind = 1
    percentage = 2
    # テンプレートに結果を渡す
    return render(  
        request,
        "sashimi/result.html",
        {"classifier":classifier, "kind":kind, "percentage":percentage}
    )

def shiromi_result(request):
    # 分類したいもの
    classes = ["カレイ", "ヒラメ"]
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
    # model = build_model(MODEL_FILE_PATH + "/azi_iwashi_downgraded_shape300_70.h5")
    # # 予測結果をresultに
    # result = model.predict([X])[0]
    # predicted = result.argmax()
    # percentage = int(result[predicted] * 100)
    # kind = classes[predicted]
    classifier = "カレイvsヒラメ"
    kind = 1
    percentage = 2
    # テンプレートに結果を渡す
    return render(  
        request,
        "sashimi/result.html",
        {"classifier":classifier, "kind":kind, "percentage":percentage}
    )

def ao_result(request):
    # 分類したいもの
    classes = ["アジ刺し身", "イワシ刺し身"]
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
    # model = build_model(MODEL_FILE_PATH + "/azi_iwashi_downgraded_shape300_70.h5")
    # # 予測結果をresultに
    # result = model.predict([X])[0]
    # predicted = result.argmax()
    # percentage = int(result[predicted] * 100)
    # kind = classes[predicted]
    classifier = "アジvsイワシ"
    kind = 1
    percentage = 2
    # テンプレートに結果を渡す
    return render(  
        request,
        "sashimi/result.html",
        {"classifier":classifier, "kind":kind, "percentage":percentage}
    )
