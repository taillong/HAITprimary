from django.shortcuts import render
import sys
from HAITprimary.settings import MODEL_FILE_PATH
from PIL import Image
import numpy as np
import cv2

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
import keras
from tensorflow.keras import regularizers
import tensorflow as tf

# Pillow->opencv変換
def pil2cv(image):
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = new_image[:, :, ::-1]
    elif new_image.shape[2] == 4:  # 透過
        new_image = new_image[:, :, [2, 1, 0, 3]]
    return new_image

# モデルを復元する関数
def build_model(path):
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
    classifier = "５種類（マグロ, サーモン, カツオ, ぶり, アジ刺し身)"
    classes = ["マグロ", "サーモン", "カツオ", "ぶり", "アジ刺し身"]
    # formから画像ファイルを取得
    file = request.FILES.get("file")
    # 画像を整形
    image = Image.open(file)
    image = image.convert("RGB")
    image = image.resize((100, 100))
    data = np.asarray(image)
    data = data.astype(np.float32)
    X = []
    X.append(data)
    X = np.array(X)
    # モデルの復元
    model = build_model(MODEL_FILE_PATH + "/maguro_salmon_katsuo_buri_azi_shape100.h5")
    # # 予測結果を出力
    result = model.predict([X])[0]
    predicted = result.argmax()
    percentage = int(result[predicted] * 100)
    kind = classes[predicted]
    # テンプレートに結果を渡す
    return render(  
        request,
        "sashimi/result.html",
        {"classifier":classifier, "kind":kind, "percentage":percentage}
    )

def shiromi_result(request):
    classifier = "カレイvsヒラメ"
    # 分類したいもの
    classes = ["カレイ", "ヒラメ"]
    # formから画像ファイルを取得
    file = request.FILES.get("file")
    # 画像を整形
    image = Image.open(file)
    image = pil2cv(image)
    image = np.asarray(image)
    image = image.astype(np.float32)   
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (250,250))

    image = cv2.resize(image, None, fx=2.0, fy=2.0)
    height, width, channel = image.shape
    x =  int(height/4)
    y =  int(width/4)
    image = image[y:3*y, x:3*x]
    
    for i in range(len(image)):
      image[i] -= image[i].min()
      image[i] /= image[i].max()

    img = []
    img.append(image)
    img = np.array(img)
    num_classes = 2


    # モデルの復元
    model = Sequential()
    model.add(Conv2D(filters=32, input_shape=(250,250,3), kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2,2)))


    model.add(Conv2D(filters=32, kernel_size=(4, 4), strides=(1, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.5))


    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.5))


    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(2))
    model.add(Activation('softmax'))   

    # モデルの最適化を宣言(loss:損失関数、metrics:評価の値(今回は正答数))
    #opt = keras.optimizers.Adam(lr=0.0001, decay=1e-6)

    model.compile(loss="categorical_crossentropy", optimizer="Adam", metrics=["accuracy"])
    # モデルのロード
    # efficientnetを使った時はcustomobjectsを入れないと無理っぽい


    model = build_model(MODEL_FILE_PATH + "/Hirame_Karei_CNN.h5")
    # 結果を出力
    result = model.predict([img])[0]
    predicted = result.argmax()
    percentage = int(result[predicted] * 100)
    kind = classes[predicted]



    # テンプレートに結果を渡す
    return render(  
        request,
        "sashimi/result.html",
        {"classifier":classifier, "kind":kind, "percentage":percentage}
    )

def ao_result(request):
    import keras.backend.tensorflow_backend as tb
    tb._SYMBOLIC_SCOPE.value = True
    # 分類したいもの
    classifier = "アジvsイワシ"
    classes = ["アジ刺し身", "イワシ刺し身"]
    # formから画像ファイルを取得
    file = request.FILES.get("file")
    # 画像を整形
    image = Image.open(file)
    image = image.convert("RGB")
    image = image.resize((300, 300))
    data = np.asarray(image)
    data = data.astype(np.float32)
    X = []
    X.append(data)
    X = np.array(X)
    # モデルの復元
    model = build_model(MODEL_FILE_PATH + "/azi_iwashi_downgraded_shape300_70.h5")
    # # 予測結果を出力
    result = model.predict([X])[0]
    predicted = result.argmax()
    percentage = int(result[predicted] * 100)
    kind = classes[predicted]

    # テンプレートに結果を渡す
    return render(  
        request,
        "sashimi/result.html",
        {"classifier":classifier, "kind":kind, "percentage":percentage}
    )
