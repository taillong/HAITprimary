from django.shortcuts import render
from PIL import Image
import numpy as np
import sys

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras import utils
from tensorflow.keras import regularizers


# モデルの復元
model = load_model("maguro_salmon_katsuo_buri_azi_shape100_81.h5")

# トップページ（仮）
def index(request):
    if request.method == "GET":
        return render(
            request,
            "sashimi/home.html",
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

    result = model.predict([X])[0]


    return render(
        request,
        "sashimi/result.html",
        {"file":X}
    )
