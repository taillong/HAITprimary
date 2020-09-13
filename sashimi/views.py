from django.shortcuts import render
from PIL import Image

def index(request):
    if request.method == "GET":
        return render(
            request,
            "sashimi/home.html",
        )


def result(request):
    file = request.POST["file"]

    # 必要になるであろう処理
    # image = Image.open(file)
    # image = image.convert("RGB")
    # image = image.resize((100, 100))
    # data = np.asarray(image)
    # X = []
    # X.append(data)
    # X = np.array(X)


    return render(
        request,
        "sashimi/result.html",
        {"file":file}
    )
