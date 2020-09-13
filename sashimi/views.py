from django.shortcuts import render


def index(request):
    if request.method == "GET":
        return render(
            request,
            "sashimi/home.html",
        )


def result(request):
    file = request.POST["file"]
    

    return render(
        request,
        "sashimi/result.html",
        {"file":file}
    )
