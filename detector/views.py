from nturl2path import url2pathname
from django.shortcuts import render
from detector.src import Inputscript, RandomForest, Index

# Create your views here.
def index(request):
    if(request.method == "POST"):
        print("Processing Request:", request.POST)

        data = request.POST
        url = data.get("url")
        result = Index.main(url)

        context = {
            "url": url,
            "result": result
        }

        return render(request, "detector/index.html", context)
    else:
        return render(request, "detector/index.html", {})