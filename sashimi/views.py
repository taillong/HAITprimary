from django.shortcuts import render
from django.views.generic import View
from .models import Post

class IndexView(View):
    def get(self, request, *args, **kwargs):
        post_data = Post.objects.order_by('-id')
        return render(request, 'sashimi/index.html'{
            'post_data':post_data
        })
