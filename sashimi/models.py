from django.db import models

# Create your models here.

class Category(models.Model):   
    class Meta:
        #テーブル名の指定
        db_table ="category"

    #カラム名の定義
    category_name = models.CharField(max_length=255,unique=True)
    content = models.CharField(max_length=1000,unique=True)
    
class Post(models.Model):
    class Meta:
        db_table = 'post'
  
    # カラム名
    image =
    category = models.ForeignKey(Category, on_delete = models.PROTECT, verbose_name="カテゴリ")
    predict = 