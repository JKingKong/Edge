from django.db import models

# Create your models here.
# 被监控的地点
class Address(models.Model):
    aid = models.AutoField(primary_key=True)                # 地址id
    address = models.CharField(max_length=200,unique=True)  # 地点名字
    threshold = models.IntegerField(default=5)              # 地点的阈值
