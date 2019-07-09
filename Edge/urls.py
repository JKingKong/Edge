from django.conf.urls import url, include
from django.contrib import admin
from edgeapp.views import *




urlpatterns = [
    url(r'^admin/', admin.site.urls),
    url(r'^address_list/', getAddressList),
    url(r'^update_threshold/', updateThreshold),            # 根据地址更新阈值
    url(r'^start_recognize/', startRecognize),
]
