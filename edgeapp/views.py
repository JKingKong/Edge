from __future__ import division, print_function, absolute_import

from django.http.response import JsonResponse
from edgeapp.models import Address


# from Edge import demo
def getAddressList(request):
    '''
     获取地址列表对象
     :param request:
     :return:
     '''
    addr = Address.objects.all()
    address_list = []
    for item in addr:
        dic = {
            "address": item.address,
            "threshold": item.threshold
        }
        address_list.append(dic)
    jsonData = JsonResponse(
        {
            "address_list": address_list
        })
    return jsonData


def updateThreshold(request):
    address = request.GET.get('address', default='Test_A')
    threshold = request.GET.get('threshold', default=5)
    addr = Address.objects.filter(address=address).first()
    addr.threshold = threshold
    addr.save()
    return JsonResponse({
        "threshold": threshold,
        "address": address
    })


from timeit import time
import warnings
import cv2
import numpy as np
from PIL import Image
from edgeapp.yolo import YOLO

from edgeapp.deep_sort import preprocessing
from edgeapp.deep_sort import nn_matching
from edgeapp.deep_sort.detection import Detection
from edgeapp.deep_sort.tracker import Tracker
from edgeapp.tools import generate_detections as gdet
import requests
from edgeapp.models import Address
from Edge import settings

warnings.filterwarnings('ignore')
import json


def startRecognize(request):
    '''
    video_list = {
                   "video_list": [
                       {
                           'address': 'Test_A',
                           'video_path': 'edgeapp/Test_Video/t1.mp4'
                       },
                       {
                           'address': 'Test_B',
                           'video_path': 'edgeapp/Test_Video/t2.mp4'
                       },
                       {
                           'address': 'Test_C',
                           'video_path': 'edgeapp/Test_Video/t3.mp4'
                       }
                   ]
               }

start_recognize = requests.post("http://127.0.0.1:8000/start_recognize/",headers = {'Content-Type': 'application/json'},
                               data=json.dumps(video_list))

    :param request:
    :return:
    '''
    # if(not settings.is_dealing):
        # 开始处理
        # print("-------------1",settings.is_dealing)
        # settings.is_dealing = True
    json_data = json.loads(request.body)
    video_list = json_data['video_list']
    # dealVideoWithYoloAndDeepSort(settings.YOLO_INIT_SETTINGS, video_list)
    dealVideoWithYoloAndDeepSort(YOLO(), video_list)

    # 处理完成
    # settings.is_dealing = False
    # print("-------------2", settings.is_dealing)
    return JsonResponse({
        "info": "1"  # 表示处理完成
    })
    # else:
    #     return JsonResponse({
    #         "info": "0"  # 表示还在处理
    #     })


def dealVideoWithYoloAndDeepSort(yolo, video_list):
    # Definition of the parameters

    for video_info in video_list:
        # 对应监控地点的阈值
        address = video_info['address']
        video_path = video_info['video_path']
        addr = Address.objects.filter(address=address).first()  # video_info传过来是字符串，eval将其转化成字典
        threshold = addr.threshold
        VID = -1

        max_cosine_distance = 0.3
        nn_budget = None
        nms_max_overlap = 1.0

        # deep_sort
        model_filename = 'edgeapp/model_data/mars-small128.pb'
        encoder = gdet.create_box_encoder(model_filename, batch_size=1)

        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        tracker = Tracker(metric)

        writeVideo_flag = True

        # 视频路径   若为0则检测摄像头

        video_capture = cv2.VideoCapture(video_path)

        if writeVideo_flag:
            # Define the codec and create VideoWriter object
            w = int(video_capture.get(3))
            h = int(video_capture.get(4))

            out_video_fps = 30
            # avi 格式
            # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            # out = cv2.VideoWriter('output.avi', fourcc, out_video_fps, (w, h))

            # 生成mp4格式的输出视频 此“不支持”html5的 video标签播放
            # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # out = cv2.VideoWriter('output.mp4', fourcc, out_video_fps, (w, h))

            # 生成mp4格式的输出视频 此“支持”html5的 video标签播放
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            out = cv2.VideoWriter('edgeapp/Result_Video/{}_output.mp4'.format(address), fourcc, out_video_fps,
                                  (w, h))

            list_file = open('detection.txt', 'w')
            frame_index = -1

        fps = 0.0
        while True:
            ret, frame = video_capture.read()  # frame shape 640*480*3
            if ret != True:
                break
            t1 = time.time()

            # image = Image.fromarray(frame)
            image = Image.fromarray(frame[..., ::-1])  # bgr to rgb

            boxs = yolo.detect_image(image)

            # 统计人数
            person_count = len(boxs)
            print("视频中的人数：{}".format(person_count))

            ## HTTP 业务逻辑处理
            if (person_count >= threshold) and (VID == -1):
                # 让web服务器新建立一个异常视频对象
                print("------开始建立异常视频对象")
                res = requests.get("http://127.0.0.1:8080/push_new/",
                                   params={
                                       "vid": VID,
                                       "address": address,
                                       "number": person_count
                                   })
                res_dic = eval(res.text)
                VID = res_dic['vid']
            if VID != -1:
                # 持续向web服务器推送当前监控地点的人数
                print("------推送人数")
                requests.get("http://127.0.0.1:8080/deal_new/",
                             params={
                                 "vid": VID,
                                 "address": address,
                                 "number": person_count
                             })

            features = encoder(frame, boxs)

            # score to 1.0 here.
            detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]

            # Run non-maxima suppression.
            boxes = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
            detections = [detections[i] for i in indices]

            # Call the tracker
            tracker.predict()
            tracker.update(detections)

            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                bbox = track.to_tlbr()
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)
                cv2.putText(frame, str(track.track_id), (int(bbox[0]), int(bbox[1])), 0, 5e-3 * 200, (0, 255, 0), 2)

            for det in detections:
                bbox = det.to_tlbr()
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)

            # 是否实时同步展示标记过的视频
            # cv2.imshow('', frame)

            if writeVideo_flag:
                # save a frame
                out.write(frame)
                frame_index = frame_index + 1
                list_file.write(str(frame_index) + ' ')
                if len(boxs) != 0:
                    for i in range(0, len(boxs)):
                        list_file.write(
                            str(boxs[i][0]) + ' ' + str(boxs[i][1]) + ' ' + str(boxs[i][2]) + ' ' + str(
                                boxs[i][3]) + ' ')
                list_file.write('\n')

            fps = (fps + (1. / (time.time() - t1))) / 2
            print("fps= %f" % (fps))

            # Press Q to stop!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        if writeVideo_flag:
            out.release()
            list_file.close()
        cv2.destroyAllWindows()
        # 传送视频
        send_video = open('edgeapp/Result_Video/{}_output.mp4'.format(address), 'rb')
        files = {'file': send_video}
        res = requests.post("http://127.0.0.1:8080/store_video/", files=files, data={
            "vid": VID
        })
        # 关闭文件
        send_video.close()

    # 销毁yolo对象
    del yolo
