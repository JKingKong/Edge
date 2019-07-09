#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

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
from Edge.edgeapp.models import Address
warnings.filterwarnings('ignore')

import os
import django


def start_deal(yolo, video_info, threshold):
    # Definition of the parameters
    VID = -1

    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0

    # deep_sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    writeVideo_flag = True

    # 视频路径   若为0则检测摄像头

    video_capture = cv2.VideoCapture(video_info['video_path'])

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
        out = cv2.VideoWriter('{}_output.mp4'.format(video_info['address']), fourcc, out_video_fps, (w, h))

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
                                   "address": video_info['address'],
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
                             "address": video_info['address'],
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
                        str(boxs[i][0]) + ' ' + str(boxs[i][1]) + ' ' + str(boxs[i][2]) + ' ' + str(boxs[i][3]) + ' ')
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
    files = {'file': open('{}_output.mp4'.format(video_info['address']), 'rb')}
    res = requests.post("http://127.0.0.1:8080/store_video/", files=files, data={
        "vid": VID
    })


def main():
    video_list = [
        {
            'address': 'Test_A',
            'video_path': 'Test_Video/t01.mp4'
        },
        {
            'address': 'Test_B',
            'video_path': 'Test_Video/t02.mp4'
        },
        {
            'address': 'Test_A',
            'video_path': 'Test_Video/t03.mp4'
        }
    ]
    # addr = Address.objects.all()
    # # {
    # #   "地址一"："地址一对应的阈值",
    # #   "地址二"："地址二对应的阈值",
    # #   ...
    # # }
    # address_list = {}
    #
    # for item in addr:
    #     address_list[item.address] = item.threshold
    address_list = {
        "Test_A": 5,
        "Test_B": 3,
        "Test_C": 7,
    }

    for video_info in video_list:
        main(YOLO(), video_info, address_list[video_info['address']])


if __name__ == '__main__':
    video_list = [
        {
            'address': 'Test_A',
            'video_path': 'Test_Video/t1.mp4'
        },
        {
            'address': 'Test_B',
            'video_path': 'Test_Video/t2.mp4'
        },
        {
            'address': 'Test_C',
            'video_path': 'Test_Video/t3.mp4'
        }
    ]
    addr = Address.objects.all()
    # {
    #   "地址一"："地址一对应的阈值",
    #   "地址二"："地址二对应的阈值",
    #   ...
    # }
    address_list = {
        "Test_A": 5,
        "Test_B": 15,
        "Test_C": 7,
    }

    for item in addr:
        address_list[item.address] = item.threshold

    address_list = {
        "Test_A": 5,
        "Test_B": 3,
        "Test_C": 7,
    }

    for video_info in video_list:
        start_deal(YOLO(), video_info, address_list[video_info['address']])
