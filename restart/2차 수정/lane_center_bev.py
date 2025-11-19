#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Bool


class LaneCenterBEV(object):
    def __init__(self):
        rospy.init_node("lane_center_bev")

        self.bridge = CvBridge()

        # 기본 RGB 카메라 토픽
        self.image_topic = rospy.get_param("~image_topic", "/camera/rgb/image_raw")

        self.pub_offset = rospy.Publisher("/lane_center_offset", Float32, queue_size=1)
        self.pub_valid  = rospy.Publisher("/lane_center_valid",  Bool, queue_size=1)

        # BEV용 Homography (640x480 가정)
        # 필요하면 나중에 같이 튜닝 가능
        self.warp_src = np.float32([
            [220, 240],
            [420, 240],
            [50,  380],
            [590, 380]
        ])
        self.warp_dst = np.float32([
            [150,   0],
            [330,   0],
            [150, 480],
            [330, 480]
        ])
        self.M = cv2.getPerspectiveTransform(self.warp_src, self.warp_dst)

        rospy.Subscriber(self.image_topic, Image, self.image_cb, queue_size=1)

        rospy.loginfo("[lane_center_bev] Subscribing image: %s", self.image_topic)

    def image_cb(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError:
            return

        # 1) BEV warp
        bev = cv2.warpPerspective(img, self.M, (480, 480))

        # 2) HSV에서 흰색+노란색 차선 추출
        hsv = cv2.cvtColor(bev, cv2.COLOR_BGR2HSV)

        # 흰색
        white_lower = np.array([0,   0, 160], dtype=np.uint8)
        white_upper = np.array([179, 40, 255], dtype=np.uint8)
        mask_white = cv2.inRange(hsv, white_lower, white_upper)

        # 노란색
        yellow_lower = np.array([15, 80, 80], dtype=np.uint8)
        yellow_upper = np.array([40, 255, 255], dtype=np.uint8)
        mask_yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)

        mask = cv2.bitwise_or(mask_white, mask_yellow)

        # 3) 하단 ROI만 사용 (하단 40%)
        h, w = mask.shape
        roi = mask[int(h * 0.6):, :]

        # 4) 세로 히스토그램 (픽셀 개수)
        bin_roi = (roi > 0).astype(np.uint8)
        column_sum = np.sum(bin_roi, axis=0)  # 각 x 컬럼마다 픽셀 개수

        if np.max(column_sum) < 10:
            # 아무것도 안 보이면 invalid
            self.pub_valid.publish(False)
            return

        midpoint = w // 2
        left_region  = column_sum[:midpoint]
        right_region = column_sum[midpoint:]

        leftx  = np.argmax(left_region)
        rightx = np.argmax(right_region) + midpoint

        left_val  = column_sum[leftx]
        right_val = column_sum[rightx]

        # 적응형 threshold (전체 평균 대비)
        base = float(np.mean(column_sum))
        thresh = max(15.0, base * 1.8)  # 너무 낮지 않게 최소 15

        valid_left  = left_val  > thresh
        valid_right = right_val > thresh

        if not (valid_left and valid_right):
            self.pub_valid.publish(False)
            return

        # 5) 중앙 계산
        lane_center_x = (leftx + rightx) / 2.0
        camera_center_x = w / 2.0

        offset_px = lane_center_x - camera_center_x

        # -1 ~ +1로 정규화 (왼쪽 +, 오른쪽 -)
        offset_norm = float(offset_px / camera_center_x)

        self.pub_offset.publish(offset_norm)
        self.pub_valid.publish(True)


if __name__ == "__main__":
    node = LaneCenterBEV()
    rospy.spin()
