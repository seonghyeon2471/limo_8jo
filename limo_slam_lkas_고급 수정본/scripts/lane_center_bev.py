#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Bool

class LaneCenterBEV:
    def __init__(self):
        rospy.init_node("lane_center_bev")

        self.bridge = CvBridge()

        self.image_topic = rospy.get_param("~image_topic", "/camera/image_raw")
        self.pub_offset = rospy.Publisher("/lane_center_offset", Float32, queue_size=1)
        self.pub_valid  = rospy.Publisher("/lane_center_valid",  Bool, queue_size=1)

        # BEV parameters
        self.warp_src = np.float32([[220, 240], [420, 240], [50, 380], [590, 380]])
        self.warp_dst = np.float32([[150, 0], [330, 0], [150, 480], [330, 480]])
        self.M = cv2.getPerspectiveTransform(self.warp_src, self.warp_dst)

        rospy.Subscriber(self.image_topic, Image, self.image_cb, queue_size=1)

    def image_cb(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except:
            return

        # ---------- 1) BEV Warp ----------
        bev = cv2.warpPerspective(img, self.M, (480, 480))

        # ---------- 2) Grayscale + Canny ----------
        gray = cv2.cvtColor(bev, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        edges = cv2.Canny(blur, 60, 150)

        # ---------- 3) ROI (하단 1/3만 사용) ----------
        h, w = edges.shape
        roi = edges[int(h*0.6):, :]

        # ---------- 4) Histogram peak detection ----------
        column_sum = np.sum(roi, axis=0)

        midpoint = w // 2
        leftx  = np.argmax(column_sum[:midpoint])
        rightx = np.argmax(column_sum[midpoint:]) + midpoint

        # threshold: 차선 검출 valid 여부
        left_val  = column_sum[leftx]
        right_val = column_sum[rightx]

        valid_left  = left_val  > 10000
        valid_right = right_val > 10000

        if not (valid_left and valid_right):
            self.pub_valid.publish(False)
            return

        # ---------- 5) 중앙 계산 ----------
        lane_center_x = (leftx + rightx) / 2.0
        camera_center_x = w / 2.0

        # BEV에서는 왼쪽 = + / 오른쪽 = - 로 처리해도 되고 반대로 가능
        offset_px = lane_center_x - camera_center_x

        # ---------- 6) pixel → normalized offset ----------
        offset_norm = offset_px / camera_center_x   # -1 ~ +1 범위

        # ---------- 7) Publish ----------
        self.pub_offset.publish(offset_norm)
        self.pub_valid.publish(True)

if __name__=="__main__":
    LaneCenterBEV()
    rospy.spin()