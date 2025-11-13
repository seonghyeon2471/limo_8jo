#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy, cv2, numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from limo_slam_lkas.msg import LaneDebug

class LaneBEV:
    def __init__(self):
        rospy.init_node('lane_bev_node')
        self.bridge = CvBridge()
        topic = rospy.get_param('~image_topic', '/camera/image_raw')
        self.img_sub = rospy.Subscriber(topic, Image, self.cb, queue_size=1)
        self.pub_dbg = rospy.Publisher('/slam_lkas_debug', LaneDebug, queue_size=1)
        self.roi_top = rospy.get_param('~roi_top_ratio', 0.45)
        self.canny_lo = rospy.get_param('~canny_lo', 50)
        self.canny_hi = rospy.get_param('~canny_hi', 150)
        self.h_th = rospy.get_param('~hough_thresh', 24)
        self.h_min = rospy.get_param('~hough_min_len', 32)
        self.h_gap = rospy.get_param('~hough_max_gap', 22)

    def cb(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            if len(img.shape)==2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        h, w = img.shape[:2]
        y0 = int(h*self.roi_top)
        roi = img[y0:, :]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, self.canny_lo, self.canny_hi)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, self.h_th, minLineLength=self.h_min, maxLineGap=self.h_gap)
        valid, ang = False, 0.0
        if lines is not None:
            angs = []
            for l in lines:
                x1,y1,x2,y2 = l[0]
                dx,dy = x2-x1, y2-y1
                if dx==0 and dy==0: continue
                a = np.arctan2(dy,dx)
                rel = (np.pi/2)-a
                if abs(rel) < np.deg2rad(80):
                    angs.append(rel)
            if len(angs)>0:
                ang = float(np.median(angs)); valid=True
        dbg = LaneDebug()
        dbg.header = Header(stamp=rospy.Time.now())
        dbg.angle_raw = ang
        dbg.angle_filt = ang
        dbg.valid = valid
        dbg.roi_top_ratio = float(self.roi_top)
        self.pub_dbg.publish(dbg)

if __name__=='__main__':
    LaneBEV()
    rospy.spin()
