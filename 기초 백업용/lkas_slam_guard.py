#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LKAS + SLAM Guard (IMU 제거 버전)

- SLAM(cmd_vel_slam)을 기본 추종
- LKAS offset 이탈 시: 정지 + 제자리 회전
- IMU 기반 bump 감속 기능 완전 제거
"""

import rospy
import threading
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32


class EWMA(object):
    def __init__(self, alpha=0.6, init_val=0.0):
        self.a = float(alpha)
        self.y = float(init_val)
        self.initialized = False

    def reset(self, v=0.0):
        self.y = float(v)
        self.initialized = True

    def filt(self, x):
        x = float(x)
        if not self.initialized:
            self.reset(x)
        self.y = self.a * self.y + (1.0 - self.a) * x
        return self.y


class LkasSlamGuard(object):
    def __init__(self):
        rospy.init_node("lkas_slam_guard")

        # ===== LKAS 파라미터 =====
        self.depart_threshold = rospy.get_param("~depart_threshold", 0.25)
        self.clear_threshold  = rospy.get_param("~clear_threshold", 0.10)
        self.turn_gain        = rospy.get_param("~turn_gain", 1.5)

        # ===== 제어 주기 =====
        self.control_rate_hz = rospy.get_param("~control_rate", 20.0)

        # ===== 내부 상태 =====
        self.slam_cmd = Twist()
        self.offset_raw = 0.0
        self.offset_filt = 0.0
        self.offset_filter = EWMA(alpha=0.6, init_val=0.0)

        self.in_depart = False
        self.cmd_lock = threading.Lock()

        # ===== Subscribers =====
        rospy.Subscriber("cmd_vel_slam", Twist, self.cb_slam)
        rospy.Subscriber("lkas_offset", Float32, self.cb_offset)

        # ===== Publisher =====
        self.pub_cmd = rospy.Publisher("cmd_vel", Twist, queue_size=1)

        # ===== Timer =====
        self.timer = rospy.Timer(rospy.Duration(1.0 / self.control_rate_hz),
                                 self.cb_timer)

        rospy.loginfo("[LKAS_GUARD] STARTED (IMU OFF)")

    # ------------------------------------------------------
    # SLAM speed callback
    def cb_slam(self, msg):
        with self.cmd_lock:
            self.slam_cmd = msg

    # LKAS offset callback
    def cb_offset(self, msg):
        with self.cmd_lock:
            self.offset_raw = msg.data
            self.offset_filt = self.offset_filter.filt(self.offset_raw)

    # ------------------------------------------------------
    # Main control timer
    def cb_timer(self, event):
        with self.cmd_lock:
            slam = self.slam_cmd
            offset = self.offset_filt

        out = Twist()

        # 1) 기본은 SLAM 명령을 따라감
        out.linear.x  = slam.linear.x
        out.angular.z = slam.angular.z

        # 2) LKAS 이탈 감지
        if self.in_depart:
            if abs(offset) <= self.clear_threshold:
                self.in_depart = False
                rospy.loginfo("[LKAS_GUARD] Lane cleared (%.3f)" % offset)
            else:
                out.linear.x = 0.0
                out.angular.z = self.turn_gain * offset
        else:
            if abs(offset) >= self.depart_threshold:
                self.in_depart = True
                rospy.logwarn("[LKAS_GUARD] Lane departure (%.3f)" % offset)
                out.linear.x = 0.0
                out.angular.z = self.turn_gain * offset

        # 최종 출력
        self.pub_cmd.publish(out)


if __name__ == "__main__":
    try:
        LkasSlamGuard()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
