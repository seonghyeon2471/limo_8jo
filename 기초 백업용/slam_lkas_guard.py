#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LKAS + SLAM Guard + IMU pitch 기반 bump 감속 (내부 계산 버전)

- SLAM(cmd_vel_slam) 기본 추종
- LKAS offset 이탈 시: 정지 + 제자리 회전
- IMU accel 기반 pitch 사용하여 bump 감속
    - |pitch| >= bump_pitch_deg → 감속 모드 ON
    - bump 모드에서는 선속도를 bump_max_speed 이하로 클램프
"""

import rospy
import threading
import math
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32
from sensor_msgs.msg import Imu


class EWMA(object):
    def __init__(self, alpha=0.9, init_val=0.0):  # ★ pitch 안정성 위해 강화
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

        # ===== IMU bump 파라미터 =====
        self.bump_pitch_deg = rospy.get_param("~bump_pitch_deg", 3.0)
        self.bump_max_speed = rospy.get_param("~bump_max_speed", 0.06)
        self.bump_hold_time = rospy.get_param("~bump_hold_time", 0.4)

        # ===== 제어 주기 =====
        self.control_rate_hz = rospy.get_param("~control_rate", 20.0)

        # ===== 내부 상태 =====
        self.slam_cmd = Twist()
        self.offset_raw = 0.0
        self.offset_filt = 0.0
        self.offset_filter = EWMA(alpha=0.6, init_val=0.0)

        self.pitch_deg = 0.0
        self.pitch_filter = EWMA(alpha=0.9)

        self.in_depart = False

        self.bump_active = False
        self.bump_release_time = 0.0

        self.cmd_lock = threading.Lock()

        # ===== Subscribers =====
        rospy.Subscriber("cmd_vel_slam", Twist, self.cb_slam)
        rospy.Subscriber("lkas_offset", Float32, self.cb_offset)
        rospy.Subscriber("/imu", Imu, self.cb_imu)
        rospy.Subscriber("/imu/data", Imu, self.cb_imu)  # 두 경우 모두 대응

        # ===== Publisher =====
        self.pub_cmd = rospy.Publisher("cmd_vel", Twist, queue_size=1)

        # ===== Timer =====
        self.timer = rospy.Timer(rospy.Duration(1.0 / self.control_rate_hz),
                                 self.cb_timer)

        rospy.loginfo("[LKAS_GUARD] STARTED (pitch_th=%.1f, bump_speed=%.2f)",
                      self.bump_pitch_deg, self.bump_max_speed)

    # ------------------------------------------------------
    # 콜백: SLAM speed
    def cb_slam(self, msg):
        with self.cmd_lock:
            self.slam_cmd = msg

    # LKAS offset
    def cb_offset(self, msg):
        with self.cmd_lock:
            self.offset_raw = msg.data
            self.offset_filt = self.offset_filter.filt(self.offset_raw)

    # IMU pitch 계산 (accel 기반)
    def cb_imu(self, msg):
        ax = msg.linear_acceleration.x
        ay = msg.linear_acceleration.y
        az = msg.linear_acceleration.z

        pitch_rad = math.atan2(ax, math.sqrt(ay * ay + az * az))
        pitch_deg = math.degrees(pitch_rad)

        pitch_deg = self.pitch_filter.filt(pitch_deg)

        with self.cmd_lock:
            self.pitch_deg = pitch_deg

    # ------------------------------------------------------
    # 메인 제어 루프
    def cb_timer(self, event):
        with self.cmd_lock:
            slam = self.slam_cmd
            offset = self.offset_filt
            pitch = self.pitch_deg

        out = Twist()

        # 1) 기본은 SLAM 명령을 따라감
        out.linear.x  = slam.linear.x
        out.angular.z = slam.angular.z

        # 2) IMU bump 감속 로직
        if not self.bump_active:
            if abs(pitch) >= self.bump_pitch_deg:
                self.bump_active = True
                self.bump_release_time = rospy.get_time() + self.bump_hold_time
                rospy.logwarn("[LKAS_GUARD] BUMP ON (pitch=%.2f°)" % pitch)
        else:
            if rospy.get_time() >= self.bump_release_time:
                self.bump_active = False
                rospy.loginfo("[LKAS_GUARD] BUMP OFF")

        # bump 중에는 선속도 제한
        if self.bump_active:
            if out.linear.x > self.bump_max_speed:
                out.linear.x = self.bump_max_speed
            elif out.linear.x < -self.bump_max_speed:
                out.linear.x = -self.bump_max_speed

        # 3) LKAS 이탈 감지
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

        # 4) publish
        self.pub_cmd.publish(out)


if __name__ == "__main__":
    try:
        LkasSlamGuard()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
