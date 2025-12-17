#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32, Bool
import threading

# ---------------------------------------------------------
# EWMA Filter
# ---------------------------------------------------------
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


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


# ---------------------------------------------------------
# LKAS + SLAM Guard (SYMMETRIC STABLE MIXING VERSION)
#  - SLAM 속도 유지
#  - 차선 이탈 시 angular만 보정
#  - 좌/우 완전 대칭
# ---------------------------------------------------------
class LkasSlamGuard(object):
    def __init__(self):
        rospy.init_node("lkas_slam_guard")

        # =====================================================
        # PARAMETERS
        # =====================================================
        self.rate_hz = rospy.get_param("~rate", 20)

        self.slam_cmd_topic = rospy.get_param("~slam_cmd_topic", "/cmd_vel_slam")
        self.out_cmd_topic  = rospy.get_param("~out_cmd_topic",  "/cmd_vel")

        # LKAS enable
        self.enable_lkas = rospy.get_param("~enable_lkas", True)

        # depart / clear (좌우 동일)
        self.depart_th = rospy.get_param("~depart_threshold", 0.25)
        self.clear_th  = rospy.get_param("~clear_threshold",  0.16)

        # lane gain (좌우 동일)
        self.lane_gain = rospy.get_param("~lane_gain", 1.3)

        # SLAM ↔ LKAS mixing ratio
        self.lane_mix = rospy.get_param("~lane_mix", 0.75)

        # angular limits / deadzone
        self.max_ang  = rospy.get_param("~max_ang", 0.9)
        self.deadzone = rospy.get_param("~deadzone", 0.04)

        # forward handling during override
        self.keep_forward_in_override = rospy.get_param("~keep_forward_in_override", True)
        self.min_forward = rospy.get_param("~min_forward", 0.06)

        # Timeouts
        self.slam_timeout = rospy.get_param("~slam_timeout", 0.5)
        self.lane_timeout = rospy.get_param("~lane_timeout", 0.5)

        # Filters
        self.ang_alpha = rospy.get_param("~ang_alpha", 0.65)
        self.off_alpha = rospy.get_param("~offset_alpha", 0.70)

        # =====================================================
        # STATE
        # =====================================================
        self.lock = threading.RLock()

        self.last_slam_cmd  = Twist()
        self.last_slam_time = 0.0

        self.lane_offset   = 0.0
        self.lane_valid    = False
        self.last_lane_time = 0.0

        self.in_override = False

        self.f_ang = EWMA(alpha=self.ang_alpha, init_val=0.0)
        self.f_off = EWMA(alpha=self.off_alpha, init_val=0.0)

        # =====================================================
        # ROS IO
        # =====================================================
        self.pub_cmd = rospy.Publisher(self.out_cmd_topic, Twist, queue_size=1)

        rospy.Subscriber(self.slam_cmd_topic, Twist,   self.cb_slam)
        rospy.Subscriber("/lane_center_offset", Float32, self.cb_offset)
        rospy.Subscriber("/lane_center_valid",  Bool,    self.cb_valid)

        rospy.loginfo("[LKAS_GUARD] READY (SYMMETRIC VERSION)")
        rospy.loginfo("SLAM → %s", self.slam_cmd_topic)
        rospy.loginfo("OUT  → %s", self.out_cmd_topic)
        rospy.loginfo("depart_th = %.3f | clear_th = %.3f", self.depart_th, self.clear_th)
        rospy.loginfo("lane_gain = %.3f | lane_mix = %.2f", self.lane_gain, self.lane_mix)

        self.spin()

    # ---------------------------------------------------------
    # CALLBACKS
    # ---------------------------------------------------------
    def cb_slam(self, msg):
        with self.lock:
            self.last_slam_cmd = msg
            self.last_slam_time = rospy.get_time()

    def cb_offset(self, msg):
        with self.lock:
            self.lane_offset = msg.data
            self.last_lane_time = rospy.get_time()

    def cb_valid(self, msg):
        with self.lock:
            self.lane_valid = msg.data

    # ---------------------------------------------------------
    # MAIN LOOP
    # ---------------------------------------------------------
    def spin(self):
        rate = rospy.Rate(self.rate_hz)

        while not rospy.is_shutdown():
            now = rospy.get_time()

            with self.lock:
                slam_cmd  = self.last_slam_cmd
                age_slam  = now - self.last_slam_time
                age_lane  = now - self.last_lane_time
                lane_ok   = (age_lane <= self.lane_timeout) and self.lane_valid
                raw_offset = self.lane_offset

            cmd = Twist()

            # 1) SLAM timeout → 정지
            if age_slam > self.slam_timeout:
                self.in_override = False
                self.pub_cmd.publish(cmd)
                rate.sleep()
                continue

            # 기본은 SLAM
            cmd = slam_cmd

            # 2) offset 필터
            offset = self.f_off.filt(raw_offset) if lane_ok else 0.0

            # 3) override 진입 / 해제 (대칭)
            if self.enable_lkas and lane_ok:
                if abs(offset) >= self.depart_th:
                    self.in_override = True
                elif abs(offset) <= self.clear_th:
                    self.in_override = False
            else:
                self.in_override = False

            # 4) LKAS angular 계산 (대칭)
            if self.enable_lkas and lane_ok:
                lane_ang = self.lane_gain * offset

                if abs(lane_ang) < self.deadzone:
                    lane_ang = 0.0

                lane_ang = clamp(lane_ang, -self.max_ang, self.max_ang)
            else:
                lane_ang = 0.0

            # 5) override 상태
            if self.enable_lkas and self.in_override and lane_ok:
                if self.keep_forward_in_override:
                    cmd.linear.x = max(cmd.linear.x, self.min_forward)
                else:
                    cmd.linear.x = 0.0

                mixed_ang = (1.0 - self.lane_mix) * cmd.angular.z + self.lane_mix * lane_ang
                cmd.angular.z = self.f_ang.filt(clamp(mixed_ang, -self.max_ang, self.max_ang))

                self.pub_cmd.publish(cmd)
                rate.sleep()
                continue

            # 6) 정상 SLAM
            cmd.angular.z = self.f_ang.filt(cmd.angular.z)
            self.pub_cmd.publish(cmd)
            rate.sleep()


if __name__ == "__main__":
    LkasSlamGuard()
