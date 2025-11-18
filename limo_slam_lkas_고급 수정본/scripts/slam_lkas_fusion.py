#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SLAM + BEV Lane Center Fusion Controller (Smooth Version)
- 카메라 BEV 차선 중앙 offset + SLAM path를 융합해서 cmd_vel 생성
- 이전 fusion의 lane angle 방식 제거 → 중앙 유지 정확도 증가
- steering LPF + steering rate limit로 틱틱거림 완전 제거
- lane 미검출 시 SLAM-only, path 없음 시 안전 정지
"""

import math
import threading
from typing import Optional, Tuple
import numpy as np

import rospy
import tf
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Bool

import cv2


def normalize_angle(a):
    return math.atan2(math.sin(a), math.cos(a))


class EWMA:
    """지수 이동평균 필터 (LPF)."""
    def __init__(self, alpha=0.6, init_val=0.0):
        self.a = float(alpha)
        self.y = float(init_val)
        self.initialized = False

    def reset(self, v):
        self.y = float(v)
        self.initialized = True

    def filt(self, x):
        x = float(x)
        if not self.initialized:
            self.reset(x)
        self.y = self.a * self.y + (1.0 - self.a) * x
        return self.y


class SlamLkasFusion:
    def __init__(self):
        rospy.init_node("fusion_lane_center")

        # ========== PARAMS ==========
        self.rate_hz = rospy.get_param("~rate", 15)

        self.path_topic = rospy.get_param("~path_topic", "/move_base/NavfnROS/plan")
        self.cmd_topic = rospy.get_param("~cmd_topic", "/cmd_vel")
        self.base_frame = rospy.get_param("~base_frame", "base_link")
        self.map_frame = rospy.get_param("~map_frame", "map")

        self.stop_when_no_input = rospy.get_param("~stop_when_no_input", True)

        # velocity params
        self.max_speed = rospy.get_param("~max_speed", 0.35)
        self.min_speed = rospy.get_param("~min_speed", 0.12)
        self.lane_speed = rospy.get_param("~lane_speed", 0.28)
        self.no_lane_speed = rospy.get_param("~no_lane_speed", 0.20)

        # acceleration limits
        self.max_lin_acc = rospy.get_param("~max_lin_acc", 0.15)
        self.max_lin_dec = rospy.get_param("~max_lin_dec", 0.25)

        # angular velocity limit
        self.max_ang = rospy.get_param("~max_angular", 1.2)
        self.max_ang_rate = rospy.get_param("~max_ang_rate", 1.4)

        # steering rate limit (중요)
        self.max_steer_rate = rospy.get_param("~max_steer_rate", 0.25)

        # fusion weights
        self.alpha_lane = rospy.get_param("~alpha_lane", 0.55)
        self.deadzone_angle = rospy.get_param("~deadzone_angle", 0.08)

        # path lookahead
        self.distance_ahead = rospy.get_param("~distance_ahead", 0.9)
        self.distance_ahead_hi = rospy.get_param("~distance_ahead_hi", 1.3)

        # slowdown on turn
        self.turn_slowdown_k = rospy.get_param("~turn_slowdown_k", 0.9)

        # ========== STATE ==========
        self.lock = threading.RLock()

        self.current_path = None
        self.last_path_time = 0.0
        self.path_timeout = rospy.get_param("~path_timeout", 0.6)

        # BEV lane center offset
        self.lane_offset = 0.0            # -1.0 ~ +1.0
        self.lane_offset_valid = False
        self.last_lane_time = 0.0
        self.lane_timeout = rospy.get_param("~lane_timeout", 0.6)

        # filters
        self.f_path = EWMA(0.5)
        self.f_steer = EWMA(0.6)
        self.f_speed = EWMA(0.7)

        self.prev_lin = 0.0
        self.prev_ang = 0.0
        self.prev_steer = 0.0
        self.prev_t = rospy.Time.now().to_sec()

        # ========== ROS IO ==========
        self.cmd_pub = rospy.Publisher(self.cmd_topic, Twist, queue_size=1)

        rospy.Subscriber(self.path_topic, Path, self.path_callback)
        rospy.Subscriber("/lane_center_offset", Float32, self.offset_callback)
        rospy.Subscriber("/lane_center_valid", Bool, self.offset_valid_callback)

        self.tf_listener = tf.TransformListener()

        rospy.loginfo("[Fusion] Ready. Using BEV lane center + SLAM")

    # ======================================
    # PATH
    # ======================================
    def path_callback(self, msg):
        with self.lock:
            self.current_path = msg
            self.last_path_time = rospy.get_time()

    # ======================================
    # LANE CENTER OFFSET
    # ======================================
    def offset_callback(self, msg):
        self.lane_offset = msg.data
        self.last_lane_time = rospy.get_time()

    def offset_valid_callback(self, msg):
        self.lane_offset_valid = msg.data

    # ======================================
    # TF Pose
    # ======================================
    def get_pose(self):
        try:
            now = rospy.Time(0)
            self.tf_listener.waitForTransform(self.map_frame, self.base_frame,
                                              now, rospy.Duration(0.3))
            trans, rot = self.tf_listener.lookupTransform(self.map_frame,
                                                          self.base_frame, now)
            x, y = trans[0], trans[1]
            yaw = tf.transformations.euler_from_quaternion(rot)[2]
            return x, y, yaw
        except:
            return None

    # ======================================
    # PATH Target Angle
    # ======================================
    def compute_path_angle(self, pose):
        with self.lock:
            path = self.current_path
            last_t = self.last_path_time

        if path is None:
            return None
        if rospy.get_time() - last_t > self.path_timeout:
            return None
        if len(path.poses) == 0:
            return None

        rx, ry, ryaw = pose

        v = max(self.prev_lin, self.min_speed)
        la_ratio = min(1.0, v / max(self.max_speed, 1e-6))
        la = self.distance_ahead + (self.distance_ahead_hi - self.distance_ahead) * la_ratio

        target_pt = None
        for ps in path.poses:
            px = ps.pose.position.x
            py = ps.pose.position.y
            if math.hypot(px - rx, py - ry) >= la:
                target_pt = (px, py)
                break

        if target_pt is None:
            last = path.poses[-1].pose
            target_pt = (last.position.x, last.position.y)

        tx, ty = target_pt

        yaw_tgt = math.atan2(ty - ry, tx - rx)
        err = normalize_angle(yaw_tgt - ryaw)

        return self.f_path.filt(err)

    # ======================================
    # 작은 기능
    # ======================================
    @staticmethod
    def clamp(x, lo, hi):
        return min(hi, max(lo, x))

    @staticmethod
    def rate_limit(prev, target, max_delta):
        if target > prev:
            return min(prev + max_delta, target)
        else:
            return max(prev - max_delta, target)

    # ======================================
    # CONTROL LOOP
    # ======================================
    def control_loop(self):
        rate = rospy.Rate(self.rate_hz)
        rospy.sleep(0.25)

        while not rospy.is_shutdown():
            now = rospy.Time.now().to_sec()
            dt = max(1.0 / self.rate_hz, now - self.prev_t)
            self.prev_t = now

            pose = self.get_pose()
            tw = Twist()

            if pose is None:
                tw.linear.x = 0.0
                tw.angular.z = 0.0
                self.cmd_pub.publish(tw)
                continue

            # -------- Path steering --------
            path_ang = self.compute_path_angle(pose)
            have_path = path_ang is not None

            if have_path and abs(path_ang) < 0.02:
                path_ang = 0.0

            # -------- Lane steering (offset) --------
            lane_ok = (rospy.get_time() - self.last_lane_time) <= self.lane_timeout and \
                      self.lane_offset_valid

            K_offset = 1.2

            if lane_ok:
                lane_steer = K_offset * self.lane_offset
            else:
                lane_steer = self.prev_steer  # 이전 값 유지

            # -------- Fusion --------
            if have_path and lane_ok:
                combined = self.alpha_lane * lane_steer + (1.0 - self.alpha_lane) * path_ang
                base_speed = self.lane_speed
            elif have_path:
                combined = path_ang
                base_speed = self.no_lane_speed
            elif lane_ok:
                combined = lane_steer
                base_speed = self.lane_speed
            else:
                combined = 0.0
                base_speed = 0.0 if self.stop_when_no_input else self.min_speed

            # deadzone
            if abs(combined) < self.deadzone_angle:
                combined = 0.0

            # steering filtering + rate limit
            steer_raw = self.f_steer.filt(combined)
            max_dsteer = self.max_steer_rate * dt
            steer = self.rate_limit(self.prev_steer, steer_raw, max_dsteer)
            self.prev_steer = steer

            # turn slowdown
            turn_scale = 1.0 / (1.0 + self.turn_slowdown_k * (abs(steer) ** 1.2))
            target_lin = self.clamp(base_speed * turn_scale, 0.0, self.max_speed)

            # steer → angular velocity
            target_ang = self.clamp(-steer, -self.max_ang, self.max_ang)

            # acceleration limits
            max_dv = (self.max_lin_acc if target_lin >= self.prev_lin else self.max_lin_dec) * dt
            cmd_lin = self.rate_limit(self.prev_lin, target_lin, max_dv)

            # angular rate limit
            cmd_ang = self.rate_limit(self.prev_ang, target_ang, self.max_ang_rate * dt)

            # smoothing
            cmd_lin = self.f_speed.filt(cmd_lin)

            tw.linear.x = cmd_lin
            tw.angular.z = cmd_ang

            self.cmd_pub.publish(tw)

            self.prev_lin = cmd_lin
            self.prev_ang = cmd_ang

            rate.sleep()


if __name__ == "__main__":
    try:
        node = SlamLkasFusion()
        node.control_loop()
    except rospy.ROSInterruptException:
        pass
