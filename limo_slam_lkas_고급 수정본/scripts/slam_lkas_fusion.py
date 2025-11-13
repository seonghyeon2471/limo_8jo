#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SLAM + LKAS Fusion Controller (ROS1)
- Robust against missing inputs and TF hiccups
- Deterministic timing based on ROS time (sim or real)
- Thread-safe handling of lane detection
- Gentle accel/decel and angular-rate limiting
- Optional stop-when-no-input behavior

Tested structure-wise; tune thresholds per platform/camera.
"""

import math
import time
import threading
from typing import Optional, Tuple

import numpy as np
import rospy
import tf
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path
from sensor_msgs.msg import Image
import cv2


def normalize_angle(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


class EWMA(object):
    def __init__(self, alpha=0.6, init_val=0.0):
        self.a = float(alpha)
        self.y = float(init_val)
        self.initialized = False

    def reset(self, val=0.0):
        self.y = float(val)
        self.initialized = True

    def filt(self, x: float) -> float:
        x = float(x)
        if not self.initialized:
            self.reset(x)
        self.y = self.a * self.y + (1.0 - self.a) * x
        return self.y


class SlamLkasFusion(object):
    def __init__(self):
        rospy.init_node('slam_lkas_fusion')

        # === params ===
        self.rate_hz = rospy.get_param('~rate', 15)
        self.path_topic = rospy.get_param('~path_topic', '/move_base/NavfnROS/plan')
        self.image_topic = rospy.get_param('~image_topic', '/camera/image_raw')
        self.cmd_topic = rospy.get_param('~cmd_topic', '/cmd_vel')
        self.base_frame = rospy.get_param('~base_frame', 'base_link')
        self.map_frame = rospy.get_param('~map_frame', 'map')

        # Behavior
        self.stop_when_no_input = rospy.get_param('~stop_when_no_input', True)

        # Speed/steer limits
        self.max_speed = rospy.get_param('~max_speed', 0.35)
        self.min_speed = rospy.get_param('~min_speed', 0.12)
        self.lane_speed = rospy.get_param('~lane_speed', 0.28)
        self.no_lane_speed = rospy.get_param('~no_lane_speed', 0.20)
        self.max_ang = rospy.get_param('~max_angular', 1.2)

        # Kinematics rate limits
        self.max_lin_acc = rospy.get_param('~max_lin_acc', 0.25)
        self.max_lin_dec = rospy.get_param('~max_lin_dec', 0.35)
        self.max_ang_rate = rospy.get_param('~max_ang_rate', 1.5)

        # Fusion / lookahead
        self.alpha_lane = rospy.get_param('~alpha_lane', 0.55)
        self.distance_ahead = rospy.get_param('~distance_ahead', 0.9)
        self.distance_ahead_hi = rospy.get_param('~distance_ahead_hi', 1.3)
        self.turn_slowdown_k = rospy.get_param('~turn_slowdown_k', 0.9)

        # Detection / trust
        self.lane_detect_timeout = rospy.get_param('~lane_detect_timeout', 0.6)
        self.path_timeout = rospy.get_param('~path_timeout', 0.6)
        self.lane_ok_hold = rospy.get_param('~lane_ok_hold', 0.5)
        self.deadzone_angle = rospy.get_param('~deadzone_angle', 0.05)

        # LPFs
        self.lpf_lane_a = rospy.get_param('~lpf_lane_alpha', 0.5)
        self.lpf_path_a = rospy.get_param('~lpf_path_alpha', 0.5)
        self.lpf_steer_a = rospy.get_param('~lpf_steer_alpha', 0.6)
        self.lpf_speed_a = rospy.get_param('~lpf_speed_alpha', 0.5)

        # Image processing params (override in launch as needed)
        self.roi_top_ratio = rospy.get_param('~roi_top_ratio', 0.45)
        self.canny_lo = rospy.get_param('~canny_lo', 50)
        self.canny_hi = rospy.get_param('~canny_hi', 150)
        self.hough_thresh = rospy.get_param('~hough_thresh', 24)
        self.hough_min_len = rospy.get_param('~hough_min_len', 32)
        self.hough_max_gap = rospy.get_param('~hough_max_gap', 22)

        # === state ===
        self.lock = threading.RLock()
        self.current_path: Optional[Path] = None
        self.last_path_time = 0.0

        self.last_lane_time = 0.0
        self.have_lane_raw = False
        self.have_lane_latched = False
        self.lane_angle_raw = 0.0  # unfiltered; filter only in control loop

        self.f_lane = EWMA(self.lpf_lane_a, 0.0)
        self.f_path = EWMA(self.lpf_path_a, 0.0)
        self.f_steer = EWMA(self.lpf_steer_a, 0.0)
        self.f_speed = EWMA(self.lpf_speed_a, self.min_speed)

        self.prev_lin = 0.0
        self.prev_ang = 0.0
        self.prev_t = rospy.Time.now().to_sec()

        # ROS I/O
        self.cmd_pub = rospy.Publisher(self.cmd_topic, Twist, queue_size=1)
        rospy.Subscriber(self.path_topic, Path, self.path_callback, queue_size=1)
        rospy.Subscriber(self.image_topic, Image, self.image_callback, queue_size=1)

        self.tf_listener = tf.TransformListener()
        self.cv_bridge = CvBridge()

        rospy.loginfo("[slam_lkas_fusion] ready. path_topic=%s image_topic=%s", self.path_topic, self.image_topic)

    # ---------------- Callbacks ----------------
    def path_callback(self, msg: Path):
        with self.lock:
            self.current_path = msg
            self.last_path_time = rospy.get_time()

    def image_callback(self, msg: Image):
        try:
            # Try bgr8 first; if camera publishes mono16/other, this will throw
            try:
                cv_img = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            except CvBridgeError:
                # Fallback: convert any to mono and then to BGR for robustness
                cv_img_any = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                if len(cv_img_any.shape) == 2:
                    cv_img = cv2.cvtColor(cv_img_any, cv2.COLOR_GRAY2BGR)
                else:
                    cv_img = cv_img_any
        except CvBridgeError as e:
            rospy.logwarn_throttle(5.0, "CvBridge error: %s", str(e))
            return

        angle, valid = self.detect_lane_angle(cv_img)

        with self.lock:
            if valid and np.isfinite(angle):
                self.lane_angle_raw = float(angle)
                self.have_lane_raw = True
                self.last_lane_time = rospy.get_time()
            else:
                self.have_lane_raw = False

            # Latch (hysteresis)
            if self.have_lane_raw:
                self.have_lane_latched = True
            else:
                if rospy.get_time() - self.last_lane_time > self.lane_ok_hold:
                    self.have_lane_latched = False

    # ---------------- Vision ----------------
    def detect_lane_angle(self, bgr: np.ndarray) -> Tuple[float, bool]:
        h, w = bgr.shape[:2]
        roi_top = int(max(0, min(h - 1, h * float(self.roi_top_ratio))))
        roi = bgr[roi_top:h, 0:w]

        if roi.size == 0:
            return 0.0, False

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, self.canny_lo, self.canny_hi)

        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180,
            threshold=int(self.hough_thresh),
            minLineLength=int(self.hough_min_len),
            maxLineGap=int(self.hough_max_gap),
        )
        if lines is None:
            return 0.0, False

        angles = []
        for l in lines:
            x1, y1, x2, y2 = l[0]
            dx = float(x2 - x1)
            dy = float(y2 - y1)
            if dx == 0.0 and dy == 0.0:
                continue
            ang = math.atan2(dy, dx)
            # angle relative to vertical lane direction
            angle_rel_vertical = normalize_angle((math.pi / 2.0) - ang)
            # Reject near-horizontal segments (likely noise)
            if abs(angle_rel_vertical) > math.radians(80):
                continue
            angles.append(angle_rel_vertical)

        if not angles:
            return 0.0, False

        med = float(np.median(np.array(angles, dtype=np.float32)))
        return med, True

    # ---------------- TF & Path ----------------
    def get_robot_pose(self) -> Optional[Tuple[float, float, float]]:
        try:
            now = rospy.Time(0)
            self.tf_listener.waitForTransform(self.map_frame, self.base_frame, now, rospy.Duration(0.3))
            (trans, rot) = self.tf_listener.lookupTransform(self.map_frame, self.base_frame, now)
            x, y = trans[0], trans[1]
            yaw = tf.transformations.euler_from_quaternion(rot)[2]
            if not all(np.isfinite([x, y, yaw])):
                return None
            return x, y, yaw
        except Exception as e:
            rospy.logwarn_throttle(2.0, "TF lookup failed: %s", str(e))
            return None

    def compute_path_target_angle(self, robot_pose: Tuple[float, float, float]) -> Optional[float]:
        with self.lock:
            path = self.current_path
            last_time = self.last_path_time

        if path is None:
            return None
        if rospy.get_time() - last_time > self.path_timeout:
            return None
        if len(path.poses) == 0:
            return None

        rx, ry, ryaw = robot_pose

        current_speed_est = max(self.prev_lin, self.min_speed)
        la = self.distance_ahead + (self.distance_ahead_hi - self.distance_ahead) * min(1.0, current_speed_est / max(self.max_speed, 1e-6))

        best_pt = None
        for ps in path.poses:
            px = ps.pose.position.x
            py = ps.pose.position.y
            if math.hypot(px - rx, py - ry) >= la:
                best_pt = (px, py)
                break
        if best_pt is None:
            last = path.poses[-1].pose
            best_pt = (last.position.x, last.position.y)

        tx, ty = best_pt
        target_yaw = math.atan2(ty - ry, tx - rx)
        err = normalize_angle(target_yaw - ryaw)
        err = self.f_path.filt(err)
        return err

    # ---------------- helpers ----------------
    @staticmethod
    def clamp(x: float, lo: float, hi: float) -> float:
        return min(hi, max(lo, x))

    @staticmethod
    def rate_limit(prev: float, target: float, max_delta: float) -> float:
        if target > prev:
            return min(prev + max_delta, target)
        else:
            return max(prev - max_delta, target)

    # ---------------- Main loop ----------------
    def control_loop(self):
        rate = rospy.Rate(self.rate_hz)

        # Small startup grace period for TF/Topics to come up
        rospy.sleep(0.25)

        while not rospy.is_shutdown():
            now_t = rospy.Time.now().to_sec()
            dt = max(1.0 / float(self.rate_hz), now_t - self.prev_t)
            self.prev_t = now_t

            pose = self.get_robot_pose()
            tw = Twist()

            if pose is None:
                target_lin = 0.0
                target_ang = 0.0
            else:
                path_ang = self.compute_path_target_angle(pose)
                have_path = (path_ang is not None)

                lane_age = rospy.get_time() - self.last_lane_time
                lane_ok_now = (lane_age <= self.lane_detect_timeout) and self.have_lane_latched

                # Compute filtered lane angle inside control loop (thread-safe)
                with self.lock:
                    lane_angle_f = self.f_lane.filt(self.lane_angle_raw) if lane_ok_now else self.f_lane.y

                # Fusion
                if have_path and lane_ok_now:
                    combined = (self.alpha_lane * lane_angle_f) + ((1.0 - self.alpha_lane) * path_ang)
                    base_speed = self.lane_speed
                elif have_path:
                    combined = path_ang
                    base_speed = max(self.no_lane_speed, self.min_speed)
                elif lane_ok_now:
                    combined = lane_angle_f
                    base_speed = max(self.lane_speed, self.min_speed)
                else:
                    combined = 0.0
                    base_speed = 0.0 if self.stop_when_no_input else self.min_speed

                if abs(combined) < self.deadzone_angle:
                    combined = 0.0

                steer = self.f_steer.filt(combined)

                # Turn-dependent slowdown
                turn_scale = 1.0 / (1.0 + self.turn_slowdown_k * (abs(steer) ** 1.2))
                target_lin = self.clamp(base_speed * turn_scale, 0.0, self.max_speed)

                # Simple proportional steering -> angular vel
                k_ang = 1.0
                target_ang = self.clamp(-k_ang * steer, -self.max_ang, self.max_ang)

            # Accel / decel limiting
            max_dv = (self.max_lin_acc if target_lin >= self.prev_lin else self.max_lin_dec) * dt
            cmd_lin = self.rate_limit(self.prev_lin, target_lin, max_dv)

            # Angular rate limiting
            max_dw = self.max_ang_rate * dt
            cmd_ang = self.rate_limit(self.prev_ang, target_ang, max_dw)

            # Speed smoothing
            cmd_lin = self.f_speed.filt(cmd_lin)

            tw.linear.x = self.clamp(cmd_lin, 0.0, self.max_speed)
            tw.angular.z = self.clamp(cmd_ang, -self.max_ang, self.max_ang)

            self.cmd_pub.publish(tw)

            self.prev_lin = tw.linear.x
            self.prev_ang = tw.angular.z

            rate.sleep()


if __name__ == '__main__':
    try:
        node = SlamLkasFusion()
        node.control_loop()
    except rospy.ROSInterruptException:
        pass
