#!/usr/bin/env python
# -*- coding: utf-8 -*-
# slam_lkas_fusion.py (smooth & fast) + Obstacle Stop + Marker Stop + IMU tilt slowdown
# ROS1 Melodic(Python2) compatible

import rospy
import math
import numpy as np
import threading
import time
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path
from sensor_msgs.msg import Image, LaserScan, Imu
import tf
from cv_bridge import CvBridge, CvBridgeError
import cv2

def normalize_angle(a):
    return math.atan2(math.sin(a), math.cos(a))

def clamp(x, lo, hi):
    return min(hi, max(lo, x))

class EWMA(object):
    def __init__(self, alpha=0.6, init_val=0.0):
        self.a = float(alpha)
        self.y = float(init_val)
        self.initialized = False
    def reset(self, val=0.0):
        self.y = float(val); self.initialized = True
    def filt(self, x):
        x = float(x)
        if not self.initialized:
            self.reset(x)
        self.y = self.a*self.y + (1.0-self.a)*x
        return self.y

class SlamLkasFusion(object):
    def __init__(self):
        rospy.init_node('slam_lkas_fusion')

        # === topics ===
        self.path_topic  = rospy.get_param('~path_topic', '/move_base/NavfnROS/plan')
        self.image_topic = rospy.get_param('~image_topic', '/camera/image_raw')
        self.cmd_topic   = rospy.get_param('~cmd_topic', '/cmd_vel')
        self.scan_topic  = rospy.get_param('~scan_topic', '/scan')
        self.imu_topic   = rospy.get_param('~imu_topic', '/imu')

        # === frames ===
        self.base_frame  = rospy.get_param('~base_frame', 'base_link')
        self.map_frame   = rospy.get_param('~map_frame',  'map')

        # === fusion loop ===
        self.rate_hz     = rospy.get_param('~rate', 15)
        self.alpha_lane  = rospy.get_param('~alpha_lane', 0.55)
        self.distance_ahead    = rospy.get_param('~distance_ahead', 0.9)
        self.distance_ahead_hi = rospy.get_param('~distance_ahead_hi', 1.3)

        # === speeds/limits ===
        self.max_speed   = rospy.get_param('~max_speed', 0.35)
        self.min_speed   = rospy.get_param('~min_speed', 0.12)
        self.lane_speed  = rospy.get_param('~lane_speed', 0.28)
        self.no_lane_speed = rospy.get_param('~no_lane_speed', 0.20)
        self.max_ang     = rospy.get_param('~max_angular', 1.2)

        self.max_lin_acc  = rospy.get_param('~max_lin_acc', 0.25)
        self.max_lin_dec  = rospy.get_param('~max_lin_dec', 0.35)
        self.max_ang_rate = rospy.get_param('~max_ang_rate', 1.5)

        self.turn_slowdown_k = rospy.get_param('~turn_slowdown_k', 0.9)
        self.deadzone_angle  = rospy.get_param('~deadzone_angle', 0.05)

        self.lane_detect_timeout = rospy.get_param('~lane_detect_timeout', 0.6)
        self.path_timeout        = rospy.get_param('~path_timeout', 0.6)
        self.lane_ok_hold        = rospy.get_param('~lane_ok_hold', 0.5)
        self.stop_when_no_input  = rospy.get_param('~stop_when_no_input', True)

        # === Obstacle stop (LiDAR) params ===
        self.stop_distance         = rospy.get_param('~stop_distance', 0.35)   # stop if anything closer than this
        self.obstacle_clear_dist   = rospy.get_param('~obstacle_clear_distance', 0.40)  # resume when distance > this
        self.front_angle_deg       = rospy.get_param('~front_angle_deg', 20)   # +/- degrees front cone

        # === Marker stop (template matching) params ===
        self.marker_template_path  = rospy.get_param('~marker_template_path', '')  # e.g. /home/agilex/markers/stop.png
        self.marker_match_thresh   = rospy.get_param('~marker_match_threshold', 0.75)
        self.marker_stop_time      = rospy.get_param('~marker_stop_time', 2.0)     # seconds to hold stop after detection

        # === IMU tilt slowdown params ===
        self.tilt_start_deg        = rospy.get_param('~tilt_start_deg', 5.0)   # begin slowing at this tilt
        self.tilt_max_deg          = rospy.get_param('~tilt_max_deg', 12.0)    # max slowdown at this tilt
        self.tilt_min_speed_factor = rospy.get_param('~tilt_min_speed_factor', 0.4)  # speed factor at max tilt
        self.tilt_hardstop_deg     = rospy.get_param('~tilt_hardstop_deg', 20.0)     # stop if tilt over this

        # === state ===
        self.lock = threading.RLock()

        self.current_path   = None
        self.last_path_time = 0.0

        self.lane_angle = 0.0
        self.have_lane_raw = False
        self.have_lane_latched = False
        self.last_lane_time = 0.0

        self.obstacle_active = False
        self.last_obstacle_time = 0.0
        self.last_marker_time = 0.0

        self.roll_deg  = 0.0
        self.pitch_deg = 0.0

        # filters
        self.f_lane  = EWMA(rospy.get_param('~lpf_lane_alpha', 0.5), 0.0)
        self.f_path  = EWMA(rospy.get_param('~lpf_path_alpha', 0.5), 0.0)
        self.f_steer = EWMA(rospy.get_param('~lpf_steer_alpha', 0.6), 0.0)
        self.f_speed = EWMA(rospy.get_param('~lpf_speed_alpha', 0.5), self.min_speed)

        # ramp
        self.prev_lin = 0.0
        self.prev_ang = 0.0
        self.prev_t   = time.time()

        # ROS I/O
        self.cmd_pub = rospy.Publisher(self.cmd_topic, Twist, queue_size=1)
        rospy.Subscriber(self.path_topic,  Path,  self.path_callback,  queue_size=1)
        rospy.Subscriber(self.image_topic, Image, self.image_callback, queue_size=1)
        rospy.Subscriber(self.scan_topic,  LaserScan, self.scan_callback, queue_size=1)
        rospy.Subscriber(self.imu_topic,   Imu,   self.imu_callback,   queue_size=1)

        self.tf_listener = tf.TransformListener()
        self.cv_bridge   = CvBridge()

        # load marker template (optional)
        self.marker_template = None
        if self.marker_template_path:
            try:
                tmp = cv2.imread(self.marker_template_path, cv2.IMREAD_GRAYSCALE)
                if tmp is None:
                    rospy.logwarn("Marker template not found at %s", self.marker_template_path)
                else:
                    self.marker_template = tmp
                    rospy.loginfo("Loaded marker template: %s (w=%d h=%d)", self.marker_template_path, tmp.shape[1], tmp.shape[0])
            except Exception as e:
                rospy.logwarn("Failed to load marker template: %s", str(e))

        rospy.loginfo("[slam_lkas_fusion] ready. path=%s img=%s scan=%s imu=%s",
                      self.path_topic, self.image_topic, self.scan_topic, self.imu_topic)

    # ---------------- callbacks ----------------
    def path_callback(self, msg):
        with self.lock:
            self.current_path = msg
            self.last_path_time = rospy.get_time()

    def image_callback(self, msg):
        try:
            bgr = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            rospy.logwarn_throttle(5.0, "CvBridge error: %s", str(e))
            return

        # lane angle
        angle, valid = self.detect_lane_angle(bgr)

        # marker stop (optional)
        if self.marker_template is not None:
            try:
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                if self.detect_marker(gray):
                    self.last_marker_time = rospy.get_time()
                    rospy.logwarn_throttle(1.0, "[SAFETY] Marker detected → STOP hold %.1fs", self.marker_stop_time)
            except Exception as e:
                rospy.logwarn_throttle(2.0, "Marker detection error: %s", str(e))

        with self.lock:
            if valid:
                self.lane_angle = angle
                self.have_lane_raw = True
                self.last_lane_time = rospy.get_time()
            else:
                self.have_lane_raw = False
            # hysteresis for lane validity
            if self.have_lane_raw:
                self.have_lane_latched = True
            else:
                if rospy.get_time() - self.last_lane_time > self.lane_ok_hold:
                    self.have_lane_latched = False

    def scan_callback(self, msg):
        # front cone min distance
        n = len(msg.ranges)
        if n == 0:
            return
        ang_inc = msg.angle_increment
        if ang_inc == 0:
            return
        # choose center indices +/- cone
        mid = n // 2
        cone_idx = int((self.front_angle_deg * math.pi/180.0) / abs(ang_inc))
        lo = max(0, mid - cone_idx)
        hi = min(n, mid + cone_idx)
        sub = msg.ranges[lo:hi]
        # filter NaN/Inf
        vals = [r for r in sub if (not math.isnan(r)) and (not math.isinf(r))]
        if not vals:
            return
        dmin = min(vals)

        # hysteresis for stop/resume
        if dmin < self.stop_distance:
            self.obstacle_active = True
            self.last_obstacle_time = rospy.get_time()
        elif dmin > self.obstacle_clear_dist:
            self.obstacle_active = False

    def imu_callback(self, msg):
        q = msg.orientation
        quat = [q.x, q.y, q.z, q.w]
        r, p, _ = tf.transformations.euler_from_quaternion(quat)
        self.roll_deg  = abs(r) * 180.0 / math.pi
        self.pitch_deg = abs(p) * 180.0 / math.pi

    # ---------------- vision helpers ----------------
    def detect_lane_angle(self, bgr):
        h, w = bgr.shape[:2]
        roi = bgr[int(h*0.45):h, 0:w]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        edges = cv2.Canny(blur, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=24, minLineLength=32, maxLineGap=22)
        if lines is None:
            return 0.0, False
        angles = []
        for l in lines:
            x1,y1,x2,y2 = l[0]
            dx = float(x2 - x1); dy = float(y2 - y1)
            if dx == 0 and dy == 0: continue
            ang = math.atan2(dy, dx)
            angles.append(normalize_angle((math.pi/2.0) - ang))
        if not angles:
            return 0.0, False
        med = float(np.median(np.array(angles)))
        med = self.f_lane.filt(med)
        return med, True

    def detect_marker(self, gray_img):
        # simple template matching
        if self.marker_template is None:
            return False
        th, tw = self.marker_template.shape[:2]
        gh, gw = gray_img.shape[:2]
        if gh < th or gw < tw:
            return False
        res = cv2.matchTemplate(gray_img, self.marker_template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, _, _ = cv2.minMaxLoc(res)
        return (max_val >= self.marker_match_thresh)

    # ---------------- TF & path helpers ----------------
    def get_robot_pose(self):
        try:
            now = rospy.Time(0)
            self.tf_listener.waitForTransform(self.map_frame, self.base_frame, now, rospy.Duration(0.3))
            (t, r) = self.tf_listener.lookupTransform(self.map_frame, self.base_frame, now)
            yaw = tf.transformations.euler_from_quaternion(r)[2]
            return t[0], t[1], yaw
        except Exception:
            return None

    def compute_path_target_angle(self, robot_pose):
        with self.lock:
            path = self.current_path
            last_time = self.last_path_time
        if path is None: return None
        if rospy.get_time() - last_time > self.path_timeout: return None
        if len(path.poses) == 0: return None

        rx, ry, ryaw = robot_pose
        current_speed_est = max(self.prev_lin, self.min_speed)
        la = self.distance_ahead + (self.distance_ahead_hi - self.distance_ahead) * min(1.0, current_speed_est/self.max_speed)

        best_pt = None
        for ps in path.poses:
            px = ps.pose.position.x; py = ps.pose.position.y
            if math.hypot(px - rx, py - ry) >= la:
                best_pt = (px, py); break
        if best_pt is None:
            last = path.poses[-1].pose
            best_pt = (last.position.x, last.position.y)

        tx, ty = best_pt
        err = normalize_angle(math.atan2(ty - ry, tx - rx) - ryaw)
        err = self.f_path.filt(err)
        return err

    # ---------------- main control loop ----------------
    def rate_limit(self, prev, target, max_delta):
        if target > prev:  return min(prev + max_delta, target)
        else:              return max(prev - max_delta, target)

    def control_loop(self):
        rate = rospy.Rate(self.rate_hz)
        while not rospy.is_shutdown():
            now_t = time.time()
            dt = max(1.0/self.rate_hz, now_t - self.prev_t)
            self.prev_t = now_t

            tw = Twist()
            pose = self.get_robot_pose()

            if pose is None:
                target_lin = 0.0
                target_ang = 0.0
            else:
                path_ang = self.compute_path_target_angle(pose)
                have_path = (path_ang is not None)

                lane_age = rospy.get_time() - self.last_lane_time
                lane_ok_now = (lane_age <= self.lane_detect_timeout) and self.have_lane_latched

                # base fusion
                if have_path and lane_ok_now:
                    combined = (self.alpha_lane * self.f_lane.y) + ((1.0 - self.alpha_lane) * path_ang)
                    base_speed = self.lane_speed
                elif have_path:
                    combined = path_ang
                    base_speed = max(self.no_lane_speed, self.min_speed)
                elif lane_ok_now:
                    combined = self.f_lane.y
                    base_speed = max(self.lane_speed, self.min_speed)
                else:
                    combined = 0.0
                    base_speed = 0.0 if self.stop_when_no_input else self.min_speed

                # deadzone
                if abs(combined) < self.deadzone_angle:
                    combined = 0.0

                # steering smooth
                steer = self.f_steer.filt(combined)

                # speed from curve
                turn_scale = 1.0 / (1.0 + self.turn_slowdown_k * (abs(steer) ** 1.2))
                target_lin = clamp(base_speed * turn_scale, 0.0, self.max_speed)
                k_ang = 1.0
                target_ang = clamp(-k_ang * steer, -self.max_ang, self.max_ang)

                # === IMU tilt slowdown/stop ===
                tilt_mag = max(self.roll_deg, self.pitch_deg)
                if tilt_mag >= self.tilt_hardstop_deg:
                    target_lin = 0.0
                    target_ang = 0.0
                    rospy.logwarn_throttle(1.0, "[SAFETY] Tilt %.1f° >= hardstop %.1f° → STOP", tilt_mag, self.tilt_hardstop_deg)
                elif tilt_mag > self.tilt_start_deg:
                    # linear interpolation between start and max
                    t = (tilt_mag - self.tilt_start_deg) / float(max(1e-6, (self.tilt_max_deg - self.tilt_start_deg)))
                    t = clamp(t, 0.0, 1.0)
                    factor = 1.0 - (1.0 - self.tilt_min_speed_factor)*t
                    target_lin *= factor

                # === Obstacle & Marker final stops ===
                marker_active = (rospy.get_time() - self.last_marker_time) <= self.marker_stop_time
                if self.obstacle_active or marker_active:
                    target_lin = 0.0
                    target_ang = 0.0
                    if self.obstacle_active:
                        rospy.logwarn_throttle(1.0, "[SAFETY] Obstacle in front → STOP")
                    if marker_active:
                        rospy.logwarn_throttle(1.0, "[SAFETY] Marker hold-stop active")

            # rate limits & smoothing
            max_dv = (self.max_lin_acc if target_lin >= self.prev_lin else self.max_lin_dec) * dt
            cmd_lin = self.rate_limit(self.prev_lin, target_lin, max_dv)
            max_dw = self.max_ang_rate * dt
            cmd_ang = self.rate_limit(self.prev_ang, target_ang, max_dw)
            cmd_lin = self.f_speed.filt(cmd_lin)

            tw.linear.x  = clamp(cmd_lin, 0.0, self.max_speed)
            tw.angular.z = clamp(cmd_ang, -self.max_ang, self.max_ang)

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
