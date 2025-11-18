#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SLAM + LKAS Fusion Controller (ROS1, smooth version)
- SLAM 경로(Path) + 차선(Lane) 조향을 융합해서 /cmd_vel 생성
- 조향/속도 LPF + rate limit 로 "틱틱 끊기는" 움직임 최소화
- TF 에러, Path 끊김, Lane 미검출 등 예외 상황에 안전하게 동작

주요 개선점:
- lane angle: median + EWMA 2중 필터
- steering 값 자체에 rate limit 적용
- path steering, speed smoothing 강화
- deadzone 확대로 직진 떨림 제거
"""

import math
import threading
from collections import deque
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
    """각도를 -pi ~ +pi 범위로 정규화."""
    return math.atan2(math.sin(a), math.cos(a))


class EWMA(object):
    """지수 이동 평균 필터 (저역통과 필터)."""
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

        # === 기본 파라미터 ===
        self.rate_hz = rospy.get_param('~rate', 15)

        self.path_topic = rospy.get_param('~path_topic', '/move_base/NavfnROS/plan')
        self.image_topic = rospy.get_param('~image_topic', '/camera/image_raw')
        self.cmd_topic = rospy.get_param('~cmd_topic', '/cmd_vel')
        self.base_frame = rospy.get_param('~base_frame', 'base_link')
        self.map_frame = rospy.get_param('~map_frame', 'map')

        # 입력이 없을 때 정지할지 여부
        self.stop_when_no_input = rospy.get_param('~stop_when_no_input', True)

        # === 속도/조향 제한 ===
        # 조금 더 부드러운 움직임을 위해 가감속/필터 기본값을 기존보다 약간 강화
        self.max_speed = rospy.get_param('~max_speed', 0.35)
        self.min_speed = rospy.get_param('~min_speed', 0.12)
        self.lane_speed = rospy.get_param('~lane_speed', 0.28)
        self.no_lane_speed = rospy.get_param('~no_lane_speed', 0.20)
        self.max_ang = rospy.get_param('~max_angular', 1.2)

        # 선속도 가속/감속 제한 (틱틱 방지용으로 다소 보수적으로)
        self.max_lin_acc = rospy.get_param('~max_lin_acc', 0.15)   # m/s^2
        self.max_lin_dec = rospy.get_param('~max_lin_dec', 0.25)   # m/s^2

        # 각속도 변화율 제한
        self.max_ang_rate = rospy.get_param('~max_ang_rate', 1.5)  # rad/s^2

        # 조향(steer) 자체 rate limit (중요!)
        self.max_steer_rate = rospy.get_param('~max_steer_rate', 0.25)  # rad/s

        # === Fusion / Lookahead 파라미터 ===
        self.alpha_lane = rospy.get_param('~alpha_lane', 0.55)  # lane 가중치
        self.distance_ahead = rospy.get_param('~distance_ahead', 0.9)
        self.distance_ahead_hi = rospy.get_param('~distance_ahead_hi', 1.3)
        self.turn_slowdown_k = rospy.get_param('~turn_slowdown_k', 0.9)

        # === 입력 신뢰성 (타임아웃 등) ===
        self.lane_detect_timeout = rospy.get_param('~lane_detect_timeout', 0.6)
        self.path_timeout = rospy.get_param('~path_timeout', 0.6)
        self.lane_ok_hold = rospy.get_param('~lane_ok_hold', 0.5)

        # 작은 조향은 무시해서 떨림 제거 (deadzone)
        self.deadzone_angle = rospy.get_param('~deadzone_angle', 0.08)

        # === 필터 파라미터 ===
        # 기본값을 약간 더 "부드럽게" 쪽으로 조정
        self.lpf_lane_a = rospy.get_param('~lpf_lane_alpha', 0.5)
        self.lpf_path_a = rospy.get_param('~lpf_path_alpha', 0.5)
        self.lpf_steer_a = rospy.get_param('~lpf_steer_alpha', 0.6)
        self.lpf_speed_a = rospy.get_param('~lpf_speed_alpha', 0.7)

        # lane median filter window size
        self.lane_window_size = int(rospy.get_param('~lane_window_size', 5))

        # === 이미지 처리 파라미터 ===
        self.roi_top_ratio = rospy.get_param('~roi_top_ratio', 0.45)
        self.canny_lo = rospy.get_param('~canny_lo', 50)
        self.canny_hi = rospy.get_param('~canny_hi', 150)
        self.hough_thresh = rospy.get_param('~hough_thresh', 24)
        self.hough_min_len = rospy.get_param('~hough_min_len', 32)
        self.hough_max_gap = rospy.get_param('~hough_max_gap', 22)

        # === 상태 변수 ===
        self.lock = threading.RLock()

        # path 관련
        self.current_path: Optional[Path] = None
        self.last_path_time = 0.0

        # lane 관련
        self.last_lane_time = 0.0
        self.have_lane_raw = False
        self.have_lane_latched = False
        self.lane_angle_raw = 0.0

        # lane angle 최근값 모음 (median filter용)
        self.lane_window = deque(maxlen=self.lane_window_size)

        # 필터 객체
        self.f_lane = EWMA(self.lpf_lane_a, 0.0)
        self.f_path = EWMA(self.lpf_path_a, 0.0)
        self.f_steer = EWMA(self.lpf_steer_a, 0.0)
        self.f_speed = EWMA(self.lpf_speed_a, self.min_speed)

        self.prev_lin = 0.0
        self.prev_ang = 0.0
        self.prev_steer = 0.0  # steering rate-limit 기준값
        self.prev_t = rospy.Time.now().to_sec()

        # ROS I/O 설정
        self.cmd_pub = rospy.Publisher(self.cmd_topic, Twist, queue_size=1)
        rospy.Subscriber(self.path_topic, Path, self.path_callback, queue_size=1)
        rospy.Subscriber(self.image_topic, Image, self.image_callback, queue_size=1)

        self.tf_listener = tf.TransformListener()
        self.cv_bridge = CvBridge()

        rospy.loginfo("[slam_lkas_fusion] ready. path_topic=%s image_topic=%s",
                      self.path_topic, self.image_topic)

    # ---------------- Path 콜백 ----------------
    def path_callback(self, msg: Path):
        with self.lock:
            self.current_path = msg
            self.last_path_time = rospy.get_time()

    # ---------------- Image 콜백 ----------------
    def image_callback(self, msg: Image):
        try:
            # BGR 우선 시도
            try:
                cv_img = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            except CvBridgeError:
                # mono 등 다른 포맷일 경우 fallback
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
                # raw angle을 window에 쌓아 median filter에 사용
                self.lane_window.append(self.lane_angle_raw)
            else:
                self.have_lane_raw = False

            # latch (히스테리시스): 일시적인 미검출은 바로 끄지 않음
            if self.have_lane_raw:
                self.have_lane_latched = True
            else:
                if rospy.get_time() - self.last_lane_time > self.lane_ok_hold:
                    self.have_lane_latched = False

    # ---------------- Vision: 차선 각도 검출 ----------------
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
            # 수직 기준 차선 방향과의 상대 각도
            angle_rel_vertical = normalize_angle((math.pi / 2.0) - ang)
            # 거의 수평인 라인은 노이즈로 간주
            if abs(angle_rel_vertical) > math.radians(80):
                continue
            angles.append(angle_rel_vertical)

        if not angles:
            return 0.0, False

        med = float(np.median(np.array(angles, dtype=np.float32)))
        return med, True

    # ---------------- TF & Path 처리 ----------------
    def get_robot_pose(self) -> Optional[Tuple[float, float, float]]:
        """map 프레임 기준 base_link의 (x, y, yaw) 리턴."""
        try:
            now = rospy.Time(0)
            self.tf_listener.waitForTransform(self.map_frame, self.base_frame,
                                              now, rospy.Duration(0.3))
            (trans, rot) = self.tf_listener.lookupTransform(self.map_frame,
                                                            self.base_frame, now)
            x, y = trans[0], trans[1]
            yaw = tf.transformations.euler_from_quaternion(rot)[2]
            if not all(np.isfinite([x, y, yaw])):
                return None
            return x, y, yaw
        except Exception as e:
            rospy.logwarn_throttle(2.0, "TF lookup failed: %s", str(e))
            return None

    def compute_path_target_angle(self, robot_pose: Tuple[float, float, float]) -> Optional[float]:
        """SLAM path로부터 목표 yaw error 계산."""
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

        # 현재 속도에 따라 lookahead 거리 조금 가변
        current_speed_est = max(self.prev_lin, self.min_speed)
        la_ratio = min(1.0, current_speed_est / max(self.max_speed, 1e-6))
        la = self.distance_ahead + (self.distance_ahead_hi - self.distance_ahead) * la_ratio

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
        # path steering도 EWMA로 필터링
        err = self.f_path.filt(err)
        return err

    # ---------------- Helper 함수들 ----------------
    @staticmethod
    def clamp(x: float, lo: float, hi: float) -> float:
        return min(hi, max(lo, x))

    @staticmethod
    def rate_limit(prev: float, target: float, max_delta: float) -> float:
        if target > prev:
            return min(prev + max_delta, target)
        else:
            return max(prev - max_delta, target)

    # ---------------- 메인 제어 루프 ----------------
    def control_loop(self):
        rate = rospy.Rate(self.rate_hz)

        # TF / 토픽들 준비될 시간 약간 대기
        rospy.sleep(0.25)

        while not rospy.is_shutdown():
            now_t = rospy.Time.now().to_sec()
            dt = max(1.0 / float(self.rate_hz), now_t - self.prev_t)
            self.prev_t = now_t

            pose = self.get_robot_pose()
            tw = Twist()

            if pose is None:
                # pose가 없으면 안전하게 정지 혹은 저속 직진
                target_lin = 0.0
                target_ang = 0.0
            else:
                path_ang = self.compute_path_target_angle(pose)
                have_path = (path_ang is not None)

                if have_path and abs(path_ang) < 0.02:
                    # path의 작은 흔들림은 0으로 처리 (직진 떨림 제거)
                    path_ang = 0.0

                lane_age = rospy.get_time() - self.last_lane_time
                lane_ok_now = (lane_age <= self.lane_detect_timeout) and self.have_lane_latched

                # lane angle 필터링 (median + EWMA)
                with self.lock:
                    if lane_ok_now and len(self.lane_window) > 0:
                        lane_med = float(np.median(np.array(self.lane_window, dtype=np.float32)))
                        lane_angle_f = self.f_lane.filt(lane_med)
                    else:
                        # lane이 불안정하면 이전 필터 값 유지
                        lane_angle_f = self.f_lane.y

                # === Fusion ===
                if have_path and lane_ok_now:
                    # 둘 다 신뢰 가능할 때 융합
                    combined = (self.alpha_lane * lane_angle_f) + ((1.0 - self.alpha_lane) * path_ang)
                    base_speed = self.lane_speed
                elif have_path:
                    # SLAM만 있을 때
                    combined = path_ang
                    base_speed = max(self.no_lane_speed, self.min_speed)
                elif lane_ok_now:
                    # lane만 있을 때
                    combined = lane_angle_f
                    base_speed = max(self.lane_speed, self.min_speed)
                else:
                    # 아무 입력도 없을 때
                    combined = 0.0
                    base_speed = 0.0 if self.stop_when_no_input else self.min_speed

                # deadzone 적용 (작은 조향은 0으로)
                if abs(combined) < self.deadzone_angle:
                    combined = 0.0

                # steer LPF + steering rate limit
                steer_raw = self.f_steer.filt(combined)

                # steering 각도 변화율 제한 (틱틱 방지 핵심)
                max_dsteer = self.max_steer_rate * dt
                steer = self.rate_limit(self.prev_steer, steer_raw, max_dsteer)
                self.prev_steer = steer

                # 회전량이 클수록 속도 줄이기
                turn_scale = 1.0 / (1.0 + self.turn_slowdown_k * (abs(steer) ** 1.2))
                target_lin = self.clamp(base_speed * turn_scale, 0.0, self.max_speed)

                # 간단한 비례제어: steer -> angular.z
                k_ang = 1.0
                target_ang = self.clamp(-k_ang * steer, -self.max_ang, self.max_ang)

            # === 선속도 가감속 제한 ===
            max_dv = (self.max_lin_acc if target_lin >= self.prev_lin else self.max_lin_dec) * dt
            cmd_lin = self.rate_limit(self.prev_lin, target_lin, max_dv)

            # === 각속도 rate limit ===
            max_dw = self.max_ang_rate * dt
            cmd_ang = self.rate_limit(self.prev_ang, target_ang, max_dw)

            # === 속도 필터링 (더 부드럽게) ===
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
