#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy, math
from nav_msgs.msg import Path
from std_msgs.msg import Float32

class PurePursuitAngle:
    def __init__(self):
        rospy.init_node('pure_pursuit')
        self.pub = rospy.Publisher('/pp_yaw_error', Float32, queue_size=1)
        self.sub = rospy.Subscriber('/move_base/NavfnROS/plan', Path, self.cb, queue_size=1)
        self.lookahead = rospy.get_param('~lookahead', 1.0)

    def cb(self, path):
        if len(path.poses)==0:
            return
        # Simple: target last pose (replace with arc-length ahead in real impl)
        tx, ty = path.poses[-1].pose.position.x, path.poses[-1].pose.position.y
        err = math.atan2(ty, tx)  # robot yaw=0 frame simplification
        self.pub.publish(err)

if __name__=='__main__':
    PurePursuitAngle(); rospy.spin()
