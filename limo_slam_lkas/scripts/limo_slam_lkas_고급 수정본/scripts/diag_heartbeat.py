#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
from diagnostic_updater import Updater, FunctionDiagnosticTask

class Heartbeat:
    def __init__(self):
        rospy.init_node('diag_heartbeat')
        self.updater = Updater()
        self.updater.setHardwareID("limo_slam_lkas")
        self.updater.add(FunctionDiagnosticTask("heartbeat", self.heartbeat))
        self.rate = rospy.Rate(1)

    def heartbeat(self, stat):
        stat.summary(0, "OK")
        return stat

    def spin(self):
        while not rospy.is_shutdown():
            self.updater.update()
            self.rate.sleep()

if __name__=='__main__':
    Heartbeat().spin()
