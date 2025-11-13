cmake_minimum_required(VERSION 3.0.2)
project(limo_slam_lkas)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  sensor_msgs
  nav_msgs
  tf
  dynamic_reconfigure
  message_generation
  diagnostic_updater
)

add_message_files(FILES
  LaneDebug.msg
)

generate_dynamic_reconfigure_options(
  cfg/FusionTuning.cfg
)

catkin_package(
  CATKIN_DEPENDS rospy std_msgs geometry_msgs sensor_msgs nav_msgs tf message_runtime
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

install(PROGRAMS
  scripts/slam_lkas_fusion.py
  scripts/lane_bev_node.py
  scripts/pure_pursuit_node.py
  scripts/diag_heartbeat.py
  scripts/bag_play_helper.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch config rviz
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
