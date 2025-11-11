cmake_minimum_required(VERSION 3.0.2)
project(limo_slam_lkas)


find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  nav_msgs
  sensor_msgs
  tf
)


# add_message_files()
# generate_messages()

catkin_package(
  CATKIN_DEPENDS rospy std_msgs geometry_msgs nav_msgs sensor_msgs tf
)


catkin_install_python(PROGRAMS
  scripts/slam_lkas_fusion.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
