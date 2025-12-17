<launch>

  <!-- ========================================================= -->
  <!-- 1) LIMO 본체 -->
  <!-- ========================================================= -->
  <include file="$(find limo_bringup)/launch/limo_start.launch">
    <arg name="pub_odom_tf" value="false"/>
  </include>

  <!-- ========================================================= -->
  <!-- 2) 카메라 -->
  <!-- ========================================================= -->
  <include file="$(find astra_camera)/launch/dabai_u3.launch"/>

  <!-- ========================================================= -->
  <!-- 3) Navigation (move_base + costmaps) -->
  <!-- ========================================================= -->
  <include file="$(find limo_bringup)/launch/limo_navigation_diff.launch"/>

  <!-- ========================================================= -->
  <!-- 4) LKAS (Lane Center Detection) -->
  <!-- ========================================================= -->
  <node pkg="limo_slam_lkas"
        type="lane_center_bev.py"
        name="lane_center_bev"
        output="screen">
    <param name="image_topic" value="/camera/rgb/image_raw"/>
  </node>

  <!-- ========================================================= -->
  <!-- 5) LKAS + SLAM Guard (⭐ LEFT-BIASED 버전) -->
  <!-- ========================================================= -->
  <node pkg="limo_slam_lkas"
        type="lkas_slam_guard.py"
        name="lkas_slam_guard"
        output="screen">

    <!-- 기본 토픽 -->
    <param name="slam_cmd_topic" value="/cmd_vel_slam"/>
    <param name="out_cmd_topic"  value="/cmd_vel"/>

    <!-- LKAS ENABLE -->
    <param name="enable_lkas" value="true"/>

    <!-- ===============================
         차선 이탈 기준 (좌/우 분리)
    =============================== -->
    <!-- LEFT : 더 민감 -->
    <param name="depart_threshold_left"  value="0.18"/>
    <param name="clear_threshold_left"   value="0.14"/>

    <!-- RIGHT : 기존 수준 -->
    <param name="depart_threshold_right" value="0.25"/>
    <param name="clear_threshold_right"  value="0.18"/>

    <!-- ===============================
         차선 보정 gain (좌/우 분리)
    =============================== -->
    <param name="lane_gain_left"  value="1.8"/>
    <param name="lane_gain_right" value="1.3"/>

    <!-- ===============================
         SLAM ↔ LKAS 혼합 비율
    =============================== -->
    <param name="lane_mix" value="0.75"/>

    <!-- ===============================
         Override 시 동작
    =============================== -->
    <param name="keep_forward_in_override" value="true"/>
    <param name="min_forward" value="0.06"/>

    <!-- ===============================
         Angular 제한 / 필터
    =============================== -->
    <param name="max_ang" value="0.9"/>
    <param name="deadzone" value="0.04"/>

    <param name="ang_alpha" value="0.65"/>
    <param name="offset_alpha" value="0.70"/>

    <!-- ===============================
         Timeout / Rate
    =============================== -->
    <param name="slam_timeout" value="0.5"/>
    <param name="lane_timeout" value="0.5"/>
    <param name="rate" value="20"/>

  </node>

  <!-- ========================================================= -->
  <!-- 6) ROSBRIDGE (Flutter 앱 연결) -->
  <!-- ========================================================= -->
  <node pkg="rosbridge_server"
        type="rosbridge_websocket"
        name="rosbridge_websocket"
        output="screen">
    <param name="port" value="9090"/>
  </node>

  <!-- ========================================================= -->
  <!-- 7) MAP HTTP 서버 (앱 지도용) -->
  <!-- ========================================================= -->
  <node pkg="limo_slam_lkas"
        type="map_http_server.py"
        name="map_http_server"
        output="screen">
    <param name="map_file" value="$(find limo_bringup)/maps/test_map.yaml"/>
    <param name="port" value="8000"/>
  </node>

  <!-- ========================================================= -->
  <!-- 8) GOAL MANAGER (앱 버튼 → move_base goal) -->
  <!-- ========================================================= -->
  <node pkg="limo_slam_lkas"
        type="goal_manager.py"
        name="goal_manager"
        output="screen"
        launch-prefix="bash -c 'sleep 5; exec $0 $@'"/>
</launch>
