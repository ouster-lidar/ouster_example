# Example ROS Node

## Contents
* `ouster_ros/` contains sample code for publishing sensor data as standard ROS topics

## Operating System Support
* Tested with [ROS Kinetic](http://wiki.ros.org/kinetic/Installation/Ubuntu) on Ubuntu 16.04, [ROS
  Melodic](http://wiki.ros.org/melodic/Installation/Ubuntu) on Ubuntu 18.04, and [ROS
  Noetic](http://wiki.ros.org/noetic/Installation/Ubuntu) on Ubuntu 20.04

## Building
* Note: In the following instructions, `/path/to/ouster_example` should be replaced with the actual
  path where you you've cloned the repository.

### Build Dependencies
* `ouster_ros/` requires ROS libraries; call `sudo apt-get install ros-<ROS-VERSION>-ros-core
  ros-<ROS-VERSION>-pcl-ros ros-<ROS-VERSION>-tf2-geometry-msgs`, and
  optionally `sudo apt install ros-<ROS-VERSION>-rviz` for visualization using ROS (rviz),
  where `<ROS-VERSION>` is `kinetic`, `melodic`, or `noetic`.
* Install additional build dependencies with: `sudo apt-get install libglfw3-dev libglew-dev
  libtclap-dev`

### Building the Sample ROS Nodes
* Be sure to source the ROS setup script before building: `source /opt/ros/*/setup.bash`
* Set up the ROS workspace with: `mkdir -p myworkspace/src && cd myworkspace && ln -s
  /path/to/ouster_example ./src/`. Do not create your workspace inside the cloned repository, as
  this will confuse the ROS build system.
* Then, build with `catkin_make -DCMAKE_BUILD_TYPE=Release`

## Running Sample ROS Nodes
* In each new terminal for each command below, make sure to source ROS environment with `source /path/to/myworkspace/devel/setup.bash` where
  `/path/to/myworkspace` is where the path to the workspace directory that was created when
  building the ROS nodes (NOTE: be aware that this is different than previous setup.bash for building)

### Connecting to a Live Sensor
* Make sure the sensor is connected to the network and has obtained a DHCP
  lease. See section 3.1 in the accompanying [software user guide](https://www.ouster.com/resources)
  for more details
* To publish ROS topics from a running sensor from within the `ouster_ros` directory (inside `/path/to/ouster_example/`):
    - Run `roslaunch ouster.launch sensor_hostname:=<sensor_hostname>
      udp_dest:=<udp_data_dest_ip> lidar_mode:=<lidar_mode> viz:=<viz>`where:
        - `<sensor_hostname>` can be the hostname (os-991xxxxxxxxx) or IP of the sensor
        - `<udp_data_dest_ip>` is the IP to which the sensor should send data
        - `<lidar_mode>` is one of `512x10`, `512x20`, `1024x10`, `1024x20`, or `2048x10`
        - `<viz>` is either `true` or `false`. If true, a window should open and start
          displaying data after a few seconds
    - See [ouster_viz](../ouster_viz/README.md) for documentation on using the visualizer

### Playing Back Recorded Data
* To publish ROS topics from recorded data from within the `ouster_ros` directory:
    - Run `roslaunch ouster.launch replay:=true
      sensor_hostname:=<sensor_hostname>`
    - In a second terminal run `rosbag play --clock <bagfile>`
    - Note: `os_node` reads and writes metadata to `${ROS_HOME}` to enable
      accurately replaying raw data. By default, the name of this file is based
      on the hostname of the sensor. The location of this file can be overridden
      using the `metadata:=<path_to_file>` flag
    - If a metadata file is not available, the visualizer will default to
      1024x10. This can be overridden with the `lidar_mode`
      parameter. Visualizer output will only be correct if the same `lidar_mode`
      parameter is used for both recording and replay

### Visualizing Data
* To display sensor output using the Ouster Visualizer in ROS:
    - set `<viz>` to `true` in the roslaunch command.
    - Example command:
        - `roslaunch ouster.launch sensor_hostname:=os-991234567890.local
          udp_dest:=192.0.2.1 lidar_mode:=2048x10 viz:=true`
* To display sensor output using built-in ROS tools (rviz):
    - Follow the instructions above for running the example ROS code with a
      sensor or recorded data
    - To visualize output using rviz, run `rviz -d /path/to/ouster_ros/viz.rviz`
      in another terminal
    - To view lidar intensity/noise/range images, add `image:=true` to either of
      the `roslaunch` commands above

### Recording Data
* To record raw sensor output use [rosbag record](https://wiki.ros.org/rosbag/Commandline#rosbag_record):
    - In another terminal instance, run `rosbag record /os_node/imu_packets
     /os_node/lidar_packets`
    - This will save a .bag file of recorded data in that directory