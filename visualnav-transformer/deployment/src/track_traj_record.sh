#!/bin/bash

# this assumes you have airsim, <airsim rosnode> launched, and the drone taken-off

# check number of arguments and print usage if necessary
if [ $# -ne 2 ]; then
  echo "Usage: $0 <session_name> <trajectory_dir>"
  exit 1
fi


# Create a new tmux session
# start a session on another window like so `tmux new-session -d -s "record_bag_$(date +%s)"`
session_name=$1

TRAJECTORY_DIR=$2
TRACK_SCRIPT="python track_traj_cheating.py"
ros_bridge_setup="~/Desktop/AirSim/ros/devel/setup.bash"

# Split the window into three panes
# some panes' last command won't be executed. 
# you should press enter in order to make sure they execute successfully

tmux selectp -t 0    # select the first (0) pane
tmux splitw -v -p 50 # split it into two halves
tmux selectp -t 0    # go back to the first pane
tmux splitw -h -p 50 # split it into two halves

tmux select-pane -t 0
tmux send-keys "source ${ros_bridge_setup}" Enter
tmux send-keys "roslaunch airsim_ros_pkgs airsim_node.launch" Enter

# Run the teleop.py script in the second pane
tmux select-pane -t 1
tmux send-keys "source ${ros_bridge_setup}" Enter
# takeoff service
tmux send-keys "rosservice call /airsim_node/SimpleFlight/takeoff \"waitOnLastTask: true\""



tmux select-pane -t 2
tmux send-keys "cd ../topomaps/bags" Enter

for file in "$TRAJECTORY_DIR"/*
    do 
        name=$(basename $file)
        tmux select-pane -t 1
        tmux send-keys "echo \"Tracking trajectory: $file\"" Enter
        tmux send-keys "$TRACK_SCRIPT \"$file\";touch ${name}.done" Enter

        tmux select-pane -t 2
        tmux send-keys "rosbag record /airsim_node/SimpleFlight/front_center/Scene/compressed \
        /airsim_node/SimpleFlight/front_center/Scene/compressedDepth \
        /airsim_node/SimpleFlight/imu/imu \
        /airsim_node/SimpleFlight/odom_local_ned \
        /airsim_node/SimpleFlight/gps/gps \
        /airsim_node/SimpleFlight/global_gps \
        /airsim_node/SimpleFlight/left_camera/Scene/compressed \
        /airsim_node/SimpleFlight/left_camera/Scene/compressedDepth \
        /airsim_node/SimpleFlight/right_camera/Scene/compressedDepth \
        /airsim_node/SimpleFlight/right_camera/Scene/compressed -o $1" # change topic if necessary
        echo "Waiting for trajectory to finish..."
        while [ ! -f "../topomaps/bags/$name.done" ]
        do
            sleep 1
        done
        
        # send ctrl-c to the rosbag record command
        tmux select-pane -t 2
        tmux send-keys C-c
        tmux send-keys "echo \"Trajectory $file recorded\"" Enter

    done
