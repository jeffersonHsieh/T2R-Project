#!/bin/bash

# Create a new tmux session
session_name="vint_locobot_$(date +%s)"
tmux new-session -d -s $session_name

# Split the window into four panes
tmux selectp -t 0    # select the first (0) pane
tmux splitw -h -p 50 # split it into two halves
tmux selectp -t 0    # select the first (0) pane
tmux splitw -v -p 50 # split it into two halves

tmux selectp -t 2    # select the new, second (2) pane
tmux splitw -v -p 50 # split it into two halves
tmux selectp -t 0    # go back to the first pane

# Run the roslaunch command in the first pane
tmux select-pane -t 0
tmux send-keys "source ~/Desktop/AirSim/ros/devel/setup.bash" Enter
tmux send-keys "roslaunch airsim_ros_pkgs airsim_node.launch" Enter

# Run the navigate.py script with command line args in the second pane
tmux select-pane -t 1
tmux send-keys "conda activate t2r_vint" Enter
tmux send-keys "python explore_airsim.py $@" Enter

# Run the teleop.py script in the third pane
tmux select-pane -t 2
# tmux send-keys "source ~/Desktop/AirSim/ros/devel/setup.bash" Enter
# tmux send-keys "roslaunch airsim_ros_pkgs position_controller_simple.launch"
tmux send-keys "conda activate t2r_vint" Enter
tmux send-keys "python keyboard_sim_controller.py"

# Run the pd_controller.py script in the fourth pane
tmux select-pane -t 3
tmux send-keys "conda activate t2r_vint" Enter
tmux send-keys "python pd_controller.py" Enter

# Attach to the tmux session
tmux -2 attach-session -t $session_name
