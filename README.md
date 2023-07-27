# nhttc_ros

nhttc_ros is a multi-agent decentralized navigation system meant to take in goals from higher level planner and follow them while avoiding collisions with other agents. It is built on top of [MUSHR fork](https://github.com/prl-mushr/nhttc_ros) of the [NHTTC system](https://github.com/davisbo/NHTTC). Below are install and run instructions.

## Installation:
(Assuming some ROS1 workspace exists at `./catkin_ws`)

``` bash
$ cd catkin_ws/src
$ git clone https://github.com/AlexanderDavid/nhttc_ros.git
$ cd ..
$ catkin_make
```

## Todo:
- [ ] Support multiple waypoints in optimization
- [ ] Documentation for adding new kinematics model
    - [ ] Documentation in general
- [ ] Remove neighbor with message timeout
- [ ] Write a plugin for MoveBase