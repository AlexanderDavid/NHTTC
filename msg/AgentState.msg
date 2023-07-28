# Constants for the kinematics, neccessary to understand the 
# control and state spaces
uint8 V = 0
uint8 A = 1
uint8 DD = 2
uint8 ADD = 3
uint8 CAR = 4
uint8 ACAR = 5

Header header
uint8 kinematics

# State and control spaces as they are defined in NH-TTC
float32[] state
float32[] control

# More traditional Odometry style reporting just in case that's useful
string child_frame_id
geometry_msgs/PoseWithCovariance pose
geometry_msgs/TwistWithCovariance twist