#pragma once

#include <Eigen/Core>

#include <nhttc_ros/AgentState.h>

#include <nhttc_interface/agent.h>
#include <sgd/ttc_sgd_problem.h>
#include <nhttc_interface/util.h>
#include <nhttc_interface/agent.h>

class NHTTCNode
{
private:
  std::vector<ros::Subscriber> _subs_neighbor;
  std::vector<std::string> _topics_neighbor;

  ros::Subscriber _sub_goal, _sub_wp, _sub_pose;
  ros::Publisher _pub_cmd, _pub_viz, pub_nhttc_pose;

  std::string odom_topic;
  std::string neighbor_topic_root;

  int own_index;

  Eigen::Vector2f goal;

  // NHTTC Hyper Parameters
  int solver_time;
  int num_agents_max;
  bool goal_received;
  float cutoff_dist;
  float steer_limit;
  float turning_radius;
  float carrot_goal_ratio;
  float max_ttc;
  float safety_radius;
  float speed_lim;
  bool obey_time;
  bool allow_reverse;
  bool adaptive_lookahead;
  SGDOptParams global_params;

  double pose_timeout;

  std::vector<Agent> agents; 
  std::vector<TTCObstacle*> obstacles{};



  void agentSetup(int i, AType agent_type, bool reactive);
  void RPYFromQuat(float rpy[3],const nav_msgs::Odometry::ConstPtr& msg);
  void sendCommands(float speed, float steer);
  void checkNewAgents(ros::NodeHandle &nh);
  void publishNHTTCPose();

  void goalCallback(const geometry_msgs::PoseStamped::ConstPtr& msg);
  void poseCallback(const nav_msgs::Odometry::ConstPtr& msg);
  void neighborCallback(const nav_msgs::Odometry::ConstPtr& msg, int neighbor_idx);
  void NHTTCNeighborCallback(const nhttc_ros::AgentState::ConstPtr& msg, int neighbor_idx);

public:
  NHTTCNode(ros::NodeHandle &nh);

  void publishDebugViz();
  void setup();
  void plan();
  inline bool ready() { return own_index != -1; };
};