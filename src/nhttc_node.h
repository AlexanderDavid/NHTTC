#pragma once

#include <Eigen/Core>

#include <sgd/ttc_sgd_problem.h>
#include "nhttc_interface.h"

class NHTTCNode
{
public:
  std::vector<ros::Subscriber> subs_neighbor;
  std::vector<std::string> topics_neighbor;

  ros::Subscriber sub_goal, sub_wp, sub_pose;
  ros::Publisher pub_cmd, pub_viz;

  std::string cmd_vel_topic, odom_topic, neighbor_topic_root;

  int own_index;
  int count;

  Eigen::Vector2f goal;

  int solver_time;
  int num_agents_max;
  bool simulation;
  bool goal_received;
  float cutoff_dist;
  float steer_limit;
  float wheelbase;
  float turning_radius;
  float carrot_goal_ratio;
  float max_ttc;
  float safety_radius;
  SGDOptParams global_params;

  std::vector<Agent> agents; //all agents

  float speed_lim = 0.46f;
  bool obey_time;
  bool allow_reverse;
  bool adaptive_lookahead;

  std::vector<TTCObstacle*> obstacles = BuildObstacleList(agents);

  ros::master::V_TopicInfo master_topics;

  NHTTCNode(ros::NodeHandle &nh);

  void agent_setup(int i, int agent_type, bool reactive);
  void rpy_from_quat(float rpy[3],const nav_msgs::Odometry::ConstPtr& msg);
  void send_commands(float speed, float steer);
  void check_new_agents(ros::NodeHandle &nh);
  void viz_publish();
  void setup();
  void plan();

  void GoalCallback(const geometry_msgs::PoseStamped::ConstPtr& msg);
  void PoseCallback(const nav_msgs::Odometry::ConstPtr& msg);
  void NeighborCallback(const nav_msgs::Odometry::ConstPtr& msg, int neighbor_idx);
};