#include <string>
#include <sstream>

#include "ros/ros.h"
#include "ros/master.h"
#include <boost/algorithm/string.hpp>

#include "nav_msgs/Odometry.h"
#include "geometry_msgs/Twist.h"
#include "geometry_msgs/Point.h"
#include "geometry_msgs/PoseStamped.h"
#include "visualization_msgs/MarkerArray.h"
#include "visualization_msgs/Marker.h"

#include "nhttc_node.h"


void NHTTCNode::viz_publish() 
{
  visualization_msgs::MarkerArray marr{};

  // Publish markers for all the agents
  for (size_t i = 0; i < agents.size(); i++)
  {
    visualization_msgs::Marker pose_marker{};
    pose_marker.header.frame_id = "odom";
    pose_marker.header.stamp = ros::Time::now();

    pose_marker.ns = "nhttc_agent";
    pose_marker.id = 2 + i;

    pose_marker.type = visualization_msgs::Marker::CUBE;
    pose_marker.action = visualization_msgs::Marker::ADD;

    Eigen::Vector2f agent_state = agents[i].GetProblem()->params.x_0.head(2);
    pose_marker.pose.position.x = agent_state[0];
    pose_marker.pose.position.y = agent_state[1];
    pose_marker.pose.orientation.w = 1.0;

    pose_marker.scale.x = 0.3;
    pose_marker.scale.y = 0.3;
    pose_marker.scale.z = 0.3;

    pose_marker.color.r = 0.0f;
    pose_marker.color.g = i == own_index ? 1.0f : 0.0f;
    pose_marker.color.b = i == own_index ? 0.0f : 1.0f;
    pose_marker.color.a = 1.0;

    pose_marker.lifetime = ros::Duration();

    marr.markers.push_back(pose_marker);
  }

  // Goal Marker
  visualization_msgs::Marker goal_marker{};
  goal_marker.header.frame_id = "odom";
  goal_marker.header.stamp = ros::Time::now();

  goal_marker.ns = "nhttc";
  goal_marker.id = 0;

  goal_marker.type = visualization_msgs::Marker::SPHERE;
  goal_marker.action = visualization_msgs::Marker::ADD;

  Eigen::Vector2f agent_goal = agents[own_index].GetGoal();
  goal_marker.pose.position.x = agent_goal[0];
  goal_marker.pose.position.y = agent_goal[1];
  goal_marker.pose.orientation.w = 1.0;

  goal_marker.scale.x = 0.3;
  goal_marker.scale.y = 0.3;
  goal_marker.scale.z = 0.3;

  goal_marker.color.r = 0.5f;
  goal_marker.color.g = 0.5f;
  goal_marker.color.b = 0.0f;
  goal_marker.color.a = 1.0;

  goal_marker.lifetime = ros::Duration();
  marr.markers.push_back(goal_marker);

  // Extrapolated Control Path Marker
  visualization_msgs::Marker path_marker{};
  path_marker.header.frame_id = "odom";
  path_marker.header.stamp = ros::Time::now();

  path_marker.ns = "nhttc";
  path_marker.id = 1;

  path_marker.type = visualization_msgs::Marker::LINE_STRIP;
  path_marker.action = visualization_msgs::Marker::ADD;

  Eigen::Vector3f agent_state = agents[own_index].GetProblem()->params.x_0;
  Eigen::Vector2f agent_control = agents[own_index].GetProblem()->params.u_curr;

  float ts = 0.01;
  int lookahead = 100;

  for(int i = 0; i < lookahead; i++)
  {
    float future = ts * i;

    agent_state[0] += agent_control[0] * std::cos(agent_state[2]) * ts;
    agent_state[1] += agent_control[0] * std::sin(agent_state[2]) * ts;
    agent_state[2] += agent_control[1] * ts;

    geometry_msgs::Point pt{};
    pt.x = agent_state[0];
    pt.y = agent_state[1];

    path_marker.points.push_back(pt);
  }

  path_marker.pose.orientation.w = 1.0;

  path_marker.scale.x = 0.3;
  
  path_marker.color.r = 0.5f;
  path_marker.color.g = 0.5f;
  path_marker.color.b = 0.0f;
  path_marker.color.a = 1.0;

  path_marker.lifetime = ros::Duration();

  marr.markers.push_back(path_marker);

  pub_viz.publish(marr);
}

/**
  * setting up agents
  *
  * This function initializes the nhttc agents with a preset starting pose and goal
  *
  * @param takes the index value corresponding to that agent
  */
void NHTTCNode::agent_setup(int i, AType agent_type, bool reactive)
{
  Eigen::Vector2f goal(0.0, 0.0);
  Eigen::VectorXf pos = Eigen::VectorXf::Zero(3);
  agents.emplace_back(agent_type, true, reactive, pos, goal, global_params);
}

/**
  * Quaternion to euler angle converter
  *
  * Converts quaternion representation to euler angle representation. rpy = [roll, pitch, yaw]. rpy is in radians, following the ENU reference frame
  * @params euler angle array (float), pose message (geometry_msgs::PoseStamped)
  */
void NHTTCNode::rpy_from_quat(float rpy[3],const nav_msgs::Odometry::ConstPtr& msg) 
{ 
  float q[4];

  q[0] = msg->pose.pose.orientation.x;
  q[1] = msg->pose.pose.orientation.y;
  q[2] = msg->pose.pose.orientation.z;
  q[3] = msg->pose.pose.orientation.w;

  rpy[2] = atan2f(2.0f * (q[0] * q[1] + q[2] * q[3]),1 - 2 * (q[1] * q[1] + q[2] * q[2]));
  rpy[0] = asinf(2.0f * (q[0] * q[2] - q[3] * q[1]));
  rpy[1] = atan2f(2.0f * (q[0] * q[3] + q[1] * q[2]), 1.0f - 2.0f * (q[2] * q[2] + q[3] * q[3]));
}

/**
  * Pose callback
  *
  * This function listens to the pose-publishers of all the agents
  *
  * @param takes the pose msg (geometry_msgs::PoseStamped) and the index value corresponding to the agent.
  */
void NHTTCNode::PoseCallback(const nav_msgs::Odometry::ConstPtr& msg)
{
  float rpy[3];

  Eigen::VectorXf x_o = Eigen::VectorXf::Zero(3);
  rpy_from_quat(rpy, msg);
  
  x_o[0] = msg->pose.pose.position.x;
  x_o[1] = msg->pose.pose.position.y;
  x_o[2] = rpy[2];

  agents[own_index].SetEgo(x_o);
  agents[own_index].SetLastUpdated(msg->header.stamp.toSec());
}

void NHTTCNode::NeighborCallback(const nav_msgs::Odometry::ConstPtr& msg, int neighbor_idx)
{
  Eigen::VectorXf x = Eigen::VectorXf::Zero(2);
  Eigen::VectorXf u = Eigen::VectorXf::Zero(2);

  x[0] = msg->pose.pose.position.x;
  x[1] = msg->pose.pose.position.y;

  u[0] = msg->twist.twist.linear.x;
  u[1] = msg->twist.twist.linear.y;

  agents[neighbor_idx].SetEgo(x);
  agents[neighbor_idx].SetControls(u);
  agents[neighbor_idx].SetLastUpdated(msg->header.stamp.toSec());
}

/**
  * Single-goal callback
  *
  * Used for a single-goal point mission.
  *
  * @param goal point message (geometry_msgs::PoseStamped)
  */
void NHTTCNode::GoalCallback(const geometry_msgs::PoseStamped::ConstPtr& msg)
{
  goal[0] = msg->pose.position.x;
  goal[1] = msg->pose.position.y;

  agents[own_index].UpdateGoal(goal);

  goal_received = true;
}

/**
  * Send throttle/steering commands
  *
  * sends commands to the vesc/mux/input/navigation topic
  *
  * @param speed (float, m/s) steering angle (float, radius)
  */
void NHTTCNode::send_commands(float speed, float steer)
{
  geometry_msgs::Twist output_msg;

  output_msg.angular.z = steer;
  output_msg.linear.x = speed;

  pub_cmd.publish(output_msg);
}

/**
  * Check for new agents
  *
  * updates the agent array by finding published topics with topic type PoseStamped and unique car_name.
  * @params: ros::NodeHandle
  * @returns: None
  */
void NHTTCNode::check_new_agents(ros::NodeHandle &nh)
{
  ros::master::getTopics(master_topics);

  for (ros::master::V_TopicInfo::iterator it = master_topics.begin() ; it != master_topics.end(); it++) 
  {
    const ros::master::TopicInfo& info = *it;

    if (info.name == odom_topic && own_index == -1)
    {
      count++;
      ROS_INFO_STREAM("Found ego (idx: " << count << ") on topic " << info.name);
      own_index = count;
      agent_setup(count, AType::DD, true);
    }

    if (info.name.find(neighbor_topic_root) != std::string::npos && std::find(topics_neighbor.begin(), topics_neighbor.end(), info.name) == topics_neighbor.end())
    {
      count++;
      ROS_INFO_STREAM("Found neighbor (idx: " << count << ") on topic " << info.name);
      agent_setup(count, AType::V, false);
      subs_neighbor.push_back(
        nh.subscribe<nav_msgs::Odometry>(info.name, 10, boost::bind(&NHTTCNode::NeighborCallback, this, _1, count))
      );
      topics_neighbor.push_back(info.name);
    }

  }
}

/**
  * nhttc ros constructor
  *
  * Sets up the publishers, subscribers and the whole shebang
  */
NHTTCNode::NHTTCNode(ros::NodeHandle &nh)
{
  own_index = -1;
  goal_received = false; // start with the assumption that the car has no goal

  if(not nh.getParam("/solver_time", solver_time))
  {
    solver_time = 10; // 10 ms solver time for each agent.
  }
  if(not nh.getParam("sim", simulation))
  {
    simulation = true; // run in simulation by default.
  }
  if(not nh.getParam("max_agents", num_agents_max))
  {
    num_agents_max = 8; // default maximum number of agents
  }
  if(not nh.getParam("/carrot_goal_ratio", carrot_goal_ratio))
  {
    carrot_goal_ratio = 1.0f; //default distance to the ever-changing goal
  }
  if(not nh.getParam("/max_ttc", max_ttc))
  {
    max_ttc = 6.0f; // default ttc 
  }
  if(not nh.getParam("/obey_time", obey_time))
  {
    obey_time = false;// false by default
  }
  if(not nh.getParam("/allow_reverse", allow_reverse))
  {
    allow_reverse = true;// true by default (default behavior is to not have any constraints on the nav engine)
  }
  if(not nh.getParam("/adaptive_lookahead",adaptive_lookahead))
  {
    adaptive_lookahead = false;
  }
  if(not nh.getParam("/safety_radius", safety_radius))
  {
    safety_radius = 0.1f;
  }
  if(not nh.getParam("/odom_topic", odom_topic))
  {
    odom_topic = "/odom";
  }
  if(not nh.getParam("/cmd_vel_topic", cmd_vel_topic))
  {
    cmd_vel_topic = "/cmd_vel";
  }
  if (not nh.getParam("/cutoff_dist", cutoff_dist))
  {
    cutoff_dist = 0.1f;
  }
  if (not nh.getParam("/neighbor_topic_root", neighbor_topic_root))
  {
    neighbor_topic_root = "/odometry/tracker_";
  }

  ConstructGlobalParams(&global_params);
  count = -1; 

  // Check 5 times over 5 seconds for the ego agent topics
  ros::Rate r(1);
  for(int i=0;i<5;i++)
  {
    check_new_agents(nh);
    r.sleep();
  } 

  // set up all the publishers/subscribers
  sub_wp = nh.subscribe("/move_base_simple/goal", 10, &NHTTCNode::GoalCallback, this);
  sub_pose = nh.subscribe(odom_topic, 10, &NHTTCNode::PoseCallback, this);

  pub_viz = nh.advertise<visualization_msgs::MarkerArray>("/nhttc/debug_markers", 1);
  pub_cmd = nh.advertise<geometry_msgs::Twist>(cmd_vel_topic, 10);

  ROS_INFO("node started"); // tell the world that the node has initialized.
}

/**
  * Parameter setting for the ego-agent.
  *
  * Sets up the tuning parameters for the ego-agent. These parameters are taken from the launch file. The tuning parameters include 
  * 1) The carrot-goal ratio: (lookahead distance)/(turning radius of the car)
  * 2) max_ttc: maximum time-to-collision
  */
void NHTTCNode::setup()
{
  steer_limit = 0.5*M_PI; 

  wheelbase = agents[own_index].GetProblem()->params.wheelbase;
  agents[own_index].GetProblem()->params.safety_radius = safety_radius;
  agents[own_index].GetProblem()->params.steer_limit = steer_limit;
  agents[own_index].GetProblem()->params.vel_limit = speed_lim;
  agents[own_index].GetProblem()->params.u_lb = allow_reverse ? Eigen::Vector2f(-speed_lim, -steer_limit) : Eigen::Vector2f(0, -steer_limit);
  agents[own_index].GetProblem()->params.u_ub = Eigen::Vector2f(speed_lim, steer_limit);
  agents[own_index].GetProblem()->params.max_ttc = max_ttc;

  ROS_INFO("carrot_goal_ratio: %f",carrot_goal_ratio);
  ROS_INFO("max_ttc: %f", max_ttc);
  ROS_INFO("solver_time: %d", solver_time);
  ROS_INFO("obey_time:%d", int(obey_time));
  ROS_INFO("allow_reverse: %d", int(allow_reverse));
  ROS_INFO("safety_radius: %f", safety_radius);
  ROS_INFO("adaptive_lookahead, %d", int(adaptive_lookahead));
}

/**
  * local planner
  *
  * Calls the nhttc solver and sends the output of the solver to the ego-agent. Note that the solver solves only for one agent but it does need
  * to know what the other agents are doing. 
  */
void NHTTCNode::plan()
{
  // create obstacle list.
  if(agents.size() == 0) return;

  obstacles = BuildObstacleList(agents, own_index, 0.5);
  agents[own_index].SetPlanTime(solver_time); //20 ms planning window TODO: see if this only needs to be done once
  agents[own_index].SetObstacles(obstacles, size_t(own_index)); // set the obstacles 

  // Controls are 0,0 by default.
  Eigen::VectorXf controls = Eigen::VectorXf::Zero(2);

  // Check if we are within set distance to goal. If so stop planning until next goal is found
  Eigen::Vector2f agent_state = agents[own_index].GetProblem()->params.x_0.head(2);
  float goal_dist = (agents[own_index].GetGoal() - agent_state).norm();

  ROS_INFO_STREAM_THROTTLE(1, "Goal Distance: " << goal_dist << " / " << cutoff_dist);

  if (goal_dist < cutoff_dist)
  {
    goal_received = false;
  }

  if(goal_received)
  { 
    controls = agents[own_index].UpdateControls();
  }

  float speed = controls[0]; //speed in m/s
  float steering_angle = controls[1]; //steering angle in radians. +ve is left. -ve is right 
  ROS_DEBUG_STREAM("New control: " << speed << ", " << steering_angle);
  send_commands(speed,steering_angle); //just sending out anything for now;
  return;
}

/**
 * main function
 *
 * Sets up the nhttc object and initializes everything.
 */
int main(int argc, char** argv)
{
  ros::init(argc, argv, "nhttc_local_planner");
  ros::NodeHandle nh("~");
  NHTTCNode local_planner(nh);
  ros::Rate r(40);
  bool init = false; //flag for node initialization (indicates if the car's car_pose topic has been found)

  while(ros::ok)
  {
    ros::spinOnce(); 

    if(local_planner.own_index != -1 && !init) //if car had not been initialized before and has now found the car_pose topic
    {
      init = true; // set init to true
      local_planner.setup(); // set up the agent
    }

    if(init) //if init
    {
      local_planner.viz_publish();
      local_planner.plan();
    }

    r.sleep();
  }
  return 0;
}
