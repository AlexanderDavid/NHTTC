#include <string>
#include <sstream>

#include "ros/ros.h"
#include "ros/master.h"
#include <boost/algorithm/string.hpp>
#include <spdlog/spdlog.h>

#include "nav_msgs/Odometry.h"
#include "geometry_msgs/Twist.h"
#include "geometry_msgs/Point.h"
#include "geometry_msgs/PoseStamped.h"
#include "visualization_msgs/MarkerArray.h"
#include "visualization_msgs/Marker.h"
#include "nhttc_ros/AgentState.h"

#include "nhttc_node.h"

void NHTTCNode::publishDebugViz()
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

    for (int i = 0; i < lookahead; i++)
    {
        float future = ts * i;

        agent_state[0] += agent_control[0] * std::cos(agent_state[2]) * future;
        agent_state[1] += agent_control[0] * std::sin(agent_state[2]) * future;
        agent_state[2] += agent_control[1] * future;

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

    _pub_viz.publish(marr);
}

void NHTTCNode::publishNHTTCPose()
{
    // Publish the current state of the NH-TTC agent, this not to be used
    // for any actual localization or included further down the pipeline.
    // This is used internally by NH-TTC to be able to more accurately
    // optimize based on the specific kinematics of the neighbors.
    nhttc_ros::AgentState msg{};

    msg.header.stamp = ros::Time::now();

    msg.kinematics = (uint8_t) agents[own_index].GetAType();

    Eigen::VectorXf x_curr = agents[own_index].GetProblem()->params.x_0;
    std::vector<float> xvec(x_curr.data(), x_curr.data() + x_curr.size());

    Eigen::VectorXf u_curr = agents[own_index].GetProblem()->params.u_curr;
    std::vector<float> uvec(u_curr.data(), u_curr.data() + u_curr.size());

    msg.state = xvec;
    msg.control = uvec;

    // Could also fill in the nav_msgs/Odometry-like portion of the message
    // here but I'm not gonna use that at all, just future proofing when 
    // defining message type

    pub_nhttc_pose.publish(msg);
}

/**
 * setting up agents
 *
 * This function initializes the nhttc agents with a preset starting pose and goal
 *
 * @param takes the index value corresponding to that agent
 */
void NHTTCNode::agentSetup(int i, AType agent_type, bool reactive)
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
void NHTTCNode::RPYFromQuat(float rpy[3], const nav_msgs::Odometry::ConstPtr &msg)
{
    float q[4];

    q[0] = msg->pose.pose.orientation.x;
    q[1] = msg->pose.pose.orientation.y;
    q[2] = msg->pose.pose.orientation.z;
    q[3] = msg->pose.pose.orientation.w;

    rpy[2] = atan2f(2.0f * (q[0] * q[1] + q[2] * q[3]), 1 - 2 * (q[1] * q[1] + q[2] * q[2]));
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
void NHTTCNode::poseCallback(const nav_msgs::Odometry::ConstPtr &msg)
{
    float rpy[3];

    Eigen::VectorXf x_o = Eigen::VectorXf::Zero(3);
    RPYFromQuat(rpy, msg);

    x_o[0] = msg->pose.pose.position.x;
    x_o[1] = msg->pose.pose.position.y;
    x_o[2] = rpy[2];

    agents[own_index].SetState(x_o);
    agents[own_index].SetLastUpdated(msg->header.stamp.toSec());
}

void NHTTCNode::neighborCallback(const nav_msgs::Odometry::ConstPtr &msg, int neighbor_idx)
{
    Eigen::VectorXf x = Eigen::VectorXf::Zero(2);
    Eigen::VectorXf u = Eigen::VectorXf::Zero(2);

    x[0] = msg->pose.pose.position.x;
    x[1] = msg->pose.pose.position.y;

    u[0] = msg->twist.twist.linear.x;
    u[1] = msg->twist.twist.linear.y;

    agents[neighbor_idx].SetState(x);
    agents[neighbor_idx].SetControls(u);
    agents[neighbor_idx].SetLastUpdated(msg->header.stamp.toSec());
}

void NHTTCNode::NHTTCNeighborCallback(const nhttc_ros::AgentState::ConstPtr &msg, int neighbor_idx)
{
    // This is going to be dangerious since we are passing arrays
    // of unknown size but it SHOULD work. If the code errors here
    // then one of the agents is publishing a control or state array
    // of different size then they're using
    std::vector<float> local_x = msg->state;
    std::vector<float> local_u = msg->control;

    Eigen::VectorXf x = Eigen::Map<Eigen::VectorXf>(local_x.data(), local_x.size());
    Eigen::VectorXf u = Eigen::Map<Eigen::VectorXf>(local_u.data(), local_u.size());

    agents[neighbor_idx].SetState(x);
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
void NHTTCNode::goalCallback(const geometry_msgs::PoseStamped::ConstPtr &msg)
{
    goal[0] = msg->pose.position.x;
    goal[1] = msg->pose.position.y;

    agents[own_index].SetGoal(goal);

    goal_received = true;
}

/**
 * Send throttle/steering commands
 *
 * sends commands to the vesc/mux/input/navigation topic
 *
 * @param speed (float, m/s) steering angle (float, radius)
 */
void NHTTCNode::sendCommands(float speed, float steer)
{
    geometry_msgs::Twist output_msg;

    output_msg.angular.z = steer;
    output_msg.linear.x = speed;

    _pub_cmd.publish(output_msg);
}

void NHTTCNode::checkNewAgents(ros::NodeHandle &nh)
{
    ros::master::V_TopicInfo master_topics;
    ros::master::getTopics(master_topics);

    for (ros::master::V_TopicInfo::iterator it = master_topics.begin(); it != master_topics.end(); it++)
    {
        const ros::master::TopicInfo &info = *it;
        int count = agents.size();

        if (info.name == odom_topic && own_index == -1)
        {
            spdlog::info("Found ego (idx: {}) on {}", count, info.name);
            own_index = count;
            agentSetup(count, AType::DD, true);
            setup();
        }

        else if (info.name.find(neighbor_topic_root) != std::string::npos &&
                 std::find(_topics_neighbor.begin(), _topics_neighbor.end(), info.name) == _topics_neighbor.end() &&
                 info.datatype == "nav_msgs/Odometry")
        {
            spdlog::info("Found nav_msgs neighbor (idx: {}) on {}", count, info.name);
            agentSetup(count, AType::V, false);
            _subs_neighbor.push_back(
                nh.subscribe<nav_msgs::Odometry>(info.name, 10, boost::bind(&NHTTCNode::neighborCallback, this, _1, count)));
            _topics_neighbor.push_back(info.name);
        }

        // The way this logic is implemented the agents will not operate in mixed environments.
        // More specifically, the agent can only listen on a single topic root, so there should
        // be no duplicates. This needs to be fixed if we want to have some agents using
        // NH-TTC and some not.
        else if (info.name.find(neighbor_topic_root) != std::string::npos &&
                 std::find(_topics_neighbor.begin(), _topics_neighbor.end(), info.name) == _topics_neighbor.end() &&
                 info.datatype == "nhttc_ros/AgentState")
        {
            spdlog::info("Found nhttc_ros neighbor (idx: {}) on {}", count, info.name);

            // We now need to wait for the next message on this topic to come in so that we can
            // get the dynamics. We will wait for 5 seconds for this message, terminating and
            // skipping this topic if no message is heard.
            nhttc_ros::AgentStateConstPtr msg = ros::topic::waitForMessage<nhttc_ros::AgentState>(
                info.name, nh, ros::Duration(5));

            if (msg == nullptr)
            {
                spdlog::warn("Could not hear odometry on {}, skipping.", info.name);
                continue;
            }

            agentSetup(count, (AType) msg->kinematics, false);
            _subs_neighbor.push_back(
                nh.subscribe<nhttc_ros::AgentState>(
                    info.name,
                    10,
                    boost::bind(&NHTTCNode::NHTTCNeighborCallback, this, _1, count)));
            _topics_neighbor.push_back(info.name);
        }
    }
}

NHTTCNode::NHTTCNode(ros::NodeHandle &nh)
{
    own_index = -1;
    goal_received = false; 

    // Optimization parameters
    solver_time         = nh.param("solver_time", 10);
    safety_radius       = nh.param("safety_radius", 0.1f);
    carrot_goal_ratio   = nh.param("carrot_goal_ratio", 1.0f);
    max_ttc             = nh.param("max_ttc", 6.0f);
    obey_time           = nh.param("obey_time", false); // false by default
    adaptive_lookahead  = nh.param("adaptive_lookahead", false);

    // Neighbor parameters
    num_agents_max      = nh.param("max_agents", 8);
    pose_timeout        = nh.param("pose_timeout", 0.5);
    odom_topic          = nh.param("odom_topic", std::string("odom"));
    neighbor_topic_root = nh.param("neighbor_topic_root", std::string("/odometry/tracker_"));

    // Control parameters
    steer_limit         = nh.param("steer_limit", 0.5 * M_PI);
    allow_reverse       = nh.param("allow_reverse", true);
    speed_lim           = nh.param("speed_lim", 0.46f);
    cutoff_dist         = nh.param("cutoff_dist", 0.1f);


    ConstructGlobalParams(&global_params);

    // Check 5 times over 5 seconds for the ego agent topics. Maybe it would
    // be better to use a timer every second for the lifecycle of the node?
    ros::Rate r(1);
    for (int i = 0; i < 5; i++)
    {
        checkNewAgents(nh);
        r.sleep();
    }

    std::string goal_topic = nh.param("goal_topic", std::string("goal"));

    _sub_wp   = nh.subscribe(goal_topic, 10, &NHTTCNode::goalCallback, this);
    _sub_pose = nh.subscribe(odom_topic, 10, &NHTTCNode::poseCallback, this);

    std::string cmd_vel_topic = nh.param("cmd_vel_topic", std::string("cmd_vel"));

    _pub_viz = nh.advertise<visualization_msgs::MarkerArray>("debug_markers", 1);
    _pub_cmd = nh.advertise<geometry_msgs::Twist>(cmd_vel_topic, 10);

    spdlog::info("node started");
}

void NHTTCNode::setup()
{
    agents[own_index].GetProblem()->params.safety_radius = safety_radius;
    agents[own_index].GetProblem()->params.steer_limit = steer_limit;
    agents[own_index].GetProblem()->params.vel_limit = speed_lim;
    agents[own_index].GetProblem()->params.u_lb = Eigen::Vector2f(allow_reverse ? -speed_lim : 0, -steer_limit);
    agents[own_index].GetProblem()->params.u_ub = Eigen::Vector2f(speed_lim, steer_limit);
    agents[own_index].GetProblem()->params.max_ttc = max_ttc;

    spdlog::info("carrot_goal_ratio:  {}", carrot_goal_ratio);
    spdlog::info("max_ttc:            {}", max_ttc);
    spdlog::info("solver_time:        {}", solver_time);
    spdlog::info("obey_time:          {}", int(obey_time));
    spdlog::info("allow_reverse:      {}", int(allow_reverse));
    spdlog::info("safety_radius:      {}", safety_radius);
    spdlog::info("adaptive_lookahead: {}", int(adaptive_lookahead));
}

void NHTTCNode::plan()
{
    // If there are no agents then do not plan
    if (agents.size() == 0)
        return;

    // Build the obstacle list
    obstacles = BuildObstacleList(agents, own_index, pose_timeout);

    // Set up the agent
    agents[own_index].SetPlanTime(solver_time);                   
    agents[own_index].SetObstacles(obstacles, size_t(own_index));

    // Controls are 0,0 by default.
    Eigen::VectorXf controls = Eigen::VectorXf::Zero(2);

    // Check if we are within set distance to goal. If so stop planning until next goal is found
    Eigen::Vector2f agent_state = agents[own_index].GetProblem()->params.x_0.head(2);
    float goal_dist = agents[own_index].getGoalDistance();

    if (goal_dist < cutoff_dist)
    {
        goal_received = false;
    }

    if (goal_received)
    {
        controls = agents[own_index].CalculateControls();
    }

    float speed = controls[0];         
    float steering_angle = controls[1];

    spdlog::debug("New control: {}, {}", speed, steering_angle);

    sendCommands(speed, steering_angle);

    // Publish the nhttc pose for other nhttc agents
    publishNHTTCPose();

    publishDebugViz();

    return;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "nhttc_local_planner");
    ros::NodeHandle nh("~");
    NHTTCNode local_planner(nh);
    ros::Rate r(40);

    while (!ros::isShuttingDown())
    {
        ros::spinOnce();

        if (local_planner.ready())
        {
            local_planner.plan();
        }

        r.sleep();
    }

    return 0;
}
