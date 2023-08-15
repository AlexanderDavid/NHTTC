/*
Copyright 2020 University of Minnesota and Clemson University

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#pragma once

#include <sgd/ttc_sgd_problem.h>
#include <Eigen/Core>

/**
 * Different types for the different kinematics of the neighbors
*/
enum class AType { 
  V = 0,
  A = 1,
  DD = 2,
  ADD = 3,
  CAR = 4,
  ACAR = 5,
};

/**
 * High level class for an NH-TTC agent
*/
class Agent {
private:
  TTCSGDProblem* _prob;
  SGDOptParams _opt_params;

  int _u_dim, _x_dim;
  bool _reactive, _controlled, _done = false;
  double _last_update;

  AType _a_type;
  std::string _type_name;
  Eigen::Vector2f _goal;

public:

  
  /**
   * @brief Create a new agent for use in planning 
   * @param kinematics Kinematics of the agent
   * @param is_controlled If the agent is being controlled by the class
   * @param is_reactive If the agent is going to react sensibly
   * @param x_0 Initial state
   * @param g Initial goal
   * @param opt_params_in SGD params
   */
  Agent(AType kinematics, bool is_controlled, bool is_reactive, Eigen::VectorXf x_0, Eigen::VectorXf g, SGDOptParams opt_params_in);

  /**
   * @note This function absolutely reeks. Exposes the implementation but required
   * because i'm too lazy to write getters and setters for all the information
   * in the SGD problem.
   */
  inline TTCSGDProblem* GetProblem() { return _prob; }

  inline bool isReactive() { return _reactive; }
  inline Eigen::Vector2f GetGoal() { return _goal; }
  inline AType GetAType() { return _a_type; }
  inline float getGoalDistance() { return (_prob->GetCollisionCenter(_prob->params.x_0) - _goal).norm(); }
  
  inline void SetLastUpdated(double time) { _last_update = time; };
  inline double GetLastUpdated() { return _last_update; }

  void SetPlanTime(float agent_plan_time_ms);

  /**
   * @brief Set the obstacles to plan around
   * @param obsts Obstacles including ego
   * @param own_index Ego's index in the obstacle list
   */
  void SetObstacles(std::vector<TTCObstacle*> obsts, size_t own_index);

  void SetGoal(Eigen::Vector2f new_goal);
  void SetState(Eigen::VectorXf new_x);
  void SetControls(Eigen::VectorXf new_controls);

  /**
   * @brief Calcualte and update controls for ego agent
   * 
   * @returns New controls
  */
  Eigen::VectorXf CalculateControls();

  void PrepareSGDParams();

private:
  float GetBestGoalCost(float dt, const Eigen::Vector2f& pos);
  float GetBestCost();
};