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
  Agent(AType kinematics, bool is_controlled, bool is_reactive, Eigen::VectorXf x_0, Eigen::VectorXf g, SGDOptParams opt_params_in);

  // This function absolutely reeks. Exposes the implementation but required
  // because i'm too lazy to write getters and setters for all the information
  // in the SGD problem.
  inline TTCSGDProblem* GetProblem() { return _prob; }
  inline bool isReactive() { return _reactive; }
  inline Eigen::Vector2f GetGoal() { return _goal; }
  
  inline void SetLastUpdated(double time) { _last_update = time; };
  inline double GetLastUpdated() { return _last_update; }

  inline AType GetAType() { return _a_type; }

  void SetPlanTime(float agent_plan_time_ms);

  // Pass in own index if agent is itself one of the obstacles passed in, otherwise pass in -1
  void SetObstacles(std::vector<TTCObstacle*> obsts, size_t own_index);

  // Update Goal by passing in a new vector. If unchanged, pass in NULL. Calling the function will update goal and pass it into SGD
  void UpdateGoal(Eigen::Vector2f new_goal);

  // Sets ego of Agent
  // Note: For the MuSHR car, x is [x, y, heading_angle] and u is [velocity, steering_angle]
  void SetEgo(Eigen::VectorXf new_x);
  void SetControls(Eigen::VectorXf new_controls); //CHANGED

  // Controls can be extracted with agent->prob.params.ucurr at any time. This function runs the update and also returns the controls.
  // Note: For the MuSHR car, x is [x, y, heading_angle] and u is [velocity, steering_angle]
  Eigen::VectorXf UpdateControls();

  // Original From nhttc_utils
  float GetBestGoalCost(float dt, const Eigen::Vector2f& pos);
  float GetBestCost();
  bool AtGoal();
  void SetStop();
  void PrepareSGDParams();
};