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

#include <sgd/ttc_sgd_problem.h>
#include <Eigen/Core>

enum class AType { V, A, DD, ADD, CAR, ACAR, MUSHR };

class Agent {
public:
  TTCSGDProblem* prob;
  SGDOptParams opt_params;
  int u_dim, x_dim;
  bool reactive, controlled, done = false;
  AType a_type;
  std::string type_name;
  Eigen::Vector2f goal;

  Agent(SGDOptParams opt_params_in);
  Agent(std::vector<std::string> parts, SGDOptParams opt_params_in);

  // Added for easier interfacing
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