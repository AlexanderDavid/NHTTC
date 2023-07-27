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

#include <spdlog/spdlog.h>

#include "agent.h"
#include "util.h"

#include <sgd/ttc_sgd_problem_models.h>

Agent::Agent(AType kinematics, bool is_controlled, bool is_reactive, Eigen::VectorXf x_0, Eigen::VectorXf g, SGDOptParams opt_params_in)
{
    TTCParams params;
    params.find_all_ttcs = false;

    _a_type = kinematics;
    _controlled = is_controlled;
    _reactive = is_reactive;
    _goal = g;

    switch (_a_type)
    {
    case AType::V:
        _u_dim = _x_dim = 2;
        SetBoundsV(params);
        _prob = new VTTCSGDProblem(params);

    case AType::DD:
        _x_dim = 3;
        _u_dim = 2;
        SetBoundsDD(params);
        _prob = new VTTCSGDProblem(params);

    default:
        spdlog::error("Unsupported Dynamics Model: {}", kinematics);
        exit(-1);
        break;
    }

    // _prob->params.u_curr = u_0;
    _prob->params.x_0 = x_0;

    _opt_params = opt_params_in;
    _opt_params.x_lb = _prob->params.u_lb;
    _opt_params.x_ub = _prob->params.u_ub;
}


void Agent::SetPlanTime(float agent_plan_time_ms) {
  _opt_params.max_time = agent_plan_time_ms;
}

void Agent::SetObstacles(std::vector<TTCObstacle *> obsts, size_t own_index) {
  _prob->params.obsts.clear();
  // Don't plan for non reactive agents
  if (!_reactive) {
    return;
  }
  for (size_t b_idx = 0; b_idx < obsts.size(); ++b_idx) {
    if (b_idx == own_index) {
      continue;
    }
    float dist =
        (_prob->params.x_0.head<2>() - obsts[b_idx]->p.head<2>()).norm();
    // Ignore any obstacles that cannot interact with us within our ttc horizon
    if (dist < 2.0 * _prob->params.vel_limit * _prob->params.max_ttc) {
      _prob->params.obsts.push_back(obsts[b_idx]);
    }
  }
}

void Agent::UpdateGoal(Eigen::Vector2f new_goal) {
  _goal = new_goal;
}

void Agent::SetEgo(Eigen::VectorXf new_x) { 
    _prob->params.x_0 = new_x;
}

void Agent::SetControls(Eigen::VectorXf new_controls) {
  _prob->params.u_curr = new_controls;
}

Eigen::VectorXf Agent::UpdateControls() {
  // Push latest Agent goal to SGD params
  _prob->params.goals.clear();
  for (size_t i = 0; i < _prob->params.ts_goal_check.size(); ++i) {
    _prob->params.goals.push_back(_goal);
  }

  // Prepare params using global params
  PrepareSGDParams();

  // Solve SGD
  float sgd_opt_cost;
  Eigen::VectorXf u_new = SGD::Solve(_prob, _opt_params, &sgd_opt_cost);

  spdlog::debug("New Control: {}, {}", u_new[0], u_new[1]);

  _prob->params.u_curr = 0.5f * (u_new + _prob->params.u_curr); // Reciprocity
  return _prob->params.u_curr;
}

float Agent::GetBestGoalCost(float dt, const Eigen::Vector2f &g_pos) {
  Eigen::Vector2f pos = _prob->params.x_0.head<2>();
  float curr_dist = (pos - g_pos).norm();
  if (curr_dist < dt * _prob->params.vel_limit) {
    return -curr_dist;
  }
  pos += _prob->params.vel_limit * dt * (g_pos - pos).normalized();
  float new_dist = (pos - g_pos).norm();
  return _prob->params.k_goal * (new_dist - curr_dist);
}

// Lower bound estimate of optimal cost
float Agent::GetBestCost() {
  float tot_cost = 0.0f;
  for (size_t i = 0; i < _prob->params.ts_goal_check.size(); ++i) {
    tot_cost +=
        GetBestGoalCost(_prob->params.ts_goal_check[i], _prob->params.goals[i]);
  }
  return tot_cost;
}

bool Agent::AtGoal() {
  return (_prob->GetCollisionCenter(_prob->params.x_0) - _goal).norm() < 0.05f;
}

void Agent::SetStop() {
  _prob->params.u_curr = Eigen::VectorXf::Zero(_u_dim);
  
  switch(_a_type) {
    case AType::A:
      _prob->params.x_0.tail<2>() = Eigen::VectorXf::Zero(2);
      break;
    case AType::ADD:
      _prob->params.x_0.tail<2>() = Eigen::VectorXf::Zero(2);
      break;
    case AType::ACAR:
      _prob->params.x_0[3] = 0.0f;
      break;
  }
}

void Agent::PrepareSGDParams() {
  float best_possible_cost = GetBestCost();
  _opt_params.polyak_best = best_possible_cost;
  _opt_params.x_0 = _prob->params.u_curr;
  _opt_params.x_lb = _prob->params.u_lb;
  _opt_params.x_ub = _prob->params.u_ub;
}