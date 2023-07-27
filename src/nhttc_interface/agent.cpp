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

Agent::Agent(SGDOptParams opt_params_in) { opt_params = opt_params_in; }

Agent::Agent(std::vector<std::string> parts, SGDOptParams opt_params_in) {
  TTCParams params;
  params.find_all_ttcs = false;
  type_name = parts[0];
  if (parts[0] == "v") {
    a_type = AType::V;
    u_dim = x_dim = 2;
    SetBoundsV(params);
    prob = new VTTCSGDProblem(params);
  } else if (parts[0] == "a") {
    a_type = AType::A;
    u_dim = 2;
    x_dim = 4;
    SetBoundsA(params);
    prob = new ATTCSGDProblem(params);
  } else if (parts[0] == "dd") {
    a_type = AType::DD;
    u_dim = 2;
    x_dim = 3;
    SetBoundsDD(params);
    prob = new DDTTCSGDProblem(params);
  } else if (parts[0] == "add") {
    a_type = AType::ADD;
    u_dim = 2;
    x_dim = 5;
    SetBoundsADD(params);
    prob = new ADDTTCSGDProblem(params);
  } else if (parts[0] == "car") {
    a_type = AType::CAR;
    u_dim = 2;
    x_dim = 3;
    SetBoundsCAR(params);
    prob = new CARTTCSGDProblem(params);
  } else if (parts[0] == "acar") {
    a_type = AType::ACAR;
    u_dim = 2;
    x_dim = 5;
    SetBoundsACAR(params);
    prob = new ACARTTCSGDProblem(params);
  } else {
    spdlog::error("Unsupported Dynamics Model: {}", parts[0]);
    exit(-1);
  }
  controlled = (parts[1] == "y");
  reactive = (parts[2] == "y");
  int offset = 3;
  Eigen::VectorXf u_0, x_0, g;
  offset = GetVector(parts, offset, x_dim, x_0);
  offset = GetVector(parts, offset, u_dim, u_0);
  offset = GetVector(parts, offset, 2, g);

  prob->params.u_curr = u_0;
  prob->params.x_0 = x_0;
  goal = g;

  opt_params = opt_params_in;
  opt_params.x_lb = prob->params.u_lb;
  opt_params.x_ub = prob->params.u_ub;
  opt_params.x_0 = u_0;
}

void Agent::SetPlanTime(float agent_plan_time_ms) {
  opt_params.max_time = agent_plan_time_ms;
}

void Agent::SetObstacles(std::vector<TTCObstacle *> obsts, size_t own_index) {
  prob->params.obsts.clear();
  // Don't plan for non reactive agents
  if (!reactive) {
    return;
  }
  for (size_t b_idx = 0; b_idx < obsts.size(); ++b_idx) {
    if (b_idx == own_index) {
      continue;
    }
    float dist =
        (prob->params.x_0.head<2>() - obsts[b_idx]->p.head<2>()).norm();
    // Ignore any obstacles that cannot interact with us within our ttc horizon
    if (dist < 2.0 * prob->params.vel_limit * prob->params.max_ttc) {
      prob->params.obsts.push_back(obsts[b_idx]);
    }
  }
}

void Agent::UpdateGoal(Eigen::Vector2f new_goal) {
  // if (new_goal != NULL) {
  goal = new_goal;
  // }
}

void Agent::SetEgo(Eigen::VectorXf new_x) { prob->params.x_0 = new_x; }

void Agent::SetControls(Eigen::VectorXf new_controls) // CHANGED
{
  prob->params.u_curr = new_controls;
}

Eigen::VectorXf Agent::UpdateControls() {

  // Push latest Agent goal to SGD params
  prob->params.goals.clear();
  for (size_t i = 0; i < prob->params.ts_goal_check.size(); ++i) {
    prob->params.goals.push_back(goal);
  }

  // Prepare params using global params
  PrepareSGDParams();

  // Solve SGD
  float sgd_opt_cost;
  Eigen::VectorXf u_new = SGD::Solve(prob, opt_params, &sgd_opt_cost);

  spdlog::debug("New Control: {}, {}", u_new[0], u_new[1]);

  prob->params.u_curr = 0.5f * (u_new + prob->params.u_curr); // Reciprocity
  return prob->params.u_curr;
}

float Agent::GetBestGoalCost(float dt, const Eigen::Vector2f &g_pos) {
  Eigen::Vector2f pos = prob->params.x_0.head<2>();
  float curr_dist = (pos - g_pos).norm();
  if (curr_dist < dt * prob->params.vel_limit) {
    return -curr_dist;
  }
  pos += prob->params.vel_limit * dt * (g_pos - pos).normalized();
  float new_dist = (pos - g_pos).norm();
  return prob->params.k_goal * (new_dist - curr_dist);
}

// Lower bound estimate of optimal cost
float Agent::GetBestCost() {
  float tot_cost = 0.0f;
  for (size_t i = 0; i < prob->params.ts_goal_check.size(); ++i) {
    tot_cost +=
        GetBestGoalCost(prob->params.ts_goal_check[i], prob->params.goals[i]);
  }
  return tot_cost;
}

bool Agent::AtGoal() {
  return (prob->GetCollisionCenter(prob->params.x_0) - goal).norm() < 0.05f;
}

void Agent::SetStop() {
  prob->params.u_curr = Eigen::VectorXf::Zero(u_dim);
  if (a_type == AType::V) {
    // Nothing to do
  } else if (a_type == AType::A) {
    prob->params.x_0.tail<2>() = Eigen::VectorXf::Zero(2);
  } else if (a_type == AType::DD) {
    // Nothing to do
  } else if (a_type == AType::ADD) {
    prob->params.x_0.tail<2>() = Eigen::VectorXf::Zero(2);
  } else if (a_type == AType::CAR) {
    // Nothing to do
  } else if (a_type == AType::ACAR) {
    prob->params.x_0[3] = 0.0f;
  } 
}

void Agent::PrepareSGDParams() {
  float best_possible_cost = GetBestCost();
  opt_params.polyak_best = best_possible_cost;
  opt_params.x_0 = prob->params.u_curr;
  opt_params.x_lb = prob->params.u_lb;
  opt_params.x_ub = prob->params.u_ub;
}