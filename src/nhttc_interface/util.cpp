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

#include <fstream>

#include "util.h"

std::vector<std::string> GetParts(std::string s, char delim) {
  std::stringstream ss(s);
  std::vector<std::string> parts;
  for (std::string part; std::getline(ss, part, delim);) {
    parts.push_back(std::move(part));
  }
  return parts;
}

std::vector<std::vector<std::string>> LoadFileByToken(std::string file_name,
                                                      int n_skip, char delim) {
  std::ifstream datafile(file_name);
  std::vector<std::vector<std::string>> data_vec;
  int n_lines_read = 0;
  for (std::string line; std::getline(datafile, line);) {
    n_lines_read++;
    if (n_lines_read <= n_skip) {
      continue;
    }
    std::vector<std::string> parts = GetParts(line, delim);
    data_vec.push_back(parts);
  }
  datafile.close();
  return data_vec;
}

void SetBoundsV(TTCParams &params) {
  params.box_constraint = false;
  params.circle_u_limit = 0.3f;
  params.u_lb = Eigen::Vector2f(-0.3f, -0.3f);
  params.u_ub = Eigen::Vector2f(0.3f, 0.3f);
}

void SetBoundsA(TTCParams &params) {
  params.u_lb = Eigen::Vector2f(-1.0f, -1.0f);
  params.u_ub = Eigen::Vector2f(1.0f, 1.0f);
}

void SetBoundsDD(TTCParams &params) {
  params.u_lb = Eigen::Vector2f(-0.4f, -1.0f);
  params.u_ub = Eigen::Vector2f(0.4f, 1.0f);
}

void SetBoundsADD(TTCParams &params) {
  params.u_lb = Eigen::Vector2f(-1.0f, -1.0f);
  params.u_ub = Eigen::Vector2f(1.0f, 1.0f);
}

void SetBoundsCAR(TTCParams &params) {
  params.u_lb = Eigen::Vector2f(-0.3f, -0.25f * M_PI);
  params.u_ub = Eigen::Vector2f(0.3f, 0.25f * M_PI);
}

void SetBoundsACAR(TTCParams &params) {
  params.u_lb = Eigen::Vector2f(-1.0f, -0.25f * M_PI);
  params.u_ub = Eigen::Vector2f(1.0f, 0.25f * M_PI);
}

int GetVector(const std::vector<std::string> &parts, int offset, int v_len,
              Eigen::VectorXf &v) {
  if (parts[offset] == "r") {
    v = Eigen::VectorXf::Random(v_len);
    offset++;
  } else {
    v = Eigen::VectorXf::Zero(v_len);
    for (int i = 0; i < v_len; ++i) {
      v[i] = static_cast<float>(std::atof(parts[offset].c_str()));
      offset++;
    }
  }
  return offset;
}

void ConstructGlobalParams(SGDOptParams *opt_params) {
  opt_params->max_time = 10.0f;
  opt_params->opt_mode = OptMode::SGD;
  opt_params->alpha_mode = SGDAlphaMode::PolyakSemiKnown;
  opt_params->sk_mode = SGDSkMode::Filtered;
}

std::vector<std::string> GetAgentParts(int agent_type, Eigen::VectorXf &pos,
                                       bool reactive) {
  Eigen::Vector2f goal(pos[0],
                       pos[1]); // Initialize goal to own position by default
  return GetAgentParts(agent_type, pos, reactive, goal);
}

std::vector<std::string> GetAgentParts(int agent_type, Eigen::VectorXf &pos,
                                       bool reactive, Eigen::Vector2f &goal) {
  std::string type;
  int p_dim, u_dim;
  switch (agent_type) {
  case 0:
    type = "v";
    p_dim = 2;
    u_dim = 2;
    break;
  case 1:
    type = "a";
    p_dim = 4;
    u_dim = 2;
    break;
  case 2:
    type = "dd";
    p_dim = 3;
    u_dim = 2;
    break;
  case 3:
    type = "add";
    p_dim = 5;
    u_dim = 2;
    break;
  case 4:
    type = "car";
    p_dim = 3;
    u_dim = 2;
    break;
  case 5:
    type = "acar";
    p_dim = 5;
    u_dim = 2;
    break;
  default:
    spdlog::error("Invalid virtual agent type: {}", agent_type);
    exit(-1);
  }
  Eigen::VectorXf p = Eigen::VectorXf::Zero(p_dim);
  Eigen::VectorXf u = Eigen::VectorXf::Zero(u_dim);

  // Set XY Heading
  p[0] = pos[0];
  p[1] = pos[1];
  if (agent_type > 1) {
    p[2] = pos[2];
  }
  std::vector<std::string> parts(3 + p_dim + u_dim + 2);
  parts[0] = type;
  parts[1] = "y";
  parts[2] = (reactive ? "y" : "n");
  for (int i = 0; i < p.size(); ++i) {
    parts[3 + i] = std::to_string(p[i]);
  }
  for (int i = 0; i < u.size(); ++i) {
    parts[3 + p_dim + i] = std::to_string(u[i]);
  }
  parts[3 + p_dim + u_dim] = std::to_string(goal[0]);
  parts[3 + p_dim + u_dim + 1] = std::to_string(goal[1]);

  return parts;
}

std::vector<TTCObstacle *> BuildObstacleList(std::vector<Agent> agents, size_t own_idx, double allowable_timeslip) {
  std::vector<TTCObstacle *> obsts;

  for (size_t a_idx = 0; a_idx < agents.size(); ++a_idx) {
    // Check that the difference between the last pose update for the ego agent
    // is within limits compared to other pose updates. 
    double timeslip = agents[a_idx].GetLastUpdated() - agents[own_idx].GetLastUpdated();
    if (timeslip > allowable_timeslip)
    {
      spdlog::warn("Agent at idx: {} ignored due to large timeslip ({} seconds)", a_idx, timeslip);
      continue;
    }

    obsts.push_back(agents[a_idx].GetProblem()->CreateObstacle());
  }
  return obsts;
}

void SetAgentObstacleList(Agent &a, size_t a_idx,
                          std::vector<TTCObstacle *> obsts) {
  a.GetProblem()->params.obsts.clear();
  if (!a.isReactive()) {
    return;
  }
  for (size_t b_idx = 0; b_idx < obsts.size(); ++b_idx) {
    if (b_idx == a_idx) {
      continue;
    }
    float dist = (a.GetProblem()->params.x_0.head<2>() - obsts[b_idx]->p.head<2>()).norm();
    // Ignore any obstacles that cannot interact with us within our ttc horizon
    if (dist < 2.0 * a.GetProblem()->params.vel_limit * a.GetProblem()->params.max_ttc) {
      a.GetProblem()->params.obsts.push_back(obsts[b_idx]);
    }
  }
}