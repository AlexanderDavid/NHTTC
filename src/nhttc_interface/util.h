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

#include <Eigen/Core>

#include "agent.h"

#include <sgd/ttc_sgd_problem.h>

std::vector<std::string> GetParts(std::string s, char delim);
std::vector<std::vector<std::string>> LoadFileByToken(std::string file_name, int n_skip, char delim);

void SetBoundsV(TTCParams &params);
void SetBoundsA(TTCParams &params);
void SetBoundsDD(TTCParams &params);
void SetBoundsADD(TTCParams &params);
void SetBoundsCAR(TTCParams &params);
void SetBoundsACAR(TTCParams &params);

int GetVector(const std::vector<std::string>& parts, int offset, int v_len, Eigen::VectorXf& v);

std::vector<std::string> GetAgentParts(int agent_type, Eigen::VectorXf& pos, bool reactive); // Default goal to current position
std::vector<std::string> GetAgentParts(int agent_type, Eigen::VectorXf& pos, bool reactive, Eigen::Vector2f& goal);

void ConstructGlobalParams(SGDOptParams *opt_params);
std::vector<TTCObstacle *> BuildObstacleList(std::vector<Agent> agents, size_t own_idx, double allowable_timeslip);