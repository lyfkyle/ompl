/*********************************************************************
* Software License Agreement (BSD License)
*
*  Copyright (c) 2008, Willow Garage, Inc.
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of the Willow Garage nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*********************************************************************/

/* Author: Ioan Sucan */

#include "ompl/geometric/planners/dofatt/RRTDof.h"
#include <limits>
#include "ompl/base/goals/GoalSampleableRegion.h"
#include "ompl/tools/config/SelfConfig.h"
#include <ompl/base/spaces/RealVectorStateSpace.h>

ompl::geometric::RRTDof::RRTDof(const base::SpaceInformationPtr &si, bool addIntermediateStates)
  : base::Planner(si, addIntermediateStates ? "RRTDofintermediate" : "RRTDof")
{
    specs_.approximateSolutions = true;
    specs_.directed = true;

    Planner::declareParam<double>("range", this, &RRTDof::setRange, &RRTDof::getRange, "0.:1.:10000.");
    Planner::declareParam<double>("goal_bias", this, &RRTDof::setGoalBias, &RRTDof::getGoalBias, "0.:.05:1.");
    Planner::declareParam<bool>("intermediate_states", this, &RRTDof::setIntermediateStates, &RRTDof::getIntermediateStates,
                                "0,1");

    addIntermediateStates_ = addIntermediateStates;
}

ompl::geometric::RRTDof::~RRTDof()
{
    freeMemory();
}

void ompl::geometric::RRTDof::clear()
{
    Planner::clear();
    sampler_.reset();
    freeMemory();
    if (nn_)
        nn_->clear();
    lastGoalMotion_ = nullptr;
}

void ompl::geometric::RRTDof::setup()
{
    Planner::setup();
    tools::SelfConfig sc(si_, getName());
    sc.configurePlannerRange(maxDistance_);

    if (!nn_)
        nn_.reset(tools::SelfConfig::getDefaultNearestNeighbors<Motion *>(this));
    nn_->setDistanceFunction([this](const Motion *a, const Motion *b) { return distanceFunction(a, b); });
}

void ompl::geometric::RRTDof::freeMemory()
{
    if (nn_)
    {
        std::vector<Motion *> motions;
        nn_->list(motions);
        for (auto &motion : motions)
        {
            if (motion->state != nullptr)
                si_->freeState(motion->state);
            delete motion;
        }
    }
}

ompl::base::PlannerStatus ompl::geometric::RRTDof::solve(const base::PlannerTerminationCondition &ptc)
{
    checkValidity();
    base::Goal *goal = pdef_->getGoal().get();
    auto *goal_s = dynamic_cast<base::GoalSampleableRegion *>(goal);

    if (!sampler_)
        sampler_ = si_->allocStateSampler();

    OMPL_INFORM("%s: Starting planning with %u states already in datastructure", getName().c_str(), nn_->size());

    std::cout << "!!here!!" << std::endl;

    base::State *rstate = si_->allocState();
    sampler_->sampleUniformNear(rstate, NULL, 0);
    auto state = rstate->as<base::RealVectorStateSpace::StateType>();
    bool solved = (state->values[0] > 0) ? true : false;

    return {solved, false};
}

void ompl::geometric::RRTDof::getPlannerData(base::PlannerData &data) const
{
    Planner::getPlannerData(data);

    std::vector<Motion *> motions;
    if (nn_)
        nn_->list(motions);

    if (lastGoalMotion_ != nullptr)
        data.addGoalVertex(base::PlannerDataVertex(lastGoalMotion_->state));

    for (auto &motion : motions)
    {
        if (motion->parent == nullptr)
            data.addStartVertex(base::PlannerDataVertex(motion->state));
        else
            data.addEdge(base::PlannerDataVertex(motion->parent->state), base::PlannerDataVertex(motion->state));
    }
}

void ompl::geometric::RRTDof::projectWithAtt(ompl::base::State* pSampledState, const ompl::base::State* pNearState, bool isGoal) {
    sampler_->sampleUniformNear(pSampledState, pNearState, isGoal);
}

double ompl::geometric::RRTDof::distanceFunctionWithAtt(const Motion *a, const Motion *b) const {
    auto* pMotion = new Motion(si_);
    si_->copyState(pMotion->state, b->state);
    sampler_->sampleUniformNear(pMotion->state, a->state, 0.0);
    double dist = si_->distance(a->state, pMotion->state);
    delete pMotion;
    return dist;
}

// torch::tensor ompl::geometric::RRTDof::getLocalOccGrid(const vector<float>& state) {
//     idx_x = round(base_x / small_occ_grid_resolution)
//     idx_y = round(base_y / small_occ_grid_resolution)

//     min_y = max(0, idx_y - small_occ_grid_size)
//     max_y = min(self.occ_grid.shape[1], idx_y + small_occ_grid_size)
//     min_x = max(0, idx_x - small_occ_grid_size)
//     max_x = min(self.occ_grid.shape[0], idx_x + small_occ_grid_size)

//     min_y_1 = 0 if min_y != 0 else small_occ_grid_size - idx_y
//     max_y_1 = 2 * small_occ_grid_size if max_y != self.occ_grid.shape[1] else self.occ_grid.shape[1] - idx_y + small_occ_grid_size
//     min_x_1 = 0 if min_x != 0 else small_occ_grid_size - idx_x
//     max_x_1 = 2 * small_occ_grid_size if max_x != self.occ_grid.shape[0] else self.occ_grid.shape[0] - idx_x + small_occ_grid_size

//     # print(state, idx_x, min_x, max_x, min_x_1, max_x_1)
//     # print(state, idx_y, min_y, max_y, min_y_1, max_y_1)

//     local_occ_grid = np.ones((2*small_occ_grid_size, 2*small_occ_grid_size, self.occ_grid.shape[2]), dtype=np.uint8)
//     local_occ_grid[min_x_1:max_x_1, min_y_1:max_y_1] = self.occ_grid[min_x:max_x, min_y:max_y]
// }