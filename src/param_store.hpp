/*******************************************************************************
 * Copyright (c) 2016 Wojciech Migda
 * All rights reserved
 * Distributed under the terms of the GNU LGPL v3
 *******************************************************************************
 *
 * Filename: param_store.hpp
 *
 * Description:
 *      description
 *
 * Authors:
 *          Wojciech Migda (wm)
 *
 *******************************************************************************
 * History:
 * --------
 * Date         Who  Ticket     Description
 * ----------   ---  ---------  ------------------------------------------------
 * 2016-03-03   wm              Initial version
 *
 ******************************************************************************/


#ifndef PARAM_STORE_HPP_
#define PARAM_STORE_HPP_

namespace params
{

const std::map<const std::string, const std::string> CURRENT
{
//    {"booster", "gblinear"},
    {"booster", "gbtree"},
    {"reg_alpha", "0"},
    {"colsample_bytree", "0.65"},
    {"silent", "1"},
    {"colsample_bylevel", "1"},
    {"scale_pos_weight", "1"},
    {"learning_rate", "0.045"},
    {"missing", "nan"},
    {"max_delta_step", "0"},
    {"base_score", "0.5"},
    {"n_estimators", "600"},
    {"subsample", "0.85"},
    {"reg_lambda", "1"},
    {"seed", "0"},
    {"min_child_weight", "65"},

    {"objective", "rank:pairwise"},
    {"num_pairsample", "2"},
//    {"objective", "binary:logitraw"},
//    {"objective", "binary:logistic"},
    {"max_depth", "7"},
    {"gamma", "0"}
};

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

const std::map<const std::string, const std::string> sub8
{
    // LB: 811038.67
    // CV: 810404
    // no FE

    {"booster", "gbtree"},
    {"reg_alpha", "0"},
    {"colsample_bytree", "0.65"},
    {"silent", "1"},
    {"colsample_bylevel", "1"},
    {"scale_pos_weight", "1"},
    {"learning_rate", "0.045"},
    {"missing", "nan"},
    {"max_delta_step", "0"},
    {"base_score", "0.5"},
    {"n_estimators", "600"},
    {"subsample", "0.85"},
    {"reg_lambda", "1"},
    {"seed", "0"},
    {"min_child_weight", "65"},

    {"objective", "rank:pairwise"},
    {"max_depth", "7"},
    {"gamma", "0"}
};


const std::map<const std::string, const std::string> sub33
{
    // sub8, n_estimators=590

    // LB: 810787.55
    // CV: 811512
    // no FE

    {"booster", "gbtree"},
    {"reg_alpha", "0"},
    {"colsample_bytree", "0.65"},
    {"silent", "1"},
    {"colsample_bylevel", "1"},
    {"scale_pos_weight", "1"},
    {"learning_rate", "0.045"},
    {"missing", "nan"},
    {"max_delta_step", "0"},
    {"base_score", "0.5"},
    {"n_estimators", "590"},
    {"subsample", "0.85"},
    {"reg_lambda", "1"},
    {"seed", "0"},
    {"min_child_weight", "65"},

    {"objective", "rank:pairwise"},
    {"max_depth", "7"},
    {"gamma", "0"}
};


const std::map<const std::string, const std::string> sub26
{
    // sub8, n_estimators=610

    // LB: 810682.24
    // CV: 810784
    // no FE

    {"booster", "gbtree"},
    {"reg_alpha", "0"},
    {"colsample_bytree", "0.65"},
    {"silent", "1"},
    {"colsample_bylevel", "1"},
    {"scale_pos_weight", "1"},
    {"learning_rate", "0.045"},
    {"missing", "nan"},
    {"max_delta_step", "0"},
    {"base_score", "0.5"},
    {"n_estimators", "610"},
    {"subsample", "0.85"},
    {"reg_lambda", "1"},
    {"seed", "0"},
    {"min_child_weight", "65"},

    {"objective", "rank:pairwise"},
    {"max_depth", "7"},
    {"gamma", "0"}
};


const std::map<const std::string, const std::string> sub24
{
    // from hyperopt log 4 [10-fold] (rank 4th) where it scored 813247.42337

    // LB: 810479.86
    // CV: 809573
    // no FE

    {"booster", "gbtree"},
    {"reg_alpha", "0"},
    {"colsample_bytree", "0.4590"},
    {"silent", "1"},
    {"colsample_bylevel", "1"},
    {"scale_pos_weight", "1"},
    {"learning_rate", "0.045"},
    {"missing", "nan"},
    {"max_delta_step", "0"},
    {"base_score", "0.5"},
    {"n_estimators", "720"},
    {"subsample", "0.6005"},
    {"reg_lambda", "1"},
    {"seed", "0"},
    {"min_child_weight", "130"},

        {"num_pairsample", "2"},

    {"objective", "rank:pairwise"},
    {"max_depth", "6"},
    {"gamma", "0"}
};


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

const std::map<const std::string, const std::string> sub18
{
    // sub8, two FE: PROP_PAGE_IMPRESSIONS_DWELL, PROP_VOD_VIEWS_DWELL

    // LB: 810488.63
    // CV: 810224
    // FE (only top-2)

    {"booster", "gbtree"},
    {"reg_alpha", "0"},
    {"colsample_bytree", "0.65"},
    {"silent", "1"},
    {"colsample_bylevel", "1"},
    {"scale_pos_weight", "1"},
    {"learning_rate", "0.045"},
    {"missing", "nan"},
    {"max_delta_step", "0"},
    {"base_score", "0.5"},
    {"n_estimators", "600"},
    {"subsample", "0.85"},
    {"reg_lambda", "1"},
    {"seed", "0"},
    {"min_child_weight", "65"},

    {"objective", "rank:pairwise"},
    {"max_depth", "7"},
    {"gamma", "0"}
};


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

}  // namespace params


#endif /* PARAM_STORE_HPP_ */
