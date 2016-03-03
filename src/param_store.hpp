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
//        {"booster", "gblinear"},
    {"booster", "gbtree"}, // default
    {"reg_alpha", "0"},
    {"colsample_bytree", "0.7709"},
    {"silent", "1"},
    {"colsample_bylevel", "1"},
    {"scale_pos_weight", "1"},
    {"learning_rate", "0.045"},
    {"missing", "nan"},
    {"max_delta_step", "0"},
    {"base_score", "0.5"},
    {"n_estimators", "600"},
    {"subsample", "0.6549"},
    {"reg_lambda", "1"},
    {"seed", "0"},
    {"min_child_weight", "85"},

    {"objective", "rank:pairwise"},
//        {"objective", "binary:logitraw"},
//        {"objective", "binary:logistic"},
    {"max_depth", "5"},
    {"gamma", "0.7745"}
};


}  // namespace params


#endif /* PARAM_STORE_HPP_ */
