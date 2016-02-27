/*******************************************************************************
 * Copyright (c) 2016 Wojciech Migda
 * All rights reserved
 * Distributed under the terms of the GNU LGPL v3
 *******************************************************************************
 *
 * Filename: tco_obj.hpp
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
 * 2016-02-27   wm              Initial version
 *
 ******************************************************************************/

/// included in regression.obj.cc

#ifndef TCO_OBJ_HPP_
#define TCO_OBJ_HPP_

#include "xgboost/objective.h"
#include "src/common/math.h"
#include "dmlc/logging.h"

#include <cmath>

namespace xgboost
{

namespace obj
{

struct MinPRRegression
{
    static float PredTransform(float x)
    {
        return common::Sigmoid(x);
    }

    static bool CheckLabel(float x)
    {
        return x >= 0.0f && x <= 1.0f;
    }

    static float FirstOrderGradient(float predt, float label)
    {
        if ((predt < 0.5f && label < 0.5f) || (predt >= 0.5f && label >= 0.5f))
        {
            return predt - label;
        }
        else
        {
            return 20.f * (predt - label);
        }
    }

    static float SecondOrderGradient(float predt, float label)
    {
        const float eps = 1e-16f;

        return std::max(predt * (1.0f - predt), eps);
    }

    static float ProbToMargin(float base_score)
    {
        CHECK(base_score > 0.0f && base_score < 1.0f)
            << "base_score must be in (0,1) for logistic loss";

        return -std::log(1.0f / base_score - 1.0f);
    }

    static const char* LabelErrorMsg()
    {
        return "label must be in [0,1] for logistic regression";
    }

    static const char* DefaultEvalMetric()
    {
        return "rmse";
    }
};


struct MinPRClassification : public MinPRRegression
{
    static const char* DefaultEvalMetric()
    {
        std::cout << "MinPRClassification::DefaultEvalMetric" << std::endl;
        return "error";
    }
};


XGBOOST_REGISTER_OBJECTIVE(MinPRClassification, "binary:minPRlogistic")
.describe("min(P,R) for binary classification task.")
.set_body([]() { return new RegLossObj<MinPRClassification>(); });


}  // namespace obj

}  // namespace xgboost


#endif /* TCO_OBJ_HPP_ */
