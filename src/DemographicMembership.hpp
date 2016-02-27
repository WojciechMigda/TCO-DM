/*******************************************************************************
 * Copyright (c) 2016 Wojciech Migda
 * All rights reserved
 * Distributed under the terms of the GNU LGPL v3
 *******************************************************************************
 *
 * Filename: DemographicMembership.hpp
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
 * 2016-02-22   wm              Initial version
 *
 ******************************************************************************/


#ifndef DEMOGRAPHICMEMBERSHIP_HPP_
#define DEMOGRAPHICMEMBERSHIP_HPP_

#include "num.hpp"
#include "array2d.hpp"

#include "xgboost/c_api.h"

#include <vector>
#include <string>
#include <iostream>
#include <memory>
#include <cmath>
#include <cstring>
#include <iterator>
#include <unordered_map>

typedef float real_type;


/*
 * Wrapper which returns DMatrixHandle directly and not through out param
 */
DMatrixHandle XGDMatrixCreateFromMat(
    const float *data,
    bst_ulong nrow,
    bst_ulong ncol,
    float missing)
{
    DMatrixHandle dmat{nullptr};

    XGDMatrixCreateFromMat(data, nrow, ncol, missing, &dmat);

    return dmat;
}

/*
 * Wrapper which returns BoosterHandle directly and not through out param
 */
BoosterHandle XGBoosterCreate(const DMatrixHandle dmats[], bst_ulong len)
{
    BoosterHandle booster{nullptr};

    XGBoosterCreate(dmats, len, &booster);

    return booster;
}

/*
 * Competition mandated class
 */
struct DemographicMembership
{
    enum TestType
    {
        Example,
        Provisional,
        System,
        Local
    };

    std::vector<int>
    predict(
        const int test_type,
        std::vector<std::string> & i_training,
        std::vector<std::string> & i_testing) const;
};

/*
 * NA/numeric converter for loadtxt
 */
auto na_xlt = [](const char * str) -> real_type
{
    return (std::strcmp(str, "NA") == 0) ? NAN : std::strtod(str, nullptr);
};

/*
 * GENDER converter for loadtxt
 */
auto gender_xlt = [](const char * str) -> real_type
{
    if (strcmp(str, "U") == 0)
    {
        return NAN;
    }
    else if (strcmp(str, "M") == 0)
    {
        return 0.;
    }
    else if (strcmp(str, "F") == 0)
    {
        return 1.;
    }
    else
    {
        assert(false);
        return NAN;
    }
};

/*
 * generic Y/N/NA converter for loadtxt
 */
auto yes_no_xlt = [](const char * str) -> real_type
{
    if (strcmp(str, "NA") == 0)
    {
        return NAN;
    }
    else if (strcmp(str, "Y") == 0)
    {
        return 0.;
    }
    else if (strcmp(str, "N") == 0)
    {
        return 1.;
    }
    else
    {
        assert(false);
        return NAN;
    }
};

/*
 * generic pattern converter for loadtxt
 */
auto from_list_xlt = [](const std::vector<std::string> & patterns, const char * str) -> real_type
{
    auto matched_it = std::find_if(patterns.cbegin(), patterns.cend(),
        [&str](const std::string & what)
        {
            return strcmp(what.c_str(), str) == 0;
        }
    );

    if (matched_it != patterns.cend())
    {
        return std::distance(patterns.cbegin(), matched_it);
    }
    else if (strcmp(str, "NA") == 0)
    {
        return NAN;
    }
    else
    {
        assert(false);
        return NAN;
    }
};

namespace XGB
{

std::unique_ptr<void, int (*)(BoosterHandle)>
fit(const num::array2d<real_type> & train_data,
    const std::vector<float> & train_y,
    const std::map<const std::string, const std::string> & params,
    const int n_iter)
{
    // prepare placeholder for raw matrix later used by xgboost
    std::vector<float> train_vec = train_data.tovector();
    std::cerr << "train_vec size: " << train_vec.size() << std::endl;

    std::unique_ptr<void, int (*)(DMatrixHandle)> tr_dmat(
        XGDMatrixCreateFromMat(
            train_vec.data(),
            train_data.shape().first,
            train_data.shape().second, NAN),
        XGDMatrixFree);

    // attach response vector to tr_dmat
    XGDMatrixSetFloatInfo(tr_dmat.get(), "label", train_y.data(), train_y.size());

    const DMatrixHandle cache[] = {tr_dmat.get()};

    // create Booster with attached tr_dmat
    std::unique_ptr<void, int (*)(BoosterHandle)> booster(
            XGBoosterCreate(cache, 1UL),
            XGBoosterFree);

    for (const auto & kv : params)
    {
        std::cerr << kv.first << " => " << kv.second << std::endl;
        XGBoosterSetParam(booster.get(), kv.first.c_str(), kv.second.c_str());
    }

    for (int iter{0}; iter < n_iter; ++iter)
    {
        XGBoosterUpdateOneIter(booster.get(), iter, tr_dmat.get());
    }

    return booster;
}


std::vector<float>
predict(
    BoosterHandle booster,
    const num::array2d<real_type> & test_data)
{
    std::vector<float> test_vec = test_data.tovector();
    std::cerr << "test_vec size: " << test_vec.size() << std::endl;

    std::unique_ptr<void, int (*)(DMatrixHandle)> te_dmat(
        XGDMatrixCreateFromMat(
            test_vec.data(),
            test_data.shape().first,
            test_data.shape().second, NAN),
        XGDMatrixFree);

    bst_ulong y_hat_len{0};
    const float * y_hat_proba{nullptr};
    XGBoosterPredict(booster, te_dmat.get(), 0, 0, &y_hat_len, &y_hat_proba);
    std::cerr << "Got y_hat_proba of length " << y_hat_len << std::endl;

    std::vector<float> y_hat(y_hat_proba, y_hat_proba + y_hat_len);

    return y_hat;
}

}


num::array2d<real_type>
get_dummies(std::valarray<real_type> && what)
{
    auto unique = std::unordered_set<real_type>();

    std::copy_if(std::begin(what), std::end(what), std::inserter(unique, unique.end()),
        [](real_type v)
        {
            return !std::isnan(v);
        }
    );

//    std::cout << what.size() << " " << unique.size() << std::endl;

    num::array2d<real_type> newmat({what.size(), unique.size()}, 0.0);

    int index{0};

    for (const auto v : unique)
    {
        const std::valarray<bool> bool_mask = (what == v);
        std::valarray<real_type> mask(bool_mask.size());

        std::copy(std::begin(bool_mask), std::end(bool_mask), std::begin(mask));

        newmat[newmat.column(index)] = mask;

        ++index;
    }

    return newmat;
}

num::array2d<real_type>
one_hot(const num::array2d<real_type> & what, std::vector<std::size_t> && columns)
{
    num::array2d<real_type> newmat(what);

    for (auto col : columns)
    {
        const auto dummies = get_dummies(what[what.column(col)]);
        newmat = num::add_columns<real_type>(newmat, dummies);
    }

    // we will only delete columns right-to-left
    std::sort(columns.begin(), columns.end(), std::greater<std::size_t>());
    for (auto col : columns)
    {
        newmat = num::del_column<real_type>(newmat, col);
    }

    return newmat;
}



std::vector<int>
DemographicMembership::predict(const int test_type,
    std::vector<std::string> & i_training,
    std::vector<std::string> & i_testing) const
{
    typedef num::array2d<real_type> array_type;


    std::cerr << "predict(): test_type: " << test_type << std::endl;

    static const std::vector<std::string> REGISTRATION_ROUTE_PATTERNS{{"A", "B", "C", "D"}};
    static const std::vector<std::string> REGISTRATION_CONTEXT_PATTERNS{
        {
            "A", "B", "C", "D", "E", "F", "G", "H",
            "I", "J", "K", "L", "M", "N", "O", "P",
            "Q", "R", "S", "T", "U", "V", "W", "X",
            "Y", "Z", "0", "1", "2", "3"
        }};
    static const std::vector<std::string> MIGRATED_USER_PATTERNS{{"A", "B", "C", "D", "E"}};

    const num::loadtxtCfg<real_type>::converters_type converters =
        {
            {1, na_xlt}, // AGE
            {5, na_xlt}, // REGISTRATION_DAYS
            {232, na_xlt}, // PLATFORM_CENTRE
            {233, na_xlt}, // TOD_CENTRE
            {234, na_xlt}, // CONTENT_CENTRE
            {235, na_xlt}, // INTEREST_BEAUTY
            {236, na_xlt}, // INTEREST_TECHNOLOGY
            {237, na_xlt}, // INTEREST_FASHION
            {238, na_xlt}, // INTEREST_COOKING
            {239, na_xlt}, // INTEREST_HOME
            {240, na_xlt}, // INTEREST_QUALITY
            {241, na_xlt}, // INTEREST_DEALS
            {242, na_xlt}, // INTEREST_GREEN
            // ---------------------------------------------------------
            {2, gender_xlt}, // GENDER
            {3, [](const char * str){return from_list_xlt(REGISTRATION_ROUTE_PATTERNS, str);}}, // REGISTRATION_ROUTE
            {4, [](const char * str){return from_list_xlt(REGISTRATION_CONTEXT_PATTERNS, str);}}, // REGISTRATION_CONTEXT
            {6, yes_no_xlt}, // OPTIN
            {7, yes_no_xlt}, // IS_DELETED
            {8, [](const char * str){return from_list_xlt(MIGRATED_USER_PATTERNS, str);}}, // MIGRATED_USER_TYPE
            {9, yes_no_xlt}, // SOCIAL_AUTH_FACEBOOK
            {10, yes_no_xlt}, // SOCIAL_AUTH_TWITTER
            {11, yes_no_xlt}, // SOCIAL_AUTH_GOOGLE
        };

    array_type i_train_data =
        num::loadtxt(
            std::move(i_training),
            std::move(
                num::loadtxtCfg<real_type>()
                .delimiter(',')
                .converters(num::loadtxtCfg<real_type>::converters_type{converters})
            )
        );

    const
    array_type i_test_data =
        num::loadtxt(
            std::move(i_testing),
            std::move(
                num::loadtxtCfg<real_type>()
                .delimiter(',')
                .converters(num::loadtxtCfg<real_type>::converters_type{converters})
            )
        );

    // retrieve response vector
    const array_type::varray_type train_y_va = i_train_data[i_train_data.column(-1)];
    const std::vector<float> train_y(std::begin(train_y_va), std::end(train_y_va));

    std::cerr << "train_y size: " << train_y.size() << std::endl;
//    std::copy(train_y.cbegin(), train_y.cbegin() + 10, std::ostream_iterator<real_type>(std::cerr, ", "));
//    std::cerr << std::endl;

    // drop the CONSUMER_ID column
    array_type test_data({i_test_data.shape().first, i_test_data.shape().second - 1}, i_test_data[i_test_data.columns(1, -1)]);
    // drop the CONSUMER_ID and DEMO_X columns
    array_type train_data({i_train_data.shape().first, i_train_data.shape().second - 2}, i_train_data[i_train_data.columns(1, -2)]);

    std::cerr << "train_data shape: " << train_data.shape() << std::endl;
    std::cerr << "test_data shape: " << test_data.shape() << std::endl;

    { // one hot
        array_type full_data({train_data.shape().first + test_data.shape().first, train_data.shape().second}, 0);
        full_data[full_data.rows(0, train_data.shape().first - 1)] = train_data[train_data.rows(0, -1)];
        full_data[full_data.rows(train_data.shape().first, -1)] = test_data[test_data.rows(0, -1)];

        const auto & c_full_data(full_data);

        // <<< feature engineering with ZERO balance

        full_data[full_data.column(18)] = c_full_data[full_data.column(18)] / c_full_data[full_data.column(17)];
        full_data[full_data.column(19)] = c_full_data[full_data.column(19)] / c_full_data[full_data.column(17)];

        // >>> feature engineering

        full_data = one_hot(full_data, {1, 2, 3, 7, 231, 232, 233});

        train_data = array_type(
            {train_data.shape().first, full_data.shape().second},
            c_full_data[full_data.rows(0, train_data.shape().first - 1)]);
        test_data = array_type(
            {test_data.shape().first, full_data.shape().second},
            c_full_data[full_data.rows(train_data.shape().first, -1)]);
    }

    std::cerr << "OneHot train_data shape: " << train_data.shape() << std::endl;
    std::cerr << "OneHot test_data shape: " << test_data.shape() << std::endl;

    // booster parameters
    const std::map<const std::string, const std::string> params
    {
//        {"booster", "gblinear"},
        {"booster", "gbtree"}, // default
        {"reg_alpha", "0"},
        {"colsample_bytree", "0.65"},
        {"silent", "1"},
        {"colsample_bylevel", "1"},
        {"scale_pos_weight", "1"},
        {"learning_rate", "0.045"},
        {"missing", "nan"},
        {"max_delta_step", "0"},
        {"base_score", "0.5"},
        {"n_estimators", "500"},
        {"subsample", "0.85"},
        {"reg_lambda", "1"},
        {"seed", "0"},
        {"min_child_weight", "65"},

        {"objective", "rank:pairwise"},
        {"max_depth", "7"},
        {"gamma", "0"}
    };

    auto booster = XGB::fit(train_data, train_y, params, std::stoi(params.at("n_estimators")));

    const auto y_hat_proba = XGB::predict(booster.get(), test_data);

    std::vector<int> y_hat(test_data.shape().first);
    std::transform(y_hat_proba.cbegin(), y_hat_proba.cend(), y_hat.begin(),
        [](const float what)
        {
            return what > 0.5;
        }
    );

    return y_hat;
}

#endif /* DEMOGRAPHICMEMBERSHIP_HPP_ */
