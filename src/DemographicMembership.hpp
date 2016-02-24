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



std::vector<int>
DemographicMembership::predict(const int test_type,
    std::vector<std::string> & i_training,
    std::vector<std::string> & i_testing) const
{
    typedef num::array2d<real_type> array_type;

    const std::string cs_feature_names = "CONSUMER_ID,AGE,GENDER,REGISTRATION_ROUTE,REGISTRATION_CONTEXT,REGISTRATION_DAYS,OPTIN,IS_DELETED,MIGRATED_USER_TYPE,SOCIAL_AUTH_FACEBOOK,SOCIAL_AUTH_TWITTER,SOCIAL_AUTH_GOOGLE,PAGE_IMPRESSIONS,SEARCH_EVENTS,VISITS,VOD_VIEW_VISITS,PAGE_IMPRESSION_VISITS,SEARCH_EVENT_VISITS,TOTAL_DWELL,VOD_VIEWS_DWELL,PAGE_IMPRESSIONS_DWELL,VIDEO_STOPS,VIDEO_COMPLETIONS,MILESTONES_25,MILESTONES_50,MILESTONES_75,VIDEO_CRITICAL_ERRORS,RESUME_NEWS,RESUME_PREVIOUS,BREAKFAST_PAGE_VIEWS,MORNING_PAGE_VIEWS,LUNCHTIME_PAGE_VIEWS,AFTERNOON_PAGE_VIEWS,EARLY_PAGE_VIEWS,LATE_PAGE_VIEWS,POST_PAGE_VIEWS,NIGHT_TIME_PAGE_VIEWS,BREAKFAST_VISITS,MORNING_VISITS,LUNCHTIME_VISITS,AFTERNOON_VISITS,EARLY_PEAK_VISITS,LATE_PEAK_VISITS,POST_PEAK_VISITS,NIGHTTIME_VISITS,TOTAL_VIEWS,WARD_WKDAY_1_2,WARD_WKDAY_3_9,WARD_WKDAY_10_16,WARD_WKDAY_17_19,WARD_WKDAY_20_24,WARD_WKEND_1_2,WARD_WKEND_3_9,WARD_WKEND_10_13,WARD_WKEND_14_20,WARD_WKEND_21_24,UNI_CLUSTER_1,UNI_CLUSTER_2,UNI_CLUSTER_3,UNI_CLUSTER_4,UNI_CLUSTER_5,UNI_CLUSTER_6,UNI_CLUSTER_7,UNI_CLUSTER_8,UNI_CLUSTER_9,UNI_CLUSTER_10,UNI_CLUSTER_11,UNI_CLUSTER_12,UNI_CLUSTER_13,UNI_CLUSTER_14,UNI_CLUSTER_15,UNI_CLUSTER_16,UNI_CLUSTER_17,UNI_CLUSTER_18,UNI_CLUSTER_19,UNI_CLUSTER_20,UNI_CLUSTER_21,UNI_CLUSTER_22,UNI_CLUSTER_23,UNI_CLUSTER_24,UNI_CLUSTER_25,UNI_CLUSTER_26,UNI_CLUSTER_27,UNI_CLUSTER_28,UNI_CLUSTER_29,UNI_CLUSTER_30,UNI_CLUSTER_31,UNI_CLUSTER_32,UNI_CLUSTER_33,VIEWS_ON_WEBSITE,VIEWS_ON_IOS,VIEWS_ON_ANDROID,BREAKFAST_VIEWS,MORNING_VIEWS,LUNCHTIME_VIEWS,AFTERNOON_VIEWS,EARLY_PEAK_VIEWS,LATE_PEAK_VIEWS,POST_PEAK_VIEWS,NIGHT_TIME_VIEWS,CATCHUP_VIEWS,ARCHIVE_VIEWS,VIEWS_MAIN,VIEWS_AFF1,VIEWS_AFF2,VIEWS_AFF3,VIEWS_AFF4,OTHER_VIEWS,FLAG_WARD_WKDAY_1_2,FLAG_WARD_WKDAY_3_9,FLAG_WARD_WKDAY_10_16,FLAG_WARD_WKDAY_17_19,FLAG_WARD_WKDAY_20_24,FLAG_WARD_WKEND_1_2,FLAG_WARD_WKEND_3_9,FLAG_WARD_WKEND_10_13,FLAG_WARD_WKEND_14_20,FLAG_WARD_WKEND_21_24,FLAG_UNI_CLUSTER_1,FLAG_UNI_CLUSTER_2,FLAG_UNI_CLUSTER_3,FLAG_UNI_CLUSTER_4,FLAG_UNI_CLUSTER_5,FLAG_UNI_CLUSTER_6,FLAG_UNI_CLUSTER_7,FLAG_UNI_CLUSTER_8,FLAG_UNI_CLUSTER_9,FLAG_UNI_CLUSTER_10,FLAG_UNI_CLUSTER_11,FLAG_UNI_CLUSTER_12,FLAG_UNI_CLUSTER_13,FLAG_UNI_CLUSTER_14,FLAG_UNI_CLUSTER_15,FLAG_UNI_CLUSTER_16,FLAG_UNI_CLUSTER_17,FLAG_UNI_CLUSTER_18,FLAG_UNI_CLUSTER_19,FLAG_UNI_CLUSTER_20,FLAG_UNI_CLUSTER_21,FLAG_UNI_CLUSTER_22,FLAG_UNI_CLUSTER_23,FLAG_UNI_CLUSTER_24,FLAG_UNI_CLUSTER_25,FLAG_UNI_CLUSTER_26,FLAG_UNI_CLUSTER_27,FLAG_UNI_CLUSTER_28,FLAG_UNI_CLUSTER_29,FLAG_UNI_CLUSTER_30,FLAG_UNI_CLUSTER_31,FLAG_UNI_CLUSTER_32,FLAG_UNI_CLUSTER_33,FLAG_WEBSITE,FLAG_IOS,FLAG_ANDROID,FLAG_BREAKFAST_VIEWS,FLAG_MORNING_VIEWS,FLAG_LUNCHTIME_VIEWS,FLAG_AFTERNOON_VIEWS,FLAG_EARLY_PEAK_VIEWS,FLAG_LATE_PEAK_VIEWS,FLAG_POST_PEAK_VIEWS,FLAG_NIGHT_TIME_VIEWS,FLAG_CATCHUP_VIEWS,FLAG_ARCHIVE_VIEWS,FLAG_MAIN,FLAG_AFF1,FLAG_AFF2,FLAG_AFF3,FLAG_AFF4,FLAG_OTHER_VIEWS,PROP_WARD_WKDAY_1_2,PROP_WARD_WKDAY_3_9,PROP_WARD_WKDAY_10_16,PROP_WARD_WKDAY_17_19,PROP_WARD_WKDAY_20_24,PROP_WARD_WKEND_1_2,PROP_WARD_WKEND_3_9,PROP_WARD_WKEND_10_13,PROP_WARD_WKEND_14_20,PROP_WARD_WKEND_21_24,PROP_UNI_CLUSTER_1,PROP_UNI_CLUSTER_2,PROP_UNI_CLUSTER_3,PROP_UNI_CLUSTER_4,PROP_UNI_CLUSTER_5,PROP_UNI_CLUSTER_6,PROP_UNI_CLUSTER_7,PROP_UNI_CLUSTER_8,PROP_UNI_CLUSTER_9,PROP_UNI_CLUSTER_10,PROP_UNI_CLUSTER_11,PROP_UNI_CLUSTER_12,PROP_UNI_CLUSTER_13,PROP_UNI_CLUSTER_14,PROP_UNI_CLUSTER_15,PROP_UNI_CLUSTER_16,PROP_UNI_CLUSTER_17,PROP_UNI_CLUSTER_18,PROP_UNI_CLUSTER_19,PROP_UNI_CLUSTER_20,PROP_UNI_CLUSTER_21,PROP_UNI_CLUSTER_22,PROP_UNI_CLUSTER_23,PROP_UNI_CLUSTER_24,PROP_UNI_CLUSTER_25,PROP_UNI_CLUSTER_26,PROP_UNI_CLUSTER_27,PROP_UNI_CLUSTER_28,PROP_UNI_CLUSTER_29,PROP_UNI_CLUSTER_30,PROP_UNI_CLUSTER_31,PROP_UNI_CLUSTER_32,PROP_UNI_CLUSTER_33,PROP_WEBSITE,PROP_IOS,PROP_ANDROID,PROP_BREAKFAST_VIEWS,PROP_MORNING_VIEWS,PROP_LUNCHTIME_VIEWS,PROP_AFTERNOON_VIEWS,PROP_EARLY_PEAK_VIEWS,PROP_LATE_PEAK_VIEWS,PROP_POST_PEAK_VIEWS,PROP_NIGHT_TIME_VIEWS,PROP_CATCHUP_VIEWS,PROP_ARCHIVE_VIEWS,PROP_MAIN,PROP_AFF1,PROP_AFF2,PROP_AFF3,PROP_AFF4,PROP_OTHER_VIEWS,PLATFORM_CENTRE,TOD_CENTRE,CONTENT_CENTRE,INTEREST_BEAUTY,INTEREST_TECHNOLOGY,INTEREST_FASHION,INTEREST_COOKING,INTEREST_HOME,INTEREST_QUALITY,INTEREST_DEALS,INTEREST_GREEN,DEMO_X";


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
//    std::cout << std::endl;

    // drop the CONSUMER_ID column
    const array_type train_data({i_train_data.shape().first, i_train_data.shape().second - 1}, i_train_data[i_train_data.columns(1, -1)]);
    const array_type test_data({i_test_data.shape().first, i_test_data.shape().second - 1}, i_test_data[i_test_data.columns(1, -1)]);

    std::cerr << "train_data shape: " << train_data.shape() << std::endl;
    std::cerr << "test_data shape: " << test_data.shape() << std::endl;

    // prepare placeholder for raw matrix later used by xgboost
    std::vector<float> train_vec = train_data.tovector();
    std::cout << train_vec.size() << std::endl;

//    std::copy(train_vec.cbegin(), train_vec.cbegin() + 10, std::ostream_iterator<real_type>(std::cerr, ", "));
//    std::cout << std::endl;

    std::unique_ptr<void, int (*)(DMatrixHandle)> tr_dmat(
        XGDMatrixCreateFromMat(
            train_vec.data(),
            train_data.shape().first,
            train_data.shape().second, NAN),
        XGDMatrixFree);


    return std::vector<int>(i_testing.size(), 1);
}

#endif /* DEMOGRAPHICMEMBERSHIP_HPP_ */
