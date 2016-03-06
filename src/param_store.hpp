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


///////////// EXTRA ???? 'colsample_bytree': 0.9705978190927813,
#if 0
// LB: .
// CV: 815894
// best score: 814257.42870
// best params: {'colsample_bytree': 0.9705978190927813,
// 'scale_pos_weight': 0.38570090801277784, 'min_child_weight': 175, 'n_estimators': 510,
// 'subsample': 0.7168114425646201, 'objective':'reg:linear', 'max_depth': 6, 'gamma': 1.676750571566339}
#endif


const std::map<const std::string, const std::string> CURRENT
{
//    {"booster", "gblinear"},
    {"booster", "gbtree"},
    {"reg_alpha", "0"},
    {"colsample_bytree", "0.5525129874711902"},
    {"silent", "1"},
    {"colsample_bylevel", "1"},
    {"scale_pos_weight", "0.38570090801277784"},
    {"learning_rate", "0.045"},
    {"missing", "nan"},
    {"max_delta_step", "0"},
    {"base_score", "0.5"},
    {"n_estimators", "510"},
    {"subsample", "0.7168114425646201"},
    {"reg_lambda", "1"},
    {"seed", "0"},
    {"min_child_weight", "175"},

//    {"objective", "rank:pairwise"},
//    {"num_pairsample", "3"},
//    {"objective", "binary:logitraw"},
    {"objective", "reg:linear"},
//    {"objective", "binary:logistic"},
    {"max_depth", "6"},
    {"gamma", "1.676750571566339"}
};


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

const std::map<const std::string, const std::string> sub40
{
    // best entry from hyper.pairwise.f20.num_pairsample.noGamma.noFE.log

    // best score: 813737.46276
    // best params: {'colsample_bytree': 0.5235071920939005, 'min_child_weight': 225, 'num_pairsample': 3,
    // 'n_estimators': 740, 'subsample': 0.5008072038378799, 'objective': 'rank:pairwise', 'max_depth': 8}

    // LB: 811878.79
    // CV: 812692
    // noFE

    {"booster", "gbtree"},
    {"reg_alpha", "0"},
    {"colsample_bytree", "0.5235071920939005"},
    {"silent", "1"},
    {"colsample_bylevel", "1"},
    {"scale_pos_weight", "0.3948607027148843"}, // remove
    {"learning_rate", "0.045"},
    {"missing", "nan"},
    {"max_delta_step", "0"},
    {"base_score", "0.5"},
    {"n_estimators", "740"},
    {"subsample", "0.5008072038378799"},
    {"reg_lambda", "1"},
    {"seed", "0"},
    {"min_child_weight", "225"},

    {"objective", "rank:pairwise"},

    {"num_pairsample", "3"},
    {"max_depth", "8"},
    {"gamma", "0"}
};


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////


const std::map<const std::string, const std::string> sub39
{
    // best score: 814667.07663 from hyper.binlogistic.f10.noFE.log

    // best params: {'colsample_bytree': 0.7383538426315808, 'scale_pos_weight': 0.40794377329914555,
    // 'min_child_weight': 60, 'n_estimators': 590, 'subsample': 0.7652394081294241, 'objective': 'binary:logistic',
    // 'max_depth': 4, 'gamma': 0.9996994618481485}

    // LB: 811104.3
    // CV: 812550
    // noFE

    {"booster", "gbtree"},
    {"reg_alpha", "0"},
    {"colsample_bytree", "0.7383538426315808"},
    {"silent", "1"},
    {"colsample_bylevel", "1"},
    {"scale_pos_weight", "0.40794377329914555"},
    {"learning_rate", "0.045"},
    {"missing", "nan"},
    {"max_delta_step", "0"},
    {"base_score", "0.5"},
    {"n_estimators", "590"},
    {"subsample", "0.7652394081294241"},
    {"reg_lambda", "1"},
    {"seed", "0"},
    {"min_child_weight", "60"},

    {"objective", "binary:logistic"},
    {"max_depth", "4"},
    {"gamma", "0.9996994618481485"}
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


const std::map<const std::string, const std::string> sub46
{
    // second best score: 814959.85754 from hyper.binlogitraw.2.f10.noFE.log

    // best score: 814959.85754
    // best params: {'colsample_bytree': 0.8713724505384853,
    // 'scale_pos_weight': 0.682580867766655, 'min_child_weight': 85, 'n_estimators': 590,
    // 'subsample': 0.8051026646094943, 'objective': 'binary:logitraw', 'max_depth': 4, 'gamma': 1.4120960639617184}

    // LB: 811022.75
    // CV: 812497
    // noFE

    {"booster", "gbtree"},
    {"reg_alpha", "0"},
    {"colsample_bytree", "0.8713724505384853"},
    {"silent", "1"},
    {"colsample_bylevel", "1"},
    {"scale_pos_weight", "0.682580867766655"},
    {"learning_rate", "0.045"},
    {"missing", "nan"},
    {"max_delta_step", "0"},
    {"base_score", "0.5"},
    {"n_estimators", "590"},
    {"subsample", "0.8051026646094943"},
    {"reg_lambda", "1"},
    {"seed", "0"},
    {"min_child_weight", "85"},

    {"objective", "binary:logitraw"},
    {"max_depth", "4"},
    {"gamma", "1.4120960639617184"}
};


const std::map<const std::string, const std::string> sub43
{
    // second best score: 814643.09775 from hyper.binlogistic.f10.noFE.log

    // best score: 814643.09775
    // best params: {'colsample_bytree': 0.4156165269795621, 'scale_pos_weight': 0.3959258247559913,
    // 'min_child_weight': 55, 'n_estimators': 560, 'subsample': 0.7262267395602292, 'objective': 'binary:logistic',
    // 'max_depth': 6, 'gamma': 1.18681684221414}

    // LB: 810836.24
    // CV: 812335
    // noFE

    {"booster", "gbtree"},
    {"reg_alpha", "0"},
    {"colsample_bytree", "0.4156165269795621"},
    {"silent", "1"},
    {"colsample_bylevel", "1"},
    {"scale_pos_weight", "0.3959258247559913"},
    {"learning_rate", "0.045"},
    {"missing", "nan"},
    {"max_delta_step", "0"},
    {"base_score", "0.5"},
    {"n_estimators", "560"},
    {"subsample", "0.7262267395602292"},
    {"reg_lambda", "1"},
    {"seed", "0"},
    {"min_child_weight", "55"},

    {"objective", "binary:logistic"},
    {"max_depth", "6"},
    {"gamma", "1.18681684221414"}
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


const std::map<const std::string, const std::string> sub51
{
    // best score: 814257.42870 from hyper.reglinear.1.f10.noFE.log

    // best score: 814257.42870
    // best params: {'colsample_bytree': 0.5525129874711902,
    // 'scale_pos_weight': 0.38570090801277784, 'min_child_weight': 175, 'n_estimators': 510,
    // 'subsample': 0.7168114425646201, 'objective':'reg:linear', 'max_depth': 6, 'gamma': 1.676750571566339}

    // LB: 810762.72
    // CV: 813146
    // noFE

    {"booster", "gbtree"},
    {"reg_alpha", "0"},
    {"colsample_bytree", "0.5525129874711902"},
    {"silent", "1"},
    {"colsample_bylevel", "1"},
    {"scale_pos_weight", "0.38570090801277784"},
    {"learning_rate", "0.045"},
    {"missing", "nan"},
    {"max_delta_step", "0"},
    {"base_score", "0.5"},
    {"n_estimators", "510"},
    {"subsample", "0.7168114425646201"},
    {"reg_lambda", "1"},
    {"seed", "0"},
    {"min_child_weight", "175"},

    {"objective", "reg:linear"},
    {"max_depth", "6"},
    {"gamma", "1.676750571566339"}
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
    // {'n_estimators': 720, 'subsample': 0.6004793023546672, 'colsample_bytree': 0.4589960445147412, 'max_depth': 6, 'min_child_weight': 130}

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

    {"objective", "rank:pairwise"},
    {"max_depth", "6"},
    {"gamma", "0"}
};


const std::map<const std::string, const std::string> sub10
{
    // sub8, n_estimators=650

    // LB: 810446.95
    // CV: 809241
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
    {"n_estimators", "650"},
    {"subsample", "0.85"},
    {"reg_lambda", "1"},
    {"seed", "0"},
    {"min_child_weight", "65"},

    {"objective", "rank:pairwise"},
    {"max_depth", "7"},
    {"gamma", "0"}
};


const std::map<const std::string, const std::string> sub13
{
    // sub8, subsample=0.8

    // LB: 810079.63
    // CV: 809129
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
    {"subsample", "0.8"},
    {"reg_lambda", "1"},
    {"seed", "0"},
    {"min_child_weight", "65"},

    {"objective", "rank:pairwise"},
    {"max_depth", "7"},
    {"gamma", "0"}
};


const std::map<const std::string, const std::string> sub52
{
    // second best score: 814251.03395 from hyper.reglinear.1.f10.noFE.log

    // best score: 814251.03395
    // best params: {'colsample_bytree': 0.6683493524938185,
    // 'scale_pos_weight': 0.3821397591786267, 'min_child_weight': 215, 'n_estimators': 560,
    // 'subsample': 0.9535898735794333, 'objective':'reg:linear', 'max_depth': 8, 'gamma': 1.6943574047322956}

    // LB: 809993.65
    // CV: 814367
    // noFE

    {"booster", "gbtree"},
    {"reg_alpha", "0"},
    {"colsample_bytree", "0.6683493524938185"},
    {"silent", "1"},
    {"colsample_bylevel", "1"},
    {"scale_pos_weight", "0.3821397591786267"},
    {"learning_rate", "0.045"},
    {"missing", "nan"},
    {"max_delta_step", "0"},
    {"base_score", "0.5"},
    {"n_estimators", "560"},
    {"subsample", "0.9535898735794333"},
    {"reg_lambda", "1"},
    {"seed", "0"},
    {"min_child_weight", "215"},

    {"objective", "reg:linear"},
    {"max_depth", "8"},
    {"gamma", "1.6943574047322956"}
};


const std::map<const std::string, const std::string> sub47
{
    // third best score: 814591.89169 from hyper.binlogitraw.2.f10.noFE.log

    // best score: 814591.89169
    // best params: {'colsample_bytree': 0.9705978190927813,
    // 'scale_pos_weight': 0.6789520700647581, 'min_child_weight': 95, 'n_estimators': 540,
    // 'subsample': 0.8708002808590953, 'objective': 'binary:logitraw', 'max_depth': 4, 'gamma': 1.617042882325554}

    // LB: 809960.89
    // CV: 811809
    // noFE

    {"booster", "gbtree"},
    {"reg_alpha", "0"},
    {"colsample_bytree", "0.9705978190927813"},
    {"silent", "1"},
    {"colsample_bylevel", "1"},
    {"scale_pos_weight", "0.6789520700647581"},
    {"learning_rate", "0.045"},
    {"missing", "nan"},
    {"max_delta_step", "0"},
    {"base_score", "0.5"},
    {"n_estimators", "540"},
    {"subsample", "0.8708002808590953"},
    {"reg_lambda", "1"},
    {"seed", "0"},
    {"min_child_weight", "95"},

    {"objective", "binary:logitraw"},
    {"max_depth", "4"},
    {"gamma", "1.617042882325554"}
};


const std::map<const std::string, const std::string> sub44
{
    // 3rd best score: 814427.55776 from hyper.binlogistic.f10.noFE.log

    // best score: 814427.55776
    // best params: {'colsample_bytree': 0.7164099207357719, 'scale_pos_weight': 0.3948607027148843,
    // 'min_child_weight': 70, 'n_estimators': 530, 'subsample': 0.7963089255980069, 'objective': 'binary:logistic',
    // 'max_depth': 4, 'gamma': 1.443989391942047}

    // LB: 809837.72
    // CV: 813546
    // noFE

    {"booster", "gbtree"},
    {"reg_alpha", "0"},
    {"colsample_bytree", "0.7164099207357719"},
    {"silent", "1"},
    {"colsample_bylevel", "1"},
    {"scale_pos_weight", "0.3948607027148843"},
    {"learning_rate", "0.045"},
    {"missing", "nan"},
    {"max_delta_step", "0"},
    {"base_score", "0.5"},
    {"n_estimators", "530"},
    {"subsample", "0.7963089255980069"},
    {"reg_lambda", "1"},
    {"seed", "0"},
    {"min_child_weight", "70"},

    {"objective", "binary:logistic"},
    {"max_depth", "4"},
    {"gamma", "1.443989391942047"}
};


const std::map<const std::string, const std::string> sub37
{
    // sub24 + num_pairsample=2

    // LB: 809772.75
    // CV: 812317
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

    {"num_pairsample", "4"},

    {"objective", "rank:pairwise"},
    {"max_depth", "6"},
    {"gamma", "0"}
};


const std::map<const std::string, const std::string> sub45
{
    // best score: 814984.82934 from hyper.binlogitraw.2.f10.noFE.log

    // best score: 814984.82934
    // best params: {'colsample_bytree': 0.7640052598785774,
    // 'scale_pos_weight': 0.7135239801678219, 'min_child_weight': 55, 'n_estimators': 760,
    // 'subsample': 0.5452538000685805, 'objective': 'binary:logitraw', 'max_depth': 5,
    // 'gamma': 0.9751037278636211}

    // LB: 809525.06
    // CV: 810875
    // noFE

    {"booster", "gbtree"},
    {"reg_alpha", "0"},
    {"colsample_bytree", "0.7640052598785774"},
    {"silent", "1"},
    {"colsample_bylevel", "1"},
    {"scale_pos_weight", "0.7135239801678219"},
    {"learning_rate", "0.045"},
    {"missing", "nan"},
    {"max_delta_step", "0"},
    {"base_score", "0.5"},
    {"n_estimators", "760"},
    {"subsample", "0.5452538000685805"},
    {"reg_lambda", "1"},
    {"seed", "0"},
    {"min_child_weight", "55"},

    {"objective", "binary:logitraw"},
    {"max_depth", "5"},
    {"gamma", "0.9751037278636211"}
};


const std::map<const std::string, const std::string> sub9
{
    // sub8, n_estimators=700

    // LB: 809303.95
    // CV: 807533
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
    {"n_estimators", "700"},
    {"subsample", "0.85"},
    {"reg_lambda", "1"},
    {"seed", "0"},
    {"min_child_weight", "65"},

    {"objective", "rank:pairwise"},
    {"max_depth", "7"},
    {"gamma", "0"}
};


const std::map<const std::string, const std::string> sub11
{
    // sub8, n_estimators=550

    // LB: 809167.39
    // CV: 811629
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
    {"n_estimators", "550"},
    {"subsample", "0.85"},
    {"reg_lambda", "1"},
    {"seed", "0"},
    {"min_child_weight", "65"},

    {"objective", "rank:pairwise"},
    {"max_depth", "7"},
    {"gamma", "0"}
};


const std::map<const std::string, const std::string> sub53
{
    // third best score: 814088.31464 from hyper.reglinear.1.f10.noFE.log

    // best score: 814088.31464
    // best params: {'colsample_bytree': 0.5774206979959366,
    // 'scale_pos_weight': 0.3837579582606276, 'min_child_weight': 215, 'n_estimators': 570,
    // 'subsample': 0.8588310863551543, 'objective':'reg:linear', 'max_depth': 7, 'gamma': 1.1270963579351068}

    // LB: 808078.83
    // CV: 814576
    // noFE

    {"booster", "gbtree"},
    {"reg_alpha", "0"},
    {"colsample_bytree", "0.5774206979959366"},
    {"silent", "1"},
    {"colsample_bylevel", "1"},
    {"scale_pos_weight", "0.3837579582606276"},
    {"learning_rate", "0.045"},
    {"missing", "nan"},
    {"max_delta_step", "0"},
    {"base_score", "0.5"},
    {"n_estimators", "570"},
    {"subsample", "0.8588310863551543"},
    {"reg_lambda", "1"},
    {"seed", "0"},
    {"min_child_weight", "215"},

    {"objective", "reg:linear"},
    {"max_depth", "7"},
    {"gamma", "1.1270963579351068"}
};


const std::map<const std::string, const std::string> sub12
{
    // sub8 + subsample=0.9,colsample_by_tree=0.67

    // LB: 806501.60
    // CV: 810355
    // no FE

    {"booster", "gbtree"},
    {"reg_alpha", "0"},
    {"colsample_bytree", "0.67"},
    {"silent", "1"},
    {"colsample_bylevel", "1"},
    {"scale_pos_weight", "1"},
    {"learning_rate", "0.045"},
    {"missing", "nan"},
    {"max_delta_step", "0"},
    {"base_score", "0.5"},
    {"n_estimators", "600"},
    {"subsample", "0.9"},
    {"reg_lambda", "1"},
    {"seed", "0"},
    {"min_child_weight", "65"},

    {"objective", "rank:pairwise"},
    {"max_depth", "7"},
    {"gamma", "0"}
};


const std::map<const std::string, const std::string> sub41
{
    // from hyper.rank.f20.noFE.log, second best scored entry at 814001.18640

    // best params: {'n_estimators': 530, 'subsample': 0.6410812525524375, 'colsample_bytree': 0.5895782044640605,
    // 'max_depth': 7, 'min_child_weight': 135}
    // CV: 812609
    // ~~CV: 810261(600 est)

    // LB: 806384.27
    // CV: 812609
    // no FE

    {"booster", "gbtree"},
    {"reg_alpha", "0"},
    {"colsample_bytree", "0.5895782044640605"},
    {"silent", "1"},
    {"colsample_bylevel", "1"},
    {"scale_pos_weight", "0.3948607027148843"},
    {"learning_rate", "0.045"},
    {"missing", "nan"},
    {"max_delta_step", "0"},
    {"base_score", "0.5"},
    {"n_estimators", "530"},
    {"subsample", "0.6410812525524375"},
    {"reg_lambda", "1"},
    {"seed", "0"},
    {"min_child_weight", "135"},

    {"objective", "rank:pairwise"},
    {"max_depth", "7"},
};


const std::map<const std::string, const std::string> sub25
{
    // hyperopt.4.sorted.log, best ranked
    // best score: 813859.62624
    // best params: {'n_estimators': 650, 'subsample': 0.8029650963196207,
    // 'colsample_bytree': 0.436229760798959, 'max_depth': 7, 'min_child_weight': 100}


    // LB: 806383.80
    // CV: 813336
    // no FE

    {"booster", "gbtree"},
    {"reg_alpha", "0"},
    {"colsample_bytree", "0.4362"},
    {"silent", "1"},
    {"colsample_bylevel", "1"},
    {"scale_pos_weight", "1"},
    {"learning_rate", "0.045"},
    {"missing", "nan"},
    {"max_delta_step", "0"},
    {"base_score", "0.5"},
    {"n_estimators", "650"},
    {"subsample", "0.8030"},
    {"reg_lambda", "1"},
    {"seed", "0"},
    {"min_child_weight", "100"},

    {"objective", "rank:pairwise"},
    {"max_depth", "7"},
    {"gamma", "0"}
};


const std::map<const std::string, const std::string> sub3
{
    // sub8, n_estimators=500

    // LB: 805800.36
    // CV: 812613
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
    {"n_estimators", "500"},
    {"subsample", "0.85"},
    {"reg_lambda", "1"},
    {"seed", "0"},
    {"min_child_weight", "65"},

    {"objective", "rank:pairwise"},
    {"max_depth", "7"},
    {"gamma", "0"}
};


const std::map<const std::string, const std::string> sub42
{
    // from hyper.rank.f20.noFE.log, second best scored entry at 815135.60524

    // best score: 815135.60524
    // best params: {'n_estimators': 510, 'subsample': 0.6399543679962626, 'colsample_bytree': 0.6612120010371612,
    // 'max_depth': 7, 'min_child_weight': 135}

    // LB: 805399.38
    // CV: 811628
    // no FE

    {"booster", "gbtree"},
    {"reg_alpha", "0"},
    {"colsample_bytree", "0.6612120010371612"},
    {"silent", "1"},
    {"colsample_bylevel", "1"},
    {"scale_pos_weight", "0.3948607027148843"},
    {"learning_rate", "0.045"},
    {"missing", "nan"},
    {"max_delta_step", "0"},
    {"base_score", "0.5"},
    {"n_estimators", "510"},
    {"subsample", "0.6399543679962626"},
    {"reg_lambda", "1"},
    {"seed", "0"},
    {"min_child_weight", "135"},

    {"objective", "rank:pairwise"},
    {"max_depth", "7"},
};


const std::map<const std::string, const std::string> sub7
{
    // sub3, MISSING=-1

    // LB: 804689.03
    // CV: .
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
    {"n_estimators", "500"},
    {"subsample", "0.85"},
    {"reg_lambda", "1"},
    {"seed", "0"},
    {"min_child_weight", "65"},

    {"objective", "rank:pairwise"},
    {"max_depth", "7"},
    {"gamma", "0"}
};









const std::map<const std::string, const std::string> sub35
{
    // sub24 + num_pairsample=6

    // LB: 801047.70
    // CV: 814476
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

    {"num_pairsample", "6"},

    {"objective", "rank:pairwise"},
    {"max_depth", "6"},
    {"gamma", "0"}
};


const std::map<const std::string, const std::string> sub36
{
    // sub24 + num_pairsample=4

    // LB: 802622.24
    // CV: 812317
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

    {"num_pairsample", "4"},

    {"objective", "rank:pairwise"},
    {"max_depth", "6"},
    {"gamma", "0"}
};


const std::map<const std::string, const std::string> sub38
{
    // sub8 + num_pairsample=2

    // LB: 801095.25
    // CV: 813410
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

    {"num_pairsample", "2"},

    {"objective", "rank:pairwise"},
    {"max_depth", "7"},
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


const std::map<const std::string, const std::string> sub6
{
    // sub3, top-2 FE

    // LB: 805800.36
    // CV: .
    // top-2 FE

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
    {"n_estimators", "500"},
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

const std::map<const std::string, const std::string> sub30
{
    // from hyper.rank.f20.noFE.log, best entry scored at 815135.60524 with 510 estimators
    // {'n_estimators': 510, 'subsample': 0.6399543679962626, 'colsample_bytree': 0.6612120010371612,
    //  'max_depth': 7, 'min_child_weight': 135}

    // LB: 809041.44
    // CV: .
    // FE full

    {"booster", "gbtree"},
    {"reg_alpha", "0"},
    {"colsample_bytree", "0.6612"},
    {"silent", "1"},
    {"colsample_bylevel", "1"},
    {"scale_pos_weight", "1"},
    {"learning_rate", "0.045"},
    {"missing", "nan"},
    {"max_delta_step", "0"},
    {"base_score", "0.5"},
    {"n_estimators", "600"},
    {"subsample", "0.6400"},
    {"reg_lambda", "1"},
    {"seed", "0"},
    {"min_child_weight", "135"},

    {"objective", "rank:pairwise"},
    {"max_depth", "7"},
    {"gamma", "0.0000"}
};


const std::map<const std::string, const std::string> sub23
{
    // from hyperopt.2.sorted.log, best entry scored at 814158.03998 with 570 estimators, here at 725
    // {'n_estimators': 570, 'subsample': 0.8546865430244482, 'colsample_bytree': 0.562540029980502, 'min_child_weight': 90}

    // LB: 808315.02
    // CV: .
    // FE full

    {"booster", "gbtree"},
    {"reg_alpha", "0"},
    {"colsample_bytree", "0.5625"},
    {"silent", "1"},
    {"colsample_bylevel", "1"},
    {"scale_pos_weight", "1"},
    {"learning_rate", "0.045"},
    {"missing", "nan"},
    {"max_delta_step", "0"},
    {"base_score", "0.5"},
    {"n_estimators", "725"},
    {"subsample", "0.8547"},
    {"reg_lambda", "1"},
    {"seed", "0"},
    {"min_child_weight", "90"},

    {"objective", "rank:pairwise"},
    {"max_depth", "7"},
    {"gamma", "0"}
};


const std::map<const std::string, const std::string> sub20
{
    // from hyperopt.2.sorted.log, best entry scored at 814158.03998 with 570 estimators, here at 650
    // {'n_estimators': 570, 'subsample': 0.8546865430244482, 'colsample_bytree': 0.562540029980502, 'min_child_weight': 90}

    // LB: 808065.26
    // CV: .
    // FE full

    {"booster", "gbtree"},
    {"reg_alpha", "0"},
    {"colsample_bytree", "0.5625"},
    {"silent", "1"},
    {"colsample_bylevel", "1"},
    {"scale_pos_weight", "1"},
    {"learning_rate", "0.045"},
    {"missing", "nan"},
    {"max_delta_step", "0"},
    {"base_score", "0.5"},
    {"n_estimators", "650"},
    {"subsample", "0.8547"},
    {"reg_lambda", "1"},
    {"seed", "0"},
    {"min_child_weight", "90"},

    {"objective", "rank:pairwise"},
    {"max_depth", "7"},
    {"gamma", "0"}
};


const std::map<const std::string, const std::string> sub14
{
    // sub8, plus full FE

    // LB: 808047.62
    // CV: 811931
    // FE full

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


const std::map<const std::string, const std::string> sub17
{
    // LB: 807893.78
    // CV: .
    // FE full

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
    {"n_estimators", "750"},
    {"subsample", "0.85"},
    {"reg_lambda", "1"},
    {"seed", "0"},
    {"min_child_weight", "60"},

    {"objective", "rank:pairwise"},
    {"max_depth", "6"},
    {"gamma", "0"}
};


const std::map<const std::string, const std::string> sub16
{
    // sub17 n_estimators=700

    // LB: 807723.33
    // CV: .
    // FE full

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
    {"n_estimators", "700"},
    {"subsample", "0.85"},
    {"reg_lambda", "1"},
    {"seed", "0"},
    {"min_child_weight", "60"},

    {"objective", "rank:pairwise"},
    {"max_depth", "6"},
    {"gamma", "0"}
};


const std::map<const std::string, const std::string> sub21
{
    // hyperopt.2.sorted.log, second best entry, done at nest=600

    // best score: 814138.30899
    // best params: {'n_estimators': 520, 'subsample': 0.8850442710992165,
    // 'colsample_bytree': 0.5598635956943869, 'min_child_weight': 70}


    // LB: 807714.72
    // CV: .
    // FE full

    {"booster", "gbtree"},
    {"reg_alpha", "0"},
    {"colsample_bytree", "0.5599"},
    {"silent", "1"},
    {"colsample_bylevel", "1"},
    {"scale_pos_weight", "1"},
    {"learning_rate", "0.045"},
    {"missing", "nan"},
    {"max_delta_step", "0"},
    {"base_score", "0.5"},
    {"n_estimators", "600"},
    {"subsample", "0.8850"},
    {"reg_lambda", "1"},
    {"seed", "0"},
    {"min_child_weight", "70"},

    {"objective", "rank:pairwise"},
    {"max_depth", "7"},
    {"gamma", "0"}
};


const std::map<const std::string, const std::string> sub28
{
    // best ranked from hyperopt.rank.pairwise.1.log

    // best score: 814524.23995
    // best params: {'colsample_bytree': 0.5486886032198215, 'min_child_weight': 150,
    // 'n_estimators': 670, 'subsample': 0.8171114174908516, 'max_depth': 8, 'gamma': 0.1652567602745394}


    // LB: 803964.73
    // CV: .
    // FE full

    {"booster", "gbtree"},
    {"reg_alpha", "0"},
    {"colsample_bytree", "0.5487"},
    {"silent", "1"},
    {"colsample_bylevel", "1"},
    {"scale_pos_weight", "1"},
    {"learning_rate", "0.045"},
    {"missing", "nan"},
    {"max_delta_step", "0"},
    {"base_score", "0.5"},
    {"n_estimators", "670"},
    {"subsample", "0.8171"},
    {"reg_lambda", "1"},
    {"seed", "0"},
    {"min_child_weight", "150"},

    {"objective", "rank:pairwise"},
    {"max_depth", "8"},
    {"gamma", "0.1653"}
};


const std::map<const std::string, const std::string> sub22
{
    // from hyperopt.2.sorted.log, best entry scored at 814158.03998 with 570 estimators
    // {'n_estimators': 570, 'subsample': 0.8546865430244482, 'colsample_bytree': 0.562540029980502, 'min_child_weight': 90}

    // LB: 808315.02
    // CV: .
    // FE full

    {"booster", "gbtree"},
    {"reg_alpha", "0"},
    {"colsample_bytree", "0.5625"},
    {"silent", "1"},
    {"colsample_bylevel", "1"},
    {"scale_pos_weight", "1"},
    {"learning_rate", "0.045"},
    {"missing", "nan"},
    {"max_delta_step", "0"},
    {"base_score", "0.5"},
    {"n_estimators", "570"},
    {"subsample", "0.8547"},
    {"reg_lambda", "1"},
    {"seed", "0"},
    {"min_child_weight", "90"},

    {"objective", "rank:pairwise"},
    {"max_depth", "7"},
    {"gamma", "0"}
};


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

const std::map<const std::string, const std::string> sub31
{
    // from logitraw.scaleposweight.f5.log, best entry scored at 812319.95208
    // {'colsample_bytree': 0.9053682410507754, 'scale_pos_weight': 0.674004598499349,
    //  'min_child_weight': 120, 'n_estimators': 700, 'subsample': 0.903602973136531,
    //  'objective': 'binary:logitraw', 'max_depth': 4, 'gamma': 0.5390140055211626}

    // LB: 808664.36
    // CV: .
    // FE full

    {"booster", "gbtree"},
    {"reg_alpha", "0"},
    {"colsample_bytree", "0.9054"},
    {"silent", "1"},
    {"colsample_bylevel", "1"},
    {"scale_pos_weight", "0.6740"},
    {"learning_rate", "0.045"},
    {"missing", "nan"},
    {"max_delta_step", "0"},
    {"base_score", "0.5"},
    {"n_estimators", "700"},
    {"subsample", "0.9036"},
    {"reg_lambda", "1"},
    {"seed", "0"},
    {"min_child_weight", "120"},

    {"objective", "binary:logitraw"},
    {"max_depth", "4"},
    {"gamma", "0.5390"}
};


const std::map<const std::string, const std::string> sub27
{
    // LB: 806684.54
    // CV: .
    // FE full

    {"booster", "gblinear"}, // !!!!!!!!!!!!!!!!!!!!!!!!!!! pomylka? 0:46 w nocy
//        {"booster", "gbtree"}, // default
    {"reg_alpha", "0"},
    {"colsample_bytree", "0.65"},
    {"silent", "1"},
    {"colsample_bylevel", "1"},
    {"scale_pos_weight", "0.7"},
    {"learning_rate", "0.045"},
    {"missing", "nan"},
    {"max_delta_step", "0"},
    {"base_score", "0.5"},
    {"n_estimators", "600"},
    {"subsample", "0.85"},
    {"reg_lambda", "1"},
    {"seed", "0"},
    {"min_child_weight", "65"},

    {"objective", "binary:logitraw"},
    {"max_depth", "7"},
    {"gamma", "0"}
};


const std::map<const std::string, const std::string> sub15
{
    // sub17 n_estimators=600

    // LB: 801164.08
    // CV: .
    // FE full

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
    {"min_child_weight", "60"},

    {"objective", "rank:pairwise"},
    {"max_depth", "6"},
    {"gamma", "0"}
};


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

}  // namespace params


#endif /* PARAM_STORE_HPP_ */
