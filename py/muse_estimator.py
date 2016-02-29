#!/opt/anaconda2/bin/python
# -*- coding: utf-8 -*-

"""
################################################################################
#
#  Copyright (c) 2016 Wojciech Migda
#  All rights reserved
#  Distributed under the terms of the MIT license
#
################################################################################
#
#  Filename: muse_estimator.py
#
#  Decription:
#      Muse Millenials estimator
#
#  Authors:
#       Wojciech Migda
#
################################################################################
#
#  History:
#  --------
#  Date         Who  Ticket     Description
#  ----------   ---  ---------  ------------------------------------------------
#  2016-02-23   wm              Initial version
#
################################################################################
"""

from __future__ import print_function


DEBUG = False

__all__ = []
__version__ = "0.0.1"
__date__ = '2016-02-23'
__updated__ = '2016-02-23'


FACTORIZABLE = [
    'GENDER',
    'REGISTRATION_ROUTE',
    'REGISTRATION_CONTEXT',
    'OPTIN',
    'IS_DELETED',
    'MIGRATED_USER_TYPE',
    'SOCIAL_AUTH_FACEBOOK',
    'SOCIAL_AUTH_TWITTER',
    'SOCIAL_AUTH_GOOGLE'
]


def OneHot(df, colnames):
    from pandas import get_dummies, concat
    for col in colnames:
        dummies = get_dummies(df[col])
        #ndumcols = dummies.shape[1]
        dummies.rename(columns={p: col + '_' + str(i + 1)  for i, p in enumerate(dummies.columns.values)}, inplace=True)
        df = concat([df, dummies], axis=1)
        pass
    df = df.drop(colnames, axis=1)
    return df


from sklearn.base import BaseEstimator, RegressorMixin


NOMINALS = ['GENDER', 'REGISTRATION_ROUTE', 'REGISTRATION_CONTEXT',
            'MIGRATED_USER_TYPE', 'PLATFORM_CENTRE', 'TOD_CENTRE',
            'CONTENT_CENTRE']

def work(out_csv_file,
         estimator,
         nest,
         njobs,
         nfolds,
         cv_grid,
         minimizer,
         nbuckets,
         mvector,
         imputer,
         clf_kwargs,
         int_fold):

    from numpy.random import seed as random_seed
    random_seed(1)

    from pandas import read_csv

    all_data = read_csv("../../data/demographic_membership_training.csv")

    train_y = all_data['DEMO_X'].values
    train_X = all_data.drop(['CONSUMER_ID', 'DEMO_X'], axis=1)

    from pandas import factorize
    train_X['GENDER'][train_X['GENDER'] == 'U'] = float('nan')

    for col in FACTORIZABLE:
        from pandas import isnull

        missing = isnull(train_X[col])

        train_X[col] = factorize(train_X[col])[0] ## NANs become -1
        train_X[col][missing] = float('nan')

        from numpy import isnan
        print("NANs after factorization", sum(train_X[col].apply(isnan)))
        pass

    train_X = OneHot(train_X, NOMINALS)

#    train_X['**PROP_PAGE_IMPRESSIONS_DWELL'] = train_X['PAGE_IMPRESSIONS_DWELL'] / train_X['TOTAL_DWELL']
#    train_X['**PROP_VOD_VIEWS_DWELL'] = train_X['VOD_VIEWS_DWELL'] / train_X['TOTAL_DWELL']
#
#    train_X['**FLAG_WARD_WKDAY_COUNT'] = train_X[train_X.columns[train_X.columns.str.startswith('FLAG_WARD_WKDAY_')]].sum(axis=1)
#    train_X['**FLAG_WARD_WKEND_COUNT'] = train_X[train_X.columns[train_X.columns.str.startswith('FLAG_WARD_WKEND_')]].sum(axis=1)
#    train_X['**FLAG_UNI_CLUSTER_COUNT'] = train_X[train_X.columns[train_X.columns.str.startswith('FLAG_UNI_CLUSTER_')]].sum(axis=1)
#    train_X['**INTERESTS_COUNT'] = train_X[train_X.columns[train_X.columns.str.startswith('INTEREST_')]].sum(axis=1)
#    train_X['**AGE_25'] = train_X['AGE'] < 25
#    train_X['**AGE_30'] = train_X['AGE'] < 30
#    train_X['**AGE_35'] = train_X['AGE'] < 35
#    train_X['**AGE_40'] = train_X['AGE'] < 40
#    train_X['**AGE_45'] = train_X['AGE'] < 45
#    train_X['**PAGE_IMP_DWELL_PER_DAY'] = train_X['PAGE_IMPRESSIONS_DWELL'] / train_X['REGISTRATION_DAYS']
#    train_X['**LATE_PAGE_VIEWS_PER_DAY'] = train_X['LATE_PAGE_VIEWS'] / train_X['REGISTRATION_DAYS']
#    train_X['**TOTAL_DWELL_PER_DAY'] = train_X['TOTAL_DWELL'] / train_X['REGISTRATION_DAYS']
#    train_X['**AFTERNOON_PAGE_VIEWS_PER_DAY'] = train_X['AFTERNOON_PAGE_VIEWS'] / train_X['REGISTRATION_DAYS']
#    train_X['**PAGE_IMPRESSION_VISITS_PER_DAY'] = train_X['PAGE_IMPRESSION_VISITS'] / train_X['REGISTRATION_DAYS']
#    train_X['**LUNCHTIME_PAGE_VIEWS_PER_DAY'] = train_X['LUNCHTIME_PAGE_VIEWS'] / train_X['REGISTRATION_DAYS']
#    train_X['**NIGHT_TIME_PAGE_VIEWS_PER_DAY'] = train_X['NIGHT_TIME_PAGE_VIEWS'] / train_X['REGISTRATION_DAYS']
#    train_X['**BREAKFAST_PAGE_VIEWS_PER_DAY'] = train_X['BREAKFAST_PAGE_VIEWS'] / train_X['REGISTRATION_DAYS']
#    train_X['**VIDEO_STOPS_PER_DAY'] = train_X['VIDEO_STOPS'] / train_X['REGISTRATION_DAYS']

#    TO_DROP = [
#        'VIEWS_AFF4', 'FLAG_WARD_WKDAY_10_16', 'FLAG_WARD_WKDAY_17_19',
#        'FLAG_WARD_WKDAY_20_24', 'FLAG_WARD_WKEND_10_13', 'FLAG_WARD_WKEND_14_20',
#        'FLAG_UNI_CLUSTER_15', 'FLAG_UNI_CLUSTER_23', 'FLAG_UNI_CLUSTER_28',
#        'FLAG_UNI_CLUSTER_29', 'FLAG_UNI_CLUSTER_33', 'FLAG_WEBSITE',
#        'FLAG_BREAKFAST_VIEWS', 'FLAG_LUNCHTIME_VIEWS', 'FLAG_AFTERNOON_VIEWS',
#        'FLAG_CATCHUP_VIEWS', 'FLAG_ARCHIVE_VIEWS', 'FLAG_AFF3', 'FLAG_AFF4',
#        'REGISTRATION_ROUTE_3', 'REGISTRATION_ROUTE_4', 'REGISTRATION_CONTEXT_3',
#        'REGISTRATION_CONTEXT_6', 'REGISTRATION_CONTEXT_8', 'REGISTRATION_CONTEXT_9',
#        'REGISTRATION_CONTEXT_10', 'REGISTRATION_CONTEXT_11', 'REGISTRATION_CONTEXT_12',
#        'REGISTRATION_CONTEXT_13', 'REGISTRATION_CONTEXT_14', 'REGISTRATION_CONTEXT_15',
#        'REGISTRATION_CONTEXT_16', 'REGISTRATION_CONTEXT_17', 'REGISTRATION_CONTEXT_18',
#        'REGISTRATION_CONTEXT_19', 'REGISTRATION_CONTEXT_20', 'REGISTRATION_CONTEXT_21',
#        'REGISTRATION_CONTEXT_22', 'REGISTRATION_CONTEXT_23', 'REGISTRATION_CONTEXT_24',
#        'REGISTRATION_CONTEXT_25', 'REGISTRATION_CONTEXT_26', 'REGISTRATION_CONTEXT_27',
#        'MIGRATED_USER_TYPE_5', 'TOD_CENTRE_3', 'CONTENT_CENTRE_1', 'CONTENT_CENTRE_2',
#        'CONTENT_CENTRE_4', 'CONTENT_CENTRE_5', 'CONTENT_CENTRE_6', 'CONTENT_CENTRE_7',
#        'CONTENT_CENTRE_8', 'CONTENT_CENTRE_9', 'CONTENT_CENTRE_12','CONTENT_CENTRE_13',
#        'CONTENT_CENTRE_15']
#    TO_DROP += [
#        'SOCIAL_AUTH_TWITTER', 'FLAG_WARD_WKEND_3_9', 'FLAG_UNI_CLUSTER_7',
#        'FLAG_UNI_CLUSTER_13', 'FLAG_UNI_CLUSTER_21', 'FLAG_UNI_CLUSTER_22',
#        'FLAG_UNI_CLUSTER_25', 'FLAG_ANDROID', 'FLAG_LATE_PEAK_VIEWS',
#        'FLAG_NIGHT_TIME_VIEWS', 'FLAG_AFF1', 'FLAG_AFF2', 'MIGRATED_USER_TYPE_4',
#        'CONTENT_CENTRE_10', 'CONTENT_CENTRE_14', 'CONTENT_CENTRE_16']
#    TO_DROP += [
#        'FLAG_WARD_WKDAY_3_9', 'FLAG_UNI_CLUSTER_5', 'FLAG_UNI_CLUSTER_8',
#        'FLAG_UNI_CLUSTER_9', 'FLAG_UNI_CLUSTER_17', 'FLAG_UNI_CLUSTER_26',
#        'FLAG_MORNING_VIEWS', 'FLAG_EARLY_PEAK_VIEWS']
#    TO_DROP += [
#        'FLAG_WARD_WKEND_1_2', 'FLAG_WARD_WKEND_21_24', 'FLAG_UNI_CLUSTER_1',
#        'FLAG_UNI_CLUSTER_14', 'FLAG_MAIN', 'FLAG_OTHER_VIEWS', 'CONTENT_CENTRE_11']
#    TO_DROP += ['FLAG_UNI_CLUSTER_12', 'FLAG_UNI_CLUSTER_19', 'FLAG_UNI_CLUSTER_27']
#    TO_DROP += ['FLAG_UNI_CLUSTER_11', 'FLAG_UNI_CLUSTER_24'] ## 814000 ?
#    TO_DROP += ['FLAG_UNI_CLUSTER_2', 'FLAG_UNI_CLUSTER_16']
#    TO_DROP += ['FLAG_UNI_CLUSTER_10', 'FLAG_UNI_CLUSTER_31', 'FLAG_POST_PEAK_VIEWS', 'TOD_CENTRE_2']
#    TO_DROP += ['FLAG_UNI_CLUSTER_30']
#    train_X = train_X.drop(TO_DROP, axis=1)

#    train_X.fillna(-1, inplace=True)

    from sklearn.cross_validation import StratifiedKFold
    from sklearn.grid_search import GridSearchCV

    if False:
        skf = StratifiedKFold(train_y, n_folds=nfolds)
        from numpy import asarray
        selection = asarray(['-'] * len(train_y))
        symbol = 0
        for train_index, test_index in skf:
            selection[test_index] =chr(symbol + 48)
            symbol += 1
            pass
        print(''.join(selection))
        return

    muse_kwargs = \
    {
        #'objective': 'reg:logistic',
        'objective': 'rank:pairwise',
        'learning_rate': 0.045,
        'min_child_weight': 50,
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        'max_depth': 7,
        'n_estimators': nest,
        'nthread': njobs,
        'seed': 0,
        'missing': float('nan')
        #'scoring': NegQWKappaScorer
    }

    # override kwargs with any changes
    for k, v in clf_kwargs.items():
        muse_kwargs[k] = v
        pass
    #clf = globals()[estimator](**muse_kwargs)
    from xgboost import XGBClassifier
    clf = XGBClassifier(**muse_kwargs)

    from sklearn.metrics import make_scorer
    def TcoScorer(y_true, y_pred):
        from sklearn.metrics import precision_score, recall_score
        P = precision_score(y_true, y_pred)
        R = recall_score(y_true, y_pred)
        score = 1000000 * min(P, R)

        return score
    tco_scorer = make_scorer(TcoScorer)

    """
binary:logistic
    grid scores:
  mean: 787812.76918, std: 1297.55109, params: {'n_estimators': 500, 'subsample': 0.9, 'colsample_bytree': 0.67, 'max_depth': 7, 'min_child_weight': 1}
  mean: 789084.73195, std: 1925.75110, params: {'n_estimators': 500, 'subsample': 0.9, 'colsample_bytree': 0.67, 'max_depth': 7, 'min_child_weight': 3}
  mean: 789651.63043, std: 1855.11841, params: {'n_estimators': 500, 'subsample': 0.9, 'colsample_bytree': 0.67, 'max_depth': 7, 'min_child_weight': 10}
  mean: 789958.10747, std: 1305.40202, params: {'n_estimators': 500, 'subsample': 0.9, 'colsample_bytree': 0.67, 'max_depth': 7, 'min_child_weight': 20}
  mean: 788739.11423, std: 952.60469, params: {'n_estimators': 500, 'subsample': 0.9, 'colsample_bytree': 0.67, 'max_depth': 7, 'min_child_weight': 50}
  mean: 788168.38281, std: 928.87371, params: {'n_estimators': 500, 'subsample': 0.9, 'colsample_bytree': 0.67, 'max_depth': 7, 'min_child_weight': 80}
best score: 789958.10747
best params: {'n_estimators': 500, 'subsample': 0.9, 'colsample_bytree': 0.67, 'max_depth': 7, 'min_child_weight': 20}

reg:logistic (to samo co wyżej)
grid scores:
  mean: 789651.63043, std: 1855.11841, params: {'n_estimators': 500, 'subsample': 0.9, 'colsample_bytree': 0.67, 'max_depth': 7, 'min_child_weight': 10}
  mean: 789958.10747, std: 1305.40202, params: {'n_estimators': 500, 'subsample': 0.9, 'colsample_bytree': 0.67, 'max_depth': 7, 'min_child_weight': 20}
  mean: 788739.11423, std: 952.60469, params: {'n_estimators': 500, 'subsample': 0.9, 'colsample_bytree': 0.67, 'max_depth': 7, 'min_child_weight': 50}
best score: 789958.10747
best params: {'n_estimators': 500, 'subsample': 0.9, 'colsample_bytree': 0.67, 'max_depth': 7, 'min_child_weight': 20}


grid scores:
  mean: 786388.90860, std: 906.72660, params: {'colsample_bytree': 0.67, 'min_child_weight': 65, 'n_estimators': 300, 'subsample': 0.9, 'objective': 'binary:logistic', 'max_depth': 7}
  mean: 789050.88848, std: 1708.63378, params: {'colsample_bytree': 0.67, 'min_child_weight': 65, 'n_estimators': 500, 'subsample': 0.9, 'objective': 'binary:logistic', 'max_depth': 7}
  mean: 789454.57872, std: 2059.68811, params: {'colsample_bytree': 0.67, 'min_child_weight': 65, 'n_estimators': 700, 'subsample': 0.9, 'objective': 'binary:logistic', 'max_depth': 7}
best score: 789454.57872
best params: {'colsample_bytree': 0.67, 'min_child_weight': 65, 'n_estimators': 700, 'subsample': 0.9, 'objective': 'binary:logistic', 'max_depth': 7}

======================================================
rank:pairwise
grid scores:
  mean: 806358.37855, std: 4488.86812, params: {'n_estimators': 500, 'subsample': 0.9, 'colsample_bytree': 0.67, 'max_depth': 7, 'min_child_weight': 20}
best score: 806358.37855
best params: {'n_estimators': 500, 'subsample': 0.9, 'colsample_bytree': 0.67, 'max_depth': 7, 'min_child_weight': 20}

grid scores:
  mean: 750119.43597, std: 9120.06057, params: {'n_estimators': 500, 'subsample': 0.9, 'colsample_bytree': 0.67, 'max_depth': 4, 'min_child_weight': 20}
  mean: 809673.54959, std: 4784.35577, params: {'n_estimators': 500, 'subsample': 0.9, 'colsample_bytree': 0.67, 'max_depth': 6, 'min_child_weight': 20}
  mean: 798151.02989, std: 2162.04583, params: {'n_estimators': 500, 'subsample': 0.9, 'colsample_bytree': 0.67, 'max_depth': 10, 'min_child_weight': 20}
  mean: 794998.50356, std: 2029.93836, params: {'n_estimators': 500, 'subsample': 0.9, 'colsample_bytree': 0.67, 'max_depth': 20, 'min_child_weight': 20}
  mean: 794548.01245, std: 2062.41505, params: {'n_estimators': 500, 'subsample': 0.9, 'colsample_bytree': 0.67, 'max_depth': 50, 'min_child_weight': 20}
best score: 809673.54959
best params: {'n_estimators': 500, 'subsample': 0.9, 'colsample_bytree': 0.67, 'max_depth': 6, 'min_child_weight': 20}
>>> 'max_depth': 6, 'min_child_weight': 20

grid scores:
  mean: 802508.37926, std: 4201.47242, params: {'n_estimators': 500, 'subsample': 0.9, 'colsample_bytree': 0.67, 'max_depth': 5, 'min_child_weight': 10}
  mean: 793935.52998, std: 7607.45918, params: {'n_estimators': 500, 'subsample': 0.9, 'colsample_bytree': 0.67, 'max_depth': 5, 'min_child_weight': 20}
  mean: 784568.74090, std: 7161.04235, params: {'n_estimators': 500, 'subsample': 0.9, 'colsample_bytree': 0.67, 'max_depth': 5, 'min_child_weight': 40}
  mean: 802325.99222, std: 1833.64884, params: {'n_estimators': 500, 'subsample': 0.9, 'colsample_bytree': 0.67, 'max_depth': 7, 'min_child_weight': 10}
  mean: 806358.37855, std: 4488.86812, params: {'n_estimators': 500, 'subsample': 0.9, 'colsample_bytree': 0.67, 'max_depth': 7, 'min_child_weight': 20}
  mean: 808437.63308, std: 3881.55687, params: {'n_estimators': 500, 'subsample': 0.9, 'colsample_bytree': 0.67, 'max_depth': 7, 'min_child_weight': 40}
  mean: 798618.25778, std: 2948.03146, params: {'n_estimators': 500, 'subsample': 0.9, 'colsample_bytree': 0.67, 'max_depth': 8, 'min_child_weight': 10}
  mean: 802665.25722, std: 2350.85430, params: {'n_estimators': 500, 'subsample': 0.9, 'colsample_bytree': 0.67, 'max_depth': 8, 'min_child_weight': 20}
  mean: 806720.10926, std: 2543.82598, params: {'n_estimators': 500, 'subsample': 0.9, 'colsample_bytree': 0.67, 'max_depth': 8, 'min_child_weight': 40}
  mean: 795701.38488, std: 2962.99442, params: {'n_estimators': 500, 'subsample': 0.9, 'colsample_bytree': 0.67, 'max_depth': 10, 'min_child_weight': 10}
  mean: 798151.02989, std: 2162.04583, params: {'n_estimators': 500, 'subsample': 0.9, 'colsample_bytree': 0.67, 'max_depth': 10, 'min_child_weight': 20}
  mean: 803385.26027, std: 2271.86591, params: {'n_estimators': 500, 'subsample': 0.9, 'colsample_bytree': 0.67, 'max_depth': 10, 'min_child_weight': 40}
best score: 808437.63308
best params: {'n_estimators': 500, 'subsample': 0.9, 'colsample_bytree': 0.67, 'max_depth': 7, 'min_child_weight': 40}
>>> 'max_depth': 7, 'min_child_weight': 40

grid scores:
  mean: 782028.41606, std: 9637.64116, params: {'n_estimators': 500, 'subsample': 0.9, 'colsample_bytree': 0.67, 'max_depth': 5, 'min_child_weight': 50}
  mean: 769010.75894, std: 6079.16367, params: {'n_estimators': 500, 'subsample': 0.9, 'colsample_bytree': 0.67, 'max_depth': 5, 'min_child_weight': 60}
  mean: 760914.24094, std: 9643.26515, params: {'n_estimators': 500, 'subsample': 0.9, 'colsample_bytree': 0.67, 'max_depth': 5, 'min_child_weight': 80}
  mean: 807557.88495, std: 4219.13250, params: {'n_estimators': 500, 'subsample': 0.9, 'colsample_bytree': 0.67, 'max_depth': 6, 'min_child_weight': 50}
  mean: 801663.63876, std: 7556.18492, params: {'n_estimators': 500, 'subsample': 0.9, 'colsample_bytree': 0.67, 'max_depth': 6, 'min_child_weight': 60}
  mean: 784727.73532, std: 7314.95469, params: {'n_estimators': 500, 'subsample': 0.9, 'colsample_bytree': 0.67, 'max_depth': 6, 'min_child_weight': 80}
  mean: 811735.94787, std: 3476.37280, params: {'n_estimators': 500, 'subsample': 0.9, 'colsample_bytree': 0.67, 'max_depth': 7, 'min_child_weight': 50}
  mean: 812694.98649, std: 4262.04853, params: {'n_estimators': 500, 'subsample': 0.9, 'colsample_bytree': 0.67, 'max_depth': 7, 'min_child_weight': 60}
  mean: 806342.26320, std: 7227.93062, params: {'n_estimators': 500, 'subsample': 0.9, 'colsample_bytree': 0.67, 'max_depth': 7, 'min_child_weight': 80}
best score: 812694.98649
best params: {'n_estimators': 500, 'subsample': 0.9, 'colsample_bytree': 0.67, 'max_depth': 7, 'min_child_weight': 60}
>>> 'max_depth': 7, 'min_child_weight': 60

    grid scores:
  mean: 811261.01220, std: 1387.81968, params: {'n_estimators': 500, 'subsample': 0.9, 'colsample_bytree': 0.67, 'max_depth': 7, 'min_child_weight': 55}
  mean: 812694.98649, std: 4262.04853, params: {'n_estimators': 500, 'subsample': 0.9, 'colsample_bytree': 0.67, 'max_depth': 7, 'min_child_weight': 60}
  mean: 813522.63431, std: 5054.98775, params: {'n_estimators': 500, 'subsample': 0.9, 'colsample_bytree': 0.67, 'max_depth': 7, 'min_child_weight': 65}
  mean: 811147.14498, std: 1469.98812, params: {'n_estimators': 500, 'subsample': 0.9, 'colsample_bytree': 0.67, 'max_depth': 7, 'min_child_weight': 70}
  mean: 810716.38989, std: 3383.29928, params: {'n_estimators': 500, 'subsample': 0.9, 'colsample_bytree': 0.67, 'max_depth': 8, 'min_child_weight': 55}
  mean: 810977.37920, std: 3039.52816, params: {'n_estimators': 500, 'subsample': 0.9, 'colsample_bytree': 0.67, 'max_depth': 8, 'min_child_weight': 60}
  mean: 809034.76724, std: 4751.72859, params: {'n_estimators': 500, 'subsample': 0.9, 'colsample_bytree': 0.67, 'max_depth': 8, 'min_child_weight': 65}
  mean: 810902.03165, std: 3741.53151, params: {'n_estimators': 500, 'subsample': 0.9, 'colsample_bytree': 0.67, 'max_depth': 8, 'min_child_weight': 70}
best score: 813522.63431
best params: {'n_estimators': 500, 'subsample': 0.9, 'colsample_bytree': 0.67, 'max_depth': 7, 'min_child_weight': 65}
>>> 'max_depth': 7, 'min_child_weight': 65
    """

    """  ONE HOT   ***
grid scores:
  mean: 808785.95756, std: 3732.33890, params: {'colsample_bytree': 0.67, 'min_child_weight': 65, 'n_estimators': 500, 'subsample': 0.7, 'objective': 'rank:pairwise', 'max_depth': 7}
  mean: 812217.95285, std: 2332.88180, params: {'colsample_bytree': 0.67, 'min_child_weight': 65, 'n_estimators': 500, 'subsample': 0.8, 'objective': 'rank:pairwise', 'max_depth': 7}
  mean: 807606.64678, std: 6685.61758, params: {'colsample_bytree': 0.67, 'min_child_weight': 65, 'n_estimators': 500, 'subsample': 0.9, 'objective': 'rank:pairwise', 'max_depth': 7}
  mean: 798856.94075, std: 8380.25083, params: {'colsample_bytree': 0.67, 'min_child_weight': 65, 'n_estimators': 500, 'subsample': 1.0, 'objective': 'rank:pairwise', 'max_depth': 7}
best score: 812217.95285
best params: {'colsample_bytree': 0.67, 'min_child_weight': 65, 'n_estimators': 500, 'subsample': 0.8, 'objective': 'rank:pairwise', 'max_depth': 7}
>>> 'subsample': 0.8

grid scores:
  mean: 807015.44496, std: 1745.61053, params: {'colsample_bytree': 0.6, 'min_child_weight': 65, 'n_estimators': 500, 'subsample': 0.7, 'objective': 'rank:pairwise', 'max_depth': 7}
  mean: 811229.10339, std: 2626.00511, params: {'colsample_bytree': 0.6, 'min_child_weight': 65, 'n_estimators': 500, 'subsample': 0.8, 'objective': 'rank:pairwise', 'max_depth': 7}
  mean: 809107.13182, std: 3766.10287, params: {'colsample_bytree': 0.6, 'min_child_weight': 65, 'n_estimators': 500, 'subsample': 0.9, 'objective': 'rank:pairwise', 'max_depth': 7}
  mean: 806700.86249, std: 1673.53048, params: {'colsample_bytree': 0.7, 'min_child_weight': 65, 'n_estimators': 500, 'subsample': 0.7, 'objective': 'rank:pairwise', 'max_depth': 7}
  mean: 808803.17397, std: 2141.65596, params: {'colsample_bytree': 0.7, 'min_child_weight': 65, 'n_estimators': 500, 'subsample': 0.8, 'objective': 'rank:pairwise', 'max_depth': 7}
  mean: 810538.01687, std: 4532.84397, params: {'colsample_bytree': 0.7, 'min_child_weight': 65, 'n_estimators': 500, 'subsample': 0.9, 'objective': 'rank:pairwise', 'max_depth': 7}
  mean: 808655.96400, std: 2917.13000, params: {'colsample_bytree': 0.8, 'min_child_weight': 65, 'n_estimators': 500, 'subsample': 0.7, 'objective': 'rank:pairwise', 'max_depth': 7}
  mean: 809115.42707, std: 2625.94051, params: {'colsample_bytree': 0.8, 'min_child_weight': 65, 'n_estimators': 500, 'subsample': 0.8, 'objective': 'rank:pairwise', 'max_depth': 7}
  mean: 809472.09619, std: 3144.06367, params: {'colsample_bytree': 0.8, 'min_child_weight': 65, 'n_estimators': 500, 'subsample': 0.9, 'objective': 'rank:pairwise', 'max_depth': 7}
best score: 811229.10339
best params: {'colsample_bytree': 0.6, 'min_child_weight': 65, 'n_estimators': 500, 'subsample': 0.8, 'objective': 'rank:pairwise', 'max_depth': 7}
>>> 'colsample_bytree': 0.6, 'subsample': 0.8

grid scores:
  mean: 804112.81441, std: 8632.61400, params: {'colsample_bytree': 0.67, 'learning_rate': 0.03, 'min_child_weight': 65, 'n_estimators': 500, 'subsample': 0.8, 'objective': 'rank:pairwise', 'max_depth': 7}
  mean: 812217.95285, std: 2332.88180, params: {'colsample_bytree': 0.67, 'learning_rate': 0.045, 'min_child_weight': 65, 'n_estimators': 500, 'subsample': 0.8, 'objective': 'rank:pairwise', 'max_depth': 7}
  mean: 807667.00596, std: 2109.00871, params: {'colsample_bytree': 0.67, 'learning_rate': 0.06, 'min_child_weight': 65, 'n_estimators': 500, 'subsample': 0.8, 'objective': 'rank:pairwise', 'max_depth': 7}
best score: 812217.95285
best params: {'colsample_bytree': 0.67, 'learning_rate': 0.045, 'min_child_weight': 65, 'n_estimators': 500, 'subsample': 0.8, 'objective': 'rank:pairwise', 'max_depth': 7}
>>> 'learning_rate': 0.045

grid scores:
  mean: 810936.40241, std: 2661.32895, params: {'colsample_bytree': 0.67, 'learning_rate': 0.04, 'min_child_weight': 65, 'n_estimators': 500, 'subsample': 0.8, 'objective': 'rank:pairwise', 'max_depth': 7}
  mean: 812217.95285, std: 2332.88180, params: {'colsample_bytree': 0.67, 'learning_rate': 0.045, 'min_child_weight': 65, 'n_estimators': 500, 'subsample': 0.8, 'objective': 'rank:pairwise', 'max_depth': 7}
  mean: 811906.13100, std: 3339.45916, params: {'colsample_bytree': 0.67, 'learning_rate': 0.05, 'min_child_weight': 65, 'n_estimators': 500, 'subsample': 0.8, 'objective': 'rank:pairwise', 'max_depth': 7}
best score: 812217.95285
best params: {'colsample_bytree': 0.67, 'learning_rate': 0.045, 'min_child_weight': 65, 'n_estimators': 500, 'subsample': 0.8, 'objective': 'rank:pairwise', 'max_depth': 7}
>>> 'learning_rate': 0.045

grid scores:
  mean: 811208.11854, std: 3435.62254, params: {'colsample_bytree': 0.65, 'learning_rate': 0.045, 'min_child_weight': 65, 'n_estimators': 500, 'subsample': 0.75, 'objective': 'rank:pairwise', 'max_depth': 7}
  mean: 809690.89826, std: 2189.70344, params: {'colsample_bytree': 0.65, 'learning_rate': 0.045, 'min_child_weight': 65, 'n_estimators': 500, 'subsample': 0.8, 'objective': 'rank:pairwise', 'max_depth': 7}
  mean: 812432.77112, std: 4630.29708, params: {'colsample_bytree': 0.65, 'learning_rate': 0.045, 'min_child_weight': 65, 'n_estimators': 500, 'subsample': 0.85, 'objective': 'rank:pairwise', 'max_depth': 7}
  mean: 808096.24696, std: 3457.47957, params: {'colsample_bytree': 0.67, 'learning_rate': 0.045, 'min_child_weight': 65, 'n_estimators': 500, 'subsample': 0.75, 'objective': 'rank:pairwise', 'max_depth': 7}
  mean: 812217.95285, std: 2332.88180, params: {'colsample_bytree': 0.67, 'learning_rate': 0.045, 'min_child_weight': 65, 'n_estimators': 500, 'subsample': 0.8, 'objective': 'rank:pairwise', 'max_depth': 7}
  mean: 811478.50583, std: 2864.64292, params: {'colsample_bytree': 0.67, 'learning_rate': 0.045, 'min_child_weight': 65, 'n_estimators': 500, 'subsample': 0.85, 'objective': 'rank:pairwise', 'max_depth': 7}
best score: 812432.77112
best params: {'colsample_bytree': 0.65, 'learning_rate': 0.045, 'min_child_weight': 65, 'n_estimators': 500, 'subsample': 0.85, 'objective': 'rank:pairwise', 'max_depth': 7}
>>> 'learning_rate': 0.045, 'colsample_bytree': 0.65, 'subsample': 0.85

grid scores:
  mean: 810759.12376, std: 1528.10128, params: {'colsample_bytree': 0.65, 'learning_rate': 0.045, 'min_child_weight': 50, 'n_estimators': 500, 'subsample': 0.85, 'objective': 'rank:pairwise', 'max_depth': 7}
  mean: 812432.77112, std: 4630.29708, params: {'colsample_bytree': 0.65, 'learning_rate': 0.045, 'min_child_weight': 65, 'n_estimators': 500, 'subsample': 0.85, 'objective': 'rank:pairwise', 'max_depth': 7}
  mean: 811977.26246, std: 4991.06664, params: {'colsample_bytree': 0.65, 'learning_rate': 0.045, 'min_child_weight': 80, 'n_estimators': 500, 'subsample': 0.85, 'objective': 'rank:pairwise', 'max_depth': 7}
best score: 812432.77112
best params: {'colsample_bytree': 0.65, 'learning_rate': 0.045, 'min_child_weight': 65, 'n_estimators': 500, 'subsample': 0.85, 'objective': 'rank:pairwise', 'max_depth': 7}
>>> no change

grid scores:
  mean: 812432.77112, std: 4630.29708, params: {'colsample_bytree': 0.65, 'learning_rate': 0.045, 'min_child_weight': 65, 'n_estimators': 500, 'subsample': 0.85, 'objective': 'rank:pairwise', 'max_depth': 7, 'gamma': 0.0}
  mean: 810059.35718, std: 1824.61383, params: {'colsample_bytree': 0.65, 'learning_rate': 0.045, 'min_child_weight': 65, 'n_estimators': 500, 'subsample': 0.85, 'objective': 'rank:pairwise', 'max_depth': 7, 'gamma': 1.0}
  mean: 810675.99076, std: 3153.29552, params: {'colsample_bytree': 0.65, 'learning_rate': 0.045, 'min_child_weight': 65, 'n_estimators': 500, 'subsample': 0.85, 'objective': 'rank:pairwise', 'max_depth': 7, 'gamma': 2.0}
  mean: 809731.68599, std: 4091.21405, params: {'colsample_bytree': 0.65, 'learning_rate': 0.045, 'min_child_weight': 65, 'n_estimators': 500, 'subsample': 0.85, 'objective': 'rank:pairwise', 'max_depth': 7, 'gamma': 4}
  mean: 751547.94084, std: 10351.41628, params: {'colsample_bytree': 0.65, 'learning_rate': 0.045, 'min_child_weight': 65, 'n_estimators': 500, 'subsample': 0.85, 'objective': 'rank:pairwise', 'max_depth': 7, 'gamma': 10}
best score: 812432.77112
best params: {'colsample_bytree': 0.65, 'learning_rate': 0.045, 'min_child_weight': 65, 'n_estimators': 500, 'subsample': 0.85, 'objective': 'rank:pairwise', 'max_depth': 7, 'gamma': 0.0}
>>> 'gamma': 0

grid scores:
  mean: 786632.68012, std: 7192.58756, params: {'colsample_bytree': 0.65, 'learning_rate': 0.045, 'min_child_weight': 65, 'n_estimators': 500, 'subsample': 0.85, 'base_score': 0.4, 'objective': 'rank:pairwise', 'max_depth': 7, 'gamma': 0.0}
  mean: 746308.67706, std: 11236.07404, params: {'colsample_bytree': 0.65, 'learning_rate': 0.045, 'min_child_weight': 65, 'n_estimators': 500, 'subsample': 0.85, 'base_score': 0.3, 'objective': 'rank:pairwise', 'max_depth': 7, 'gamma': 0.0}
  mean: 695506.87990, std: 7719.89820, params: {'colsample_bytree': 0.65, 'learning_rate': 0.045, 'min_child_weight': 65, 'n_estimators': 500, 'subsample': 0.85, 'base_score': 0.2, 'objective': 'rank:pairwise', 'max_depth': 7, 'gamma': 0.0}
  mean: 812432.77112, std: 4630.29708, params: {'colsample_bytree': 0.65, 'learning_rate': 0.045, 'min_child_weight': 65, 'n_estimators': 500, 'subsample': 0.85, 'base_score': 0.5, 'objective': 'rank:pairwise', 'max_depth': 7, 'gamma': 0.0}
  mean: 802604.50597, std: 2869.94386, params: {'colsample_bytree': 0.65, 'learning_rate': 0.045, 'min_child_weight': 65, 'n_estimators': 500, 'subsample': 0.85, 'base_score': 0.7, 'objective': 'rank:pairwise', 'max_depth': 7, 'gamma': 0.0}
  mean: 798443.86047, std: 1863.64175, params: {'colsample_bytree': 0.65, 'learning_rate': 0.045, 'min_child_weight': 65, 'n_estimators': 500, 'subsample': 0.85, 'base_score': 0.8, 'objective': 'rank:pairwise', 'max_depth': 7, 'gamma': 0.0}
best score: 812432.77112
best params: {'colsample_bytree': 0.65, 'learning_rate': 0.045, 'min_child_weight': 65, 'n_estimators': 500, 'subsample': 0.85, 'base_score': 0.5, 'objective': 'rank:pairwise', 'max_depth': 7, 'gamma': 0.0}
>>> 'base_score': 0.5

    """

    param_grid = {
                #'objective': ['binary:logitraw'],
                'objective': ['rank:pairwise'],
                #'booster': ['gblinear'],
                'n_estimators': [600],

                'max_depth': [7],
                'min_child_weight': [65],
                'gamma': [0.],

                'subsample': [0.85],
                'colsample_bytree': [0.65],

                'learning_rate': [0.045],
                }
    for k, v in cv_grid.items():
        param_grid[k] = v


    grid = GridSearchCV(estimator=clf,
                            param_grid=param_grid,
                            cv=StratifiedKFold(train_y, n_folds=nfolds),
                            scoring=tco_scorer,
                            n_jobs=1,
                            verbose=2,
                            refit=False)

    grid.fit(train_X, train_y)
    print('grid scores:')
    for item in grid.grid_scores_:
        print('  {:s}'.format(item))
    print('best score: {:.5f}'.format(grid.best_score_))
    print('best params:', grid.best_params_)

    if False:
        clf.fit(train_X, train_y)
        feature_names = train_X.columns.values.tolist()
        from numpy import zeros
        feature_importances = zeros(len(feature_names))
        importances = clf.booster().get_fscore()
        for i, feat in enumerate(feature_names):
            if feat in importances:
                feature_importances[i] += importances[feat]
                pass
            pass
        import operator
        sorted_importances = sorted(zip(feature_names, feature_importances), key=operator.itemgetter(1), reverse=True)
        for k, v in sorted_importances:
            print("{}\t{}".format(v, k))
            pass
        print([k for k, v in sorted_importances if v == 0])
        pass

    return



def main(argv=None): # IGNORE:C0111
    '''Command line options.'''
    from sys import argv as Argv

    if argv is None:
        argv = Argv
        pass
    else:
        Argv.extend(argv)
        pass

    from os.path import basename
    program_name = basename(Argv[0])
    program_version = "v%s" % __version__
    program_build_date = str(__updated__)
    program_version_message = '%%(prog)s %s (%s)' % (program_version, program_build_date)
    try:
        program_shortdesc = __import__('__main__').__doc__.split("\n")[1]
    except:
        program_shortdesc = __import__('__main__').__doc__
    program_license = '''%s

  Created by Wojciech Migda on %s.
  Copyright 2016 Wojciech Migda. All rights reserved.

  Licensed under the MIT License

  Distributed on an "AS IS" basis without warranties
  or conditions of any kind, either express or implied.

USAGE
''' % (program_shortdesc, str(__date__))

    try:
        from argparse import ArgumentParser
        from argparse import RawDescriptionHelpFormatter
        from argparse import FileType
        from sys import stdout

        # Setup argument parser
        parser = ArgumentParser(description=program_license, formatter_class=RawDescriptionHelpFormatter)

        parser.add_argument("-n", "--num-est",
            type=int, default=700, action='store', dest="nest",
            help="number of Random Forest estimators")

        parser.add_argument("-j", "--jobs",
            type=int, default=-1, action='store', dest="njobs",
            help="number of jobs")

        parser.add_argument("-f", "--cv-fold",
            type=int, default=5, action='store', dest="nfolds",
            help="number of cross-validation folds")

        parser.add_argument("--int-fold",
            type=int, default=6, action='store', dest="int_fold",
            help="internal fold for PrudentialRegressorCVO2FO")

        parser.add_argument("-b", "--n-buckets",
            type=int, default=8, action='store', dest="nbuckets",
            help="number of buckets for digitizer")

        parser.add_argument("-o", "--out-csv",
            action='store', dest="out_csv_file", default=stdout,
            type=FileType('w'),
            help="output CSV file name")

        parser.add_argument("-m", "--minimizer",
            action='store', dest="minimizer", default='BFGS',
            type=str, choices=['Powell', 'CG', 'BFGS'],
            help="minimizer method for scipy.optimize.minimize")

        parser.add_argument("-M", "--mvector",
            action='store', dest="mvector", default=[-1.5, -2.6, -3.6, -1.2, -0.8, 0.04, 0.7, 3.6],
            type=float, nargs='*',
            help="minimizer's initial params vector")

        parser.add_argument("-I", "--imputer",
            action='store', dest="imputer", default=None,
            type=str, choices=['mean', 'median', 'most_frequent'],
            help="Imputer strategy, None is -1")

        parser.add_argument("--clf-params",
            type=str, default="{}", action='store', dest="clf_params",
            help="classifier parameters subset to override defaults")

        parser.add_argument("-G", "--cv-grid",
            type=str, default="{}", action='store', dest="cv_grid",
            help="cross-validation grid params (used if NFOLDS > 0)")

        parser.add_argument("-E", "--estimator",
            action='store', dest="estimator", default='PrudentialRegressor',
            type=str,# choices=['mean', 'median', 'most_frequent'],
            help="Estimator class to use")

        # Process arguments
        args = parser.parse_args()

        for k, v in args.__dict__.items():
            print(str(k) + ' => ' + str(v))
            pass

        work(args.out_csv_file,
             args.estimator,
             args.nest,
             args.njobs,
             args.nfolds,
             eval(args.cv_grid),
             args.minimizer,
             args.nbuckets,
             args.mvector,
             args.imputer,
             eval(args.clf_params),
             args.int_fold)


        return 0
    except KeyboardInterrupt:
        ### handle keyboard interrupt ###
        return 0
    except Exception as e:
        if DEBUG:
            raise(e)
            pass
        indent = len(program_name) * " "
        from sys import stderr
        stderr.write(program_name + ": " + repr(e) + "\n")
        stderr.write(indent + "  for help use --help")
        return 2

    pass


if __name__ == "__main__":
    if DEBUG:
        from sys import argv
        argv.append("-n 700")
        argv.append("--minimizer=Powell")
        argv.append("--clf-params={'learning_rate': 0.05, 'min_child_weight': 240, 'subsample': 0.9, 'colsample_bytree': 0.67, 'max_depth': 6, 'initial_params': [0.1, -1, -2, -1, -0.8, 0.02, 0.8, 1]}")
        argv.append("-f 10")
        pass
    from sys import exit as Exit
    Exit(main())
    pass
