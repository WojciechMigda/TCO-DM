# TCO-DemographicMembership

 1. One Hot encoding of nominal features: GENDER, REGISTRATION_ROUTE, REGISTRATION_CONTEXT, MIGRATED_USER_TYPE, PLATFORM_CENTRE, TOD_CENTRE, CONTENT_CENTRE
 2. Training three XGBoost estimators, each configured with a different objective function, namely: `rank:pairwise`, `binary:logistic`, and `reg:linear`.

Build:
 `cmake . && make`

NOTE 1: estimator parameters were obtained from `hyperopt` runs using `muse_estimator.py` script.
NOTE 2: XGBoost code was tweaked a bit to allow for: 1) amalgamation, 2) compilation on the TopCoder testing engine, and last but not least 3) `std::log` and `std::exp` invocations were
forced to use double precision, as it turned out to be a score booster (my guess for the reason for that is the `glibc` version on TopCoder machines which supposedly predates `glibc` changes
described here: http://developerblog.redhat.com/2015/01/02/improving-math-performance-in-glibc/).
