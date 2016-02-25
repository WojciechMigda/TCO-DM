################################################################################
project( "TCO-DemographicMembership" C CXX )

cmake_minimum_required( VERSION 2.8.3 )

set( CMAKE_ECLIPSE_VERSION 3.8 )
set( _ECLIPSE_VERSION 3.8 )

set( CMAKE_CXX_COMPILER_ARG1 -std=c++11 ) ### for Eclipse's discovery extra arguments


#add_definitions( -DNO_STOCH )
add_definitions( -DDMLC_ENABLE_STD_THREAD=0 )

#add_definitions( -O0 -ggdb -std=c++11 -Wall -pedantic )
add_definitions( -O2 -ggdb -std=c++11 -Wall -pedantic )
#add_definitions( -O2 -ffast-math -fno-finite-math-only  -msse2 -march=native -std=c++11 -Wall -g -pedantic )
#add_definitions( -O2 -ftree-vectorize -ffast-math -fno-finite-math-only  -msse2 -march=native -std=c++11 -Wall -g -pedantic )
#add_definitions( -O3 -ffast-math -msse2 -march=native -std=c++11 -Wall -g -pedantic )

################################################################################

include_directories(
    src
    src/include
    src/dmlc-core/include
    src/rabit/include
)

################################################################################

add_executable( main
    src/main.cpp

    src/src/c_api/c_api.cc
    src/src/data/simple_csr_source.cc
    src/src/data/data.cc
    src/src/logging.cc
    src/rabit/src/engine.cc
    src/rabit/src/allreduce_base.cc
    src/rabit/src/allreduce_robust.cc
    src/src/c_api/c_api_error.cc
    src/dmlc-core/src/io.cc
    src/dmlc-core/src/io/local_filesys.cc
    src/src/learner.cc
    src/src/objective/objective.cc
    src/dmlc-core/src/io/input_split_base.cc
    src/src/common/common.cc
    src/src/gbm/gbm.cc
    src/dmlc-core/src/io/recordio_split.cc
    src/src/metric/metric.cc
    src/src/data/simple_dmatrix.cc
    src/dmlc-core/src/data.cc
    src/dmlc-core/src/io/line_split.cc
    src/src/objective/regression_obj.cc
    src/src/objective/multiclass_obj.cc
    src/src/metric/elementwise_metric.cc
    src/src/metric/multiclass_metric.cc
    src/src/metric/rank_metric.cc
    src/src/gbm/gbtree.cc
    src/src/gbm/gblinear.cc
    src/src/data/sparse_page_raw_format.cc
    src/src/objective/rank_obj.cc
    src/src/tree/tree_updater.cc
    src/src/tree/updater_histmaker.cc
    src/src/tree/updater_colmaker.cc
    src/src/tree/updater_skmaker.cc
    src/src/tree/updater_prune.cc
    src/src/tree/updater_refresh.cc
    src/src/tree/updater_sync.cc
    src/src/tree/tree_model.cc
)

################################################################################
