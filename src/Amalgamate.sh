#!/bin/sh

/repo/topcoder/DemographicMembership/Amalgamate/Amalgamate \
-w "*.cpp;*.c;*.cc;*.h;*.hpp" \
-i . \
-i include \
-i dmlc-core/include \
-i rabit/include \
amalgamate.cpp submission.cpp

g++ -std=c++11 -c submission.cpp
