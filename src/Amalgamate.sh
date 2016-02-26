#!/bin/sh

/repo/topcoder/Amalgamate/Amalgamate \
-w "*.cpp;*.c;*.cc;*.h;*.hpp" \
-i . \
-i include \
-i dmlc-core/include \
-i rabit/include \
amalgamate.cpp submission.cpp

sed -i  "s/#pragma omp .*//g" submission.cpp

#g++ -std=c++11 -c submission.cpp
g++ -std=c++11 -c submission.cpp -W -Wall -Wno-sign-compare -O2 -s -pipe -mmmx -msse -msse2 -msse3

#g++ -std=c++11 -c submission.cpp -W -Wall -Wno-sign-compare -Os -s -pipe #-mmmx -msse -msse2 -msse3
