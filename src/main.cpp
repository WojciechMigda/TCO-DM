/*******************************************************************************
 * Copyright (c) 2016 Wojciech Migda
 * All rights reserved
 * Distributed under the terms of the MIT License
 *******************************************************************************
 *
 * Filename: main.cpp
 *
 * Description:
 *      TCO-DM
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

#include "DemographicMembership.hpp"

#include <cstdlib>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <random>
#include <algorithm>

std::vector<std::string>
read_file(std::string && fname)
{
    std::ifstream fcsv(fname);
    std::vector<std::string> vcsv;

    for (std::string line; std::getline(fcsv, line); /* nop */)
    {
        vcsv.push_back(line);
    }
    fcsv.close();

    return vcsv;
}


int main(int argc, char **argv)
{
    const int SEED = (argc == 2 ? std::atoi(argv[1]) : 1);
    const char * FNAME = (argc == 3 ? argv[2] : "../data/demographic_membership_training.csv");

    std::cerr << "SEED: " << SEED << ", CSV: " << FNAME << std::endl;


    std::vector<std::string> vcsv = read_file(std::string(FNAME));
    std::cerr << "Read " << vcsv.size() << " lines" << std::endl;


    // shuffle, except first row which carries feature names
    std::mt19937 g(SEED);
    std::shuffle(vcsv.begin() + 1, vcsv.end(), g);


    // simple split train/test
    const std::size_t PIVOT = 0.67 * vcsv.size();

    std::vector<std::string> train_data;
    std::vector<std::string> test_data;

    // for train data skip first row with feature names
    std::copy(vcsv.cbegin() + 1, vcsv.cbegin() + PIVOT, std::back_inserter(train_data));
    std::copy(vcsv.cbegin() + PIVOT, vcsv.cend(), std::back_inserter(test_data));

    std::cerr << "After split train data has " << train_data.size() << " rows" << std::endl;
    std::cerr << "After split test data has " << test_data.size() << " rows" << std::endl;

    // remove response from test data
    for (auto  & s : test_data)
    {
        s.resize(s.rfind(','));
    }


    ////////////////////////////////////////////////////////////////////////////


    const DemographicMembership solver;

    auto result = solver.predict(
        DemographicMembership::TestType::Local,
        train_data, test_data);

    return 0;
}
