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
#include <utility>

static const char * SKF5 =  "0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000001100000111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111122222122222222122121212222221212222222222222222112222222222222221221211222222222221221222222222222222222212121121221212221221222122122212222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222233232333333323322333332333332323333323333332323232333322323322333333333232322333333332323333332333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333334444444343444443444434444443444444433443444434434443443444444344334444434434344344334444343444434443443444444334334444444444444444443443444444344443434334444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444";
static const char * SKF10 = "0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000011100010001000001100000000000000000100000000010011111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111211111111111111112211111222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222223322333323332333333232323233333333223333333333233333233333332332323333333323323322333333332233333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333344444344444444344343434444443434444444444444444334444444444444443443433444444444443443444444444444444444434343343443434443443444344344434444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444455455554555554555455555555555554555545555554555555555545455555555554545554555555554455445545555545545554455544555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555566565666666656655666665666665656666656666665656565666655656655666666666565655666666665656666665666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666677767776777777677677777766767777776777677777776776677677676777777777677777676767776766677777777776777777776777677677777767777776777777777777776776677777777766777777777767777677776777777677777677777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777778888888787888887888878888887888888877887888878878887887888888788778888878878788788778888787888878887887888888778778888888888888888887887888888788887878778888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888889899998899998899989989999998998989998988999999998989898999899999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999";

template <typename _Tp>
std::pair<std::vector<_Tp>, std::vector<_Tp>>
KFold(const char * descriptor, int iter, const std::vector<_Tp> & df)
{
    std::vector<_Tp> tr;
    std::vector<_Tp> te;

    for (std::size_t pos{0}; pos < df.size(); ++pos)
    {
        auto decoder = [](const char c)
            {
                if (std::isdigit(c))
                {
                    return c - '0';
                }
                else if (std::isupper(c))
                {
                    return c - 'A' + 10;
                }
                else if (std::islower(c))
                {
                    return c - 'a' + 10 + 26;
                }
                else
                {
                    assert(false);
                }
            };
        if (decoder(descriptor[pos]) == iter)
        {
            te.push_back(df[pos]);
        }
        else
        {
            tr.push_back(df[pos]);
        }
    }

    return {tr, te};
}

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

float minPRscore(const std::vector<int> & y_labels, const std::vector<int> & y_hat)
{
//    const float TP = std::accumulate(y_labels.cbegin(), y_labels.cend(), 0);
    const float TP = std::inner_product(y_labels.cbegin(), y_labels.cend(), y_hat.cbegin(), 0,
        [](int a, int b){return a + b;},
        [](int a, int b){return a == 1 && b == 1;}
    );
    std::cerr << "TP: " << TP << std::endl;

    const float FP = std::inner_product(y_labels.cbegin(), y_labels.cend(), y_hat.cbegin(), 0,
        [](int a, int b){return a + b;},
        [](int a, int b){return a == 0 && b == 1;}
    );
    std::cerr << "FP: " << FP << std::endl;
    const float FN = std::inner_product(y_labels.cbegin(), y_labels.cend(), y_hat.cbegin(), 0,
        [](int a, int b){return a + b;},
        [](int a, int b){return a == 1 && b == 0;}
    );
    std::cerr << "FN: " << FN << std::endl;

    std::cerr << "Precision: " << TP / (TP + FP) << std::endl;
    std::cerr << "Recall: " << TP / (TP + FN) << std::endl;

    return 1e6 * std::min(TP / (TP + FP), TP / (TP + FN));
}

int main(int argc, char **argv)
{
    const int SEED = (argc == 2 ? std::atoi(argv[1]) : 1);
    const char * FNAME = (argc == 3 ? argv[2] : "../data/demographic_membership_training.csv");

    std::cerr << "SEED: " << SEED << ", CSV: " << FNAME << std::endl;


    std::vector<std::string> vcsv = read_file(std::string(FNAME));
    std::cerr << "Read " << vcsv.size() << " lines" << std::endl;

    // for train data skip first row with feature names
    vcsv.erase(vcsv.begin());

#if 0
    std::mt19937 g(SEED);
    std::shuffle(vcsv.begin(), vcsv.end(), g);

    // simple split train/test
    const std::size_t PIVOT = 0.67 * vcsv.size();

    std::vector<std::string> train_data;
    std::vector<std::string> test_data;

    std::copy(vcsv.cbegin(), vcsv.cbegin() + PIVOT, std::back_inserter(train_data));
    std::copy(vcsv.cbegin() + PIVOT, vcsv.cend(), std::back_inserter(test_data));

    std::cerr << "After split train data has " << train_data.size() << " rows" << std::endl;
    std::cerr << "After split test data has " << test_data.size() << " rows" << std::endl;

    // remove response from test data and append it to test_y
    std::vector<int> test_y;
    for (auto  & s : test_data)
    {
        auto pos = s.rfind(',');

        test_y.push_back(s[pos + 1] - '0');

        s.resize(pos);
    }
#endif


    ////////////////////////////////////////////////////////////////////////////

    const int NFOLDS{5};
    const DemographicMembership solver;

    std::vector<float> CVscores(NFOLDS);

    for (int fold{0}; fold < NFOLDS; ++fold)
    {
        std::vector<std::string> train_data;
        std::vector<std::string> test_data;


        std::tie(train_data, test_data) = KFold(SKF5, fold, vcsv);


        // remove response from test data and append it to test_y
        std::vector<int> test_y;
        for (auto  & s : test_data)
        {
            auto pos = s.rfind(',');

            test_y.push_back(s[pos + 1] - '0');

            s.resize(pos);
        }

        const auto y_hat = solver.predict(
            DemographicMembership::TestType::Local,
            train_data, test_data);

        auto score = minPRscore(test_y, y_hat);
        std::cerr << "[" << fold + 1 << "] score: " << score << std::endl;

        CVscores[fold] = score;
    }

    std::cerr << "  mean score: " << std::accumulate(CVscores.cbegin(), CVscores.cend(), 0) / NFOLDS << std::endl;

    return 0;
}
