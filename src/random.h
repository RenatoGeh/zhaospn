#ifndef _SPN_RANDOM_H
#define _SPN_RANDOM_H

#include <boost/random/mersenne_twister.hpp>

#include "fmath.hpp"

namespace SPN {
  namespace random {
    typedef boost::random::mt19937 rand_gen;

    rand_gen& get_generator(void);

    void set_seed(uint seed);

    double gumbel(void);

    double gaussian(double loc, double scale);
  }
}

#endif
