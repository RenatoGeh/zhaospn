#include <boost/random/mersenne_twister.hpp>
#include <boost/random/extreme_value_distribution.hpp>
#include <boost/random/normal_distribution.hpp>

#include "fmath.hpp"

#include "random.h"

namespace SPN {
  namespace random {
    rand_gen _def_generator;
    boost::random::extreme_value_distribution<double> _gumbel(0.0, 1.0);
    boost::random::normal_distribution<double> _gaussian(0.0, 1.0);

    rand_gen& get_generator(void) { return _def_generator; }

    void set_seed(uint seed) { _def_generator.seed(seed); }

    double gumbel(void) { return _gumbel(_def_generator); }

    double gaussian(double loc, double scale) {
      _gaussian.param(boost::random::normal_distribution<double>::param_type(loc, scale));
      return _gaussian(_def_generator);
    }
  }
}
