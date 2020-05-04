#include <pybind11/pybind11.h>

#include "bindings/node.h"
#include "bindings/network.h"

PYBIND11_MODULE(zhaospn, m) {
  /* Bindings for SPNNode */
  bind_spn_enums(m);
  bind_spn_node(m);
  bind_spn_sum(m);
  bind_spn_prod(m);
  bind_spn_var(m);
  bind_spn_bin(m);
  bind_spn_top(m);
  bind_spn_bot(m);
  bind_spn_bernoulli(m);
  bind_spn_normal(m);

  /* Bindings for SPNetwork */
  bind_net_spnetwork(m);
}

