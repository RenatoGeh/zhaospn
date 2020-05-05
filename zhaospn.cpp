#include <pybind11/pybind11.h>

#include "bindings/node.h"
#include "bindings/network.h"
#include "bindings/utils.h"
#include "bindings/batch.h"
#include "bindings/online.h"
#include "bindings/stream.h"

PYBIND11_MODULE(zhaospn, m) {
  /* SPN submodule */
  auto spn = m.def_submodule("spn", "Sum-product network module containing node types and basic "
      "inference methods");

  /* Bindings for SPNNode */
  bind_spn_enums(spn);
  bind_spn_node(spn);
  bind_spn_sum(spn);
  bind_spn_prod(spn);
  bind_spn_var(spn);
  bind_spn_bin(spn);
  bind_spn_top(spn);
  bind_spn_bot(spn);
  bind_spn_bernoulli(spn);
  bind_spn_normal(spn);

  /* Bindings for SPNetwork */
  bind_net_spnetwork(spn);

  /* Utils submodule */
  auto utils = m.def_submodule("utils", "Utility functions submodule");

  /* Bindings for utils */
  bind_utils(utils);

  /* Batch submodule */
  auto batch = m.def_submodule("batch", "Batch learning submodule");

  /* Bindings for batch learning methods */
  bind_batch_base(batch);
  bind_batch_em(batch);
  bind_batch_cvb(batch);
  bind_batch_projgd(batch);
  bind_batch_lbfgs(batch);
  bind_batch_expgd(batch);
  bind_batch_sma(batch);

  /* Online submodule */
  auto online = m.def_submodule("online", "Online learning submodule");

  /* Bindings for online learning methods */
  bind_online_base(online);
  bind_online_projgd(online);
  bind_online_expgd(online);
  bind_online_sma(online);
  bind_online_em(online);
  bind_online_cvb(online);
  bind_online_adf(online);
  bind_online_bmm(online);


  /* Streaming submodule */
  auto streaming = m.def_submodule("stream", "Streaming learning submodule");

  /* Bindings for stream learning methods */
  bind_stream_base(streaming);
  bind_stream_projgd(streaming);
  bind_stream_expgd(streaming);
  bind_stream_sma(streaming);
  bind_stream_em(streaming);
}
