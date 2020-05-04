#ifndef _BINDINGS_NODE_H
#define _BINDINGS_NODE_H

#include <pybind11/pybind11.h>

void bind_spn_enums(pybind11::module&);
void bind_spn_node(pybind11::module&);
void bind_spn_sum(pybind11::module&);
void bind_spn_prod(pybind11::module&);
void bind_spn_var(pybind11::module&);
void bind_spn_bin(pybind11::module&);
void bind_spn_top(pybind11::module&);
void bind_spn_bot(pybind11::module&);
void bind_spn_bernoulli(pybind11::module&);
void bind_spn_normal(pybind11::module&);

#endif
