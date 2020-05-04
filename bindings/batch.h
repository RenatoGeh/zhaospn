#ifndef _BINDINGS_BATCH_H
#define _BINDINGS_BATCH_H

#include <pybind11/pybind11.h>

void bind_batch_base(pybind11::module&);
void bind_batch_em(pybind11::module&);
void bind_batch_cvb(pybind11::module&);
void bind_batch_projgd(pybind11::module&);
void bind_batch_lbfgs(pybind11::module&);
void bind_batch_expgd(pybind11::module&);
void bind_batch_sma(pybind11::module&);

#endif
