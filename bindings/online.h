#ifndef _BINDINGS_ONLINE_H
#define _BINDINGS_ONLINE_H

#include <pybind11/pybind11.h>

void bind_online_base(pybind11::module&);
void bind_online_projgd(pybind11::module&);
void bind_online_expgd(pybind11::module&);
void bind_online_sma(pybind11::module&);
void bind_online_em(pybind11::module&);
void bind_online_cvb(pybind11::module&);
void bind_online_adf(pybind11::module&);
void bind_online_bmm(pybind11::module&);

#endif
