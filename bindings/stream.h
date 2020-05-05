#ifndef _BINDINGS_STREAM_H
#define _BINDINGS_STREAM_H

#include <pybind11/pybind11.h>

void bind_stream_base(pybind11::module&);
void bind_stream_projgd(pybind11::module&);
void bind_stream_expgd(pybind11::module&);
void bind_stream_sma(pybind11::module&);
void bind_stream_em(pybind11::module&);

#endif
