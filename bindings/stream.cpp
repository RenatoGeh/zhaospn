#include "../src/fmath.hpp"
#include "../src/SPNetwork.h"
#include "../src/StreamParamLearning.h"

#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "stream.h"

using namespace SPN;

namespace py = pybind11;

void bind_stream_base(py::module &m) {
  class PyStreamParamLearning : public StreamParamLearning {
    public:
      using StreamParamLearning::StreamParamLearning;

      void fit(std::vector<double> &train, SPNetwork &spn, bool verbose = false) override {
        PYBIND11_OVERLOAD(void, StreamParamLearning, fit, train, spn, verbose);
      }
  };

  py::class_<StreamParamLearning, PyStreamParamLearning>(m, "StreamParamLearning", "Abstract class "
      "for stream parameter learning of SPNs")
    .def(py::init<>(), "Constructs a stream parameter learning object")
    .def("fit", &StreamParamLearning::fit, py::arg("train"), py::arg("spn"),
          py::arg("verbose") = false, "Fit spn to given trains dataset and valids validation set")
    .def("algo_name", &StreamParamLearning::algo_name, "Returns the name of the stream parameter "
        "learning algorithm");
}

void bind_stream_projgd(py::module &m) {
  class PyStreamProjectedGD : public StreamProjectedGD {
    public:
      using StreamProjectedGD::StreamProjectedGD;

      void fit(std::vector<double> &train, SPNetwork &spn, bool verbose = false) override {
        PYBIND11_OVERLOAD(void, StreamProjectedGD, fit, train, spn, verbose);
      }
  };

  py::class_<StreamProjectedGD, StreamParamLearning, PyStreamProjectedGD>(m, "StreamProjectedGD",
      "Projected gradient descent (PGD) stream learner")
    .def(py::init<>(), "Constructs a PGD learner with default parameters")
    .def(py::init<double, double>(), py::arg("proj_eps") = 1e-2, py::arg("lrate") = 1e-1,
        "Constructs a PGD learner with given parameters");
}

void bind_stream_expgd(py::module &m) {
  class PyStreamExpoGD : public StreamExpoGD {
    public:
      using StreamExpoGD::StreamExpoGD;

      void fit(std::vector<double> &train, SPNetwork &spn, bool verbose = false) override {
        PYBIND11_OVERLOAD(void, StreamExpoGD, fit, train, spn, verbose);
      }
  };

  py::class_<StreamExpoGD, StreamParamLearning, PyStreamExpoGD>(m, "StreamExpoGD",
      "Exponentiated gradient descent (EGD) stream learner")
    .def(py::init<>(), "Constructs an EGD learner with default parameters")
    .def(py::init<double>(), py::arg("lrate") = 1e-1, "Constructs an EGD learner with given parameters");
}

void bind_stream_sma(py::module &m) {
  class PyStreamSMA : public StreamSMA {
    public:
      using StreamSMA::StreamSMA;

      void fit(std::vector<double> &train, SPNetwork &spn, bool verbose = false) override {
        PYBIND11_OVERLOAD(void, StreamSMA, fit, train, spn, verbose);
      }
  };

  py::class_<StreamSMA, StreamParamLearning, PyStreamSMA>(m, "StreamSMA", "Sequential onomial "
      "minimization algorithm (SMA) stream learner")
    .def(py::init<>(), "Constructs an SMA learner with default parameters")
    .def(py::init<double>(), py::arg("lrate") = 1e-1, "Constructs an SMA learner with given parameters");
}

void bind_stream_em(py::module &m) {
  class PyStreamExpectMax : public StreamExpectMax {
    public:
      using StreamExpectMax::StreamExpectMax;

      void fit(std::vector<double> &train, SPNetwork &spn, bool verbose = false) override {
        PYBIND11_OVERLOAD(void, StreamExpectMax, fit, train, spn, verbose);
      }
  };

  py::class_<StreamExpectMax, StreamParamLearning, PyStreamExpectMax>(m, "StreamExpectMax",
      "Expectation maximization (EM), equivalent to concave-convex procedure (CCCP), stream learner")
    .def(py::init<>(), "Constructs an EM learner with default parameters")
    .def(py::init<double>(), py::arg("lrate") = 1.0, "Constructs an EM learner with given parameters");
}
