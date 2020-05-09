#include "../src/SPNetwork.h"
#include "../src/OnlineParamLearning.h"
#include "../src/fmath.hpp"

#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "online.h"

using namespace SPN;

namespace py = pybind11;

void bind_online_base(pybind11::module &m){
  class PyOnlineParamLearning : public OnlineParamLearning {
    public:
      using OnlineParamLearning::OnlineParamLearning;

      void fit(std::vector<std::vector<double>> &trains, std::vector<std::vector<double>> &valids,
          SPNetwork &spn, size_t num_iters, bool verbose = false) override {
        PYBIND11_OVERLOAD(void, OnlineParamLearning, fit, trains, valids, spn, num_iters, verbose);
      }
  };

  py::class_<OnlineParamLearning, PyOnlineParamLearning>(m, "ParamLearning", "Abstract class "
      "for online parameter learning of SPNs")
    .def(py::init<>(), "Constructs an online parameter learning object")
    .def("fit", &OnlineParamLearning::fit, py::arg("trains"), py::arg("valids"), py::arg("spn"),
        py::arg("num_iters"), py::arg("verbose") = false, "Fit spn to given trains dataset and "
        "valids validation set")
    .def("algo_name", &OnlineParamLearning::algo_name, "Returns the name of the online parameter "
        "learning algorithm");
}

void bind_online_projgd(pybind11::module &m){
  class PyOnlineProjectedGD : public OnlineProjectedGD {
    public:
      using OnlineProjectedGD::OnlineProjectedGD;

      void fit(std::vector<std::vector<double>> &trains, std::vector<std::vector<double>> &valids,
          SPNetwork &spn, size_t num_iters, bool verbose = false) override {
        PYBIND11_OVERLOAD(void, OnlineProjectedGD, fit, trains, valids, spn, num_iters, verbose);
      }
  };

  py::class_<OnlineProjectedGD, OnlineParamLearning, PyOnlineProjectedGD>(m, "ProjectedGD",
         "Projected gradient descent (PGD) online learner")
   .def(py::init<>(), "Constructs a PGD learner with default parameters")
   .def(py::init<double, double, double, double>(), py::arg("proj_eps") = 1e-2,
      py::arg("stop_thred") = 1e-3, py::arg("lrate") = 1e-1, py::arg("shrink_weight") = 0.8,
      "Constructs a PGD learner with given parameters");
}

void bind_online_expgd(pybind11::module &m){
  class PyOnlineExpoGD : public OnlineExpoGD {
    public:
      using OnlineExpoGD::OnlineExpoGD;

      void fit(std::vector<std::vector<double>> &trains, std::vector<std::vector<double>> &valids,
          SPNetwork &spn, size_t num_iters, bool verbose = false) override {
        PYBIND11_OVERLOAD(void, OnlineExpoGD, fit, trains, valids, spn, num_iters, verbose);
      }
  };

  py::class_<OnlineExpoGD, OnlineParamLearning, PyOnlineExpoGD>(m, "ExpoGD",
      "Exponentiated stochastic gradient descent (EGD) online learner")
    .def(py::init<>(), "Constructs an EGD learner with default parameters")
    .def(py::init<double, double, double>(), py::arg("stop_thred") = 1e-3, py::arg("lrate") = 1e-1,
        py::arg("shrink_weight") = 0.8, "Constructs an EGD learner with given parameters");
}

void bind_online_sma(pybind11::module &m){
  class PyOnlineSMA : public OnlineSMA {
    public:
      using OnlineSMA::OnlineSMA;

      void fit(std::vector<std::vector<double>> &trains, std::vector<std::vector<double>> &valids,
          SPNetwork &spn, size_t num_iters, bool verbose = false) override {
        PYBIND11_OVERLOAD(void, OnlineSMA, fit, trains, valids, spn, num_iters, verbose);
      }
  };

  py::class_<OnlineSMA, OnlineParamLearning, PyOnlineSMA>(m, "SMA", "Stochastic sequential "
      "monomial minimization algorithm (SMA) online learner")
    .def(py::init<>(), "Constructs an SMA learner with default parameters")
    .def(py::init<double, double, double>(), py::arg("stop_thred") = 1e-3, py::arg("lrate") = 1e-1,
        py::arg("shrink_weight") = 0.8, "Constructs an SMA learner with given parameters");
}

void bind_online_em(pybind11::module &m){
  class PyOnlineExpectMax : public OnlineExpectMax {
    public:
      using OnlineExpectMax::OnlineExpectMax;

      void fit(std::vector<std::vector<double>> &trains, std::vector<std::vector<double>> &valids,
          SPNetwork &spn, size_t num_iters, bool verbose = false) override {
        PYBIND11_OVERLOAD(void, OnlineExpectMax, fit, trains, valids, spn, num_iters, verbose);
      }
  };

  py::class_<OnlineExpectMax, OnlineParamLearning, PyOnlineExpectMax>(m, "ExpectMax",
      "Expectation maximization (EM) online learner")
    .def(py::init<>(), "Constructs an EM learner with default parameters")
    .def(py::init<double, double>(), py::arg("stop_thred") = 1e-4, py::arg("lap_lambda") = 1.0,
        "Constructs an EM learner with given parameters");
}

void bind_online_cvb(pybind11::module &m){
  class PyOnlineCollapsedVB : public OnlineCollapsedVB {
    using OnlineCollapsedVB::OnlineCollapsedVB;

    void fit(std::vector<std::vector<double>> &trains, std::vector<std::vector<double>> &valids,
        SPNetwork &spn, size_t num_iters, bool verbose = false) override {
      PYBIND11_OVERLOAD(void, OnlineCollapsedVB, fit, trains, valids, spn, num_iters, verbose);
    }
  };

  py::class_<OnlineCollapsedVB, OnlineParamLearning, PyOnlineCollapsedVB>(m, "CollapsedVB",
      "Collapsed variational bayes (CVB) online learner")
    .def(py::init<>(), "Constructs a CVB learner with default parameters")
    .def(py::init<double, double, double, uint>(), py::arg("stop_thred") = 1e-3,
        py::arg("lrate") = 1e-1, py::arg("prior_scale") = 1e2, py::arg("seed") = 42, "Constructs "
        "a CVB learner with given parameters");
}

void bind_online_adf(pybind11::module &m){
  class PyOnlineADF : public OnlineADF {
    using OnlineADF::OnlineADF;

    void fit(std::vector<std::vector<double>> &trains, std::vector<std::vector<double>> &valids,
        SPNetwork &spn, size_t num_iters, bool verbose = false) override {
      PYBIND11_OVERLOAD(void, OnlineADF, fit, trains, valids, spn, num_iters, verbose);
    }
  };

  py::class_<OnlineADF, OnlineParamLearning, PyOnlineADF>(m, "ADF", "Assumed-density "
      "filtering (ADF) online learner")
    .def(py::init<>(), "Constructs an ADF learner with default parameters")
    .def(py::init<double, double>(), py::arg("stop_thred") = 1e-3, py::arg("prior_scale") = 1e2,
        "Constructs an ADF learner with given parameters");
}

void bind_online_bmm(pybind11::module &m){
  class PyOnlineBMM : public OnlineBMM {
    public:
      using OnlineBMM::OnlineBMM;

      void fit(std::vector<std::vector<double>> &trains, std::vector<std::vector<double>> &valids,
          SPNetwork &spn, size_t num_iters, bool verbose = false) override {
        PYBIND11_OVERLOAD(void, OnlineBMM, fit, trains, valids, spn, num_iters, verbose);
      }
  };

  py::class_<OnlineBMM, OnlineParamLearning, PyOnlineBMM>(m, "BMM", "Bayesian moment "
      "matching (BMM) online learner")
    .def(py::init<>(), "Constructs a BMM learner with default parameters")
    .def(py::init<double, double>(), py::arg("stop_thred") = 1e-3, py::arg("prior_scale") = 1e2,
        "Constructs a BMM learner with given parameters");
}

