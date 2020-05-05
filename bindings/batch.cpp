#include "../src/fmath.hpp"
#include "../src/SPNetwork.h"
#include "../src/BatchParamLearning.h"

#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "batch.h"

using namespace SPN;

namespace py = pybind11;

void bind_batch_base(py::module &m) {
  class PyBatchParamLearning : public BatchParamLearning {
    public:
      using BatchParamLearning::BatchParamLearning;

      void fit(const std::vector<std::vector<double>> &trains,
          const std::vector<std::vector<double>> &valids,
          SPNetwork &spn, bool verbose = false) override {
        PYBIND11_OVERLOAD(void, BatchParamLearning, fit, trains, valids, spn, verbose);
      }
  };

  py::class_<BatchParamLearning, PyBatchParamLearning>(m, "BatchParamLearning", "Abstract class for "
      "batch parameter learning of SPNs")
    .def(py::init<>(), "Constructs a batch parameter learning object")
    .def("fit", &BatchParamLearning::fit, py::arg("trains"), py::arg("valids"), py::arg("spn"),
          py::arg("verbose") = false, "Fit spn to given trains dataset and valids validation set")
    .def("algo_name", &BatchParamLearning::algo_name, "Returns the name of the batch parameter "
        "learning algorithm");
}

void bind_batch_em(pybind11::module &m) {
  class PyExpectMax : public ExpectMax {
    public:
      using ExpectMax::ExpectMax;

      void fit(const std::vector<std::vector<double>> &trains,
          const std::vector<std::vector<double>> &valids,
          SPNetwork &spn, bool verbose = false) override {
        PYBIND11_OVERLOAD(void, ExpectMax, fit, trains, valids, spn, verbose);
      }
  };

  py::class_<ExpectMax, BatchParamLearning, PyExpectMax>(m, "ExpectMax", "Expectation maximization "
      "(EM) batch learner, which is equivalent to concave-convex procedure (CCCP)")
    .def(py::init<>(), "Constructs an EM learner with default parameters")
    .def(py::init<int, double, double>(), py::arg("num_iters") = 50, py::arg("stop_thred") = 1e-4,
        py::arg("lap_lambda") = 1, "Constructs an EM learner with given parameters");
}

void bind_batch_cvb(pybind11::module &m) {
  class PyCollapsedVB : public CollapsedVB {
    public:
      using CollapsedVB::CollapsedVB;

      void fit(const std::vector<std::vector<double>> &trains,
          const std::vector<std::vector<double>> &valids,
          SPNetwork &spn, bool verbose = false) override {
        PYBIND11_OVERLOAD(void, CollapsedVB, fit, trains, valids, spn, verbose);
      }
  };

  py::class_<CollapsedVB, BatchParamLearning, PyCollapsedVB>(m, "CollapsedVB", "Collapsed "
      "variational bayes (CVB) batch learner")
    .def(py::init<>(), "Constructs a CVB learner with default parameters")
    .def(py::init<int, double, double, double, uint>(), py::arg("num_iters") = 50,
        py::arg("stop_thred") = 1e-4, py::arg("lrate") = 1e-1, py::arg("prior_scale") = 100.0,
        py::arg("seed") = 42, "Constructs a CVB learner with given parameters");
}

void bind_batch_projgd(pybind11::module &m) {
  class PyProjectedGD : public ProjectedGD {
    public:
      using ProjectedGD::ProjectedGD;

      void fit(const std::vector<std::vector<double>> &trains,
          const std::vector<std::vector<double>> &valids,
          SPNetwork &spn, bool verbose = false) override {
        PYBIND11_OVERLOAD(void, ProjectedGD, fit, trains, valids, spn, verbose);
      }
  };

  py::class_<ProjectedGD, BatchParamLearning, PyProjectedGD>(m, "Projected gradient descent (PGD) batch learner")
    .def(py::init<>(), "Constructs a PGD learner with default parameters")
    .def(py::init<int, double, double, double, double, bool, double, uint>(),
        py::arg("num_iters") = 50, py::arg("proj_eps") = 1e-2, py::arg("stop_thred") = 1e-3,
        py::arg("lrate") = 1e-1, py::arg("shrink_weight") = 8e-1, py::arg("map_prior") = true,
        py::arg("prior_scale") = 100.0, py::arg("seed") = 42, "Constructs a PGD learner with "
        "given parameters");
}

void bind_batch_lbfgs(pybind11::module &m) {
  class PyLBFGS : public LBFGS {
    using LBFGS::LBFGS;

    void fit(const std::vector<std::vector<double>> &trains,
        const std::vector<std::vector<double>> &valids,
        SPNetwork &spn, bool verbose = false) override {
      PYBIND11_OVERLOAD(void, LBFGS, fit, trains, valids, spn, verbose);
    }
  };

  py::class_<LBFGS, BatchParamLearning, PyLBFGS>(m, "Limited-memory Broyden-Fletcher-Goldfarb-Shanno "
      "(L-BFGS) batch learner")
    .def(py::init<>(), "Constructs an L-BFGS learner with default parameters")
    .def(py::init<int, double, double, double, double, uint>(), py::arg("num_iters") = 50,
        py::arg("proj_eps") = 1e-2, py::arg("stop_thred") = 1e-3, py::arg("lrate") = 1e-1,
        py::arg("shrink_weight") = 8e-1, py::arg("history_window") = 5, "Constructs an L-BFGS "
        "learner with given parameters");
}

void bind_batch_expgd(pybind11::module &m) {
  class PyExpoGD : public ExpoGD {
    public:
      using ExpoGD::ExpoGD;

      void fit(const std::vector<std::vector<double>> &trains,
          const std::vector<std::vector<double>> &valids,
          SPNetwork &spn, bool verbose = false) override {
        PYBIND11_OVERLOAD(void, ExpoGD, fit, trains, valids, spn, verbose);
      }
  };

  py::class_<ExpoGD, BatchParamLearning, PyExpoGD>(m, "Exponentiated gradient descent (EGD) batch learner")
    .def(py::init<>(), "Constructs an EGD learner with default parameters")
    .def(py::init<int, double, double, double>(), py::arg("num_iters") = 50,
        py::arg("stop_thred") = 1e-3, py::arg("lrate") = 1e-1, py::arg("shrink_weight") = 8e-1,
        "Constructs an EGD learner with given parameters");
}

void bind_batch_sma(pybind11::module &m) {
  class PySMA : public SMA {
    public:
      using SMA::SMA;

      void fit(const std::vector<std::vector<double>> &trains,
          const std::vector<std::vector<double>> &valids,
          SPNetwork &spn, bool verbose = false) override {
        PYBIND11_OVERLOAD(void, SMA, fit, trains, valids, spn, verbose);
      }
  };

  py::class_<SMA, BatchParamLearning, PySMA>(m, "Sequential monomial approximation (SMA) batch learner")
    .def(py::init<>(), "Constructs an SMA learner with default parameters")
    .def(py::init<int, double, double, double>(), py::arg("num_iters") = 50,
        py::arg("stop_thred") = 1e-3, py::arg("lrate") = 1e-1, py::arg("shrink_weight") = 8e-1,
        "Constructs an SMA learner with given parameters");
}

