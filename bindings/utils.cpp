#include <string>

#include <pybind11/pybind11.h>

#include "../src/SPNetwork.h"
#include "../src/utils.h"

using namespace SPN;

namespace py = pybind11;

void bind_utils(py::module &m) {
  m.def("load", &utils::load, py::arg("filename"), "Loads an SPN from file");
  m.def("save", &utils::save, py::arg("spn"), py::arg("filename"), "Saves an SPN to file");
  m.def("load_data", &utils::load_data, py::arg("filename"), "Loads a dataset from file");
}
