#include <string>

#include <pybind11/pybind11.h>

#include "../src/SPNetwork.h"
#include "../src/utils.h"

using namespace SPN;

namespace py = pybind11;

void bind_utils(py::module &m) {
  m.def("load", &utils::load, "Loads an SPN from file");
  m.def("save", &utils::save, "Saves an SPN to file");
  m.def("load_data", &utils::load_data, "Loads a dataset from file");
}
