#include "../src/SPNetwork.h"

#include <iostream>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>

#include "network.h"

using namespace SPN;

namespace py = pybind11;

void bind_net_spnetwork(py::module &m) {
  py::class_<SPNetwork>(m, "SPNetwork", "A sum-product network class with basic network functions")
    .def(py::init<SPNNode*>(), py::return_value_policy::reference, py::keep_alive<1, 2>(),
        py::arg("root"), "Constructs a sum-product network given a SPN root node")
    .def("size", &SPNetwork::size, "Returns network size")
    .def("height", &SPNetwork::height, "Returns network height")
    .def("num_nodes", &SPNetwork::num_nodes, "Returns number of nodes in network")
    .def("num_edges", &SPNetwork::num_edges, "Returns number of edges in network")
    .def("num_var_nodes", &SPNetwork::num_var_nodes, "Returns number of leaf nodes in network")
    .def("num_sum_nodes", &SPNetwork::num_sum_nodes, "Returns number of sum nodes in network")
    .def("num_prod_nodes", &SPNetwork::num_prod_nodes, "Returns number of product nodes in network")
    .def("inference", (double (SPNetwork::*)(const std::vector<double>&, bool)) &SPNetwork::inference,
        py::arg("input"), py::arg("verbose") = false, "Computes marginal inference given an instantiation")
    .def("inference", (std::vector<double> (SPNetwork::*)(const std::vector<std::vector<double>>&, bool))
        &SPNetwork::inference, py::arg("inputs"), py::arg("verbose") = false, "Computes marginal "
        "inference given a dataset")
    .def("logprob", (double (SPNetwork::*)(const std::vector<double>&, bool)) &SPNetwork::logprob,
        py::arg("input"), py::arg("verbose") = false, "Computes the probability of the SPN in "
        "logarithmic space given a single instantiation")
    .def("logprob", (std::vector<double> (SPNetwork::*)(const std::vector<std::vector<double>>&, bool))
        &SPNetwork::logprob, py::arg("inputs"), py::arg("verbose") = false, "Computes the "
        "probabilty of the SPN in logarithmic space given a dataset")
    .def("bottom_up_order", &SPNetwork::bottom_up_order, py::return_value_policy::reference,
        "Returns nodes of this SPN in post-order")
    .def("top_down_order", &SPNetwork::top_down_order, py::return_value_policy::reference,
        "Returns nodes of this SPN in pre-order")
    .def("dist_nodes", &SPNetwork::dist_nodes, py::return_value_policy::reference,
        "Returns all leaf nodes of this SPN")
    .def("EvalDiff", &SPNetwork::EvalDiff, py::arg("input"), py::arg("mask"),
        "Set the fr and dr values at each node with input x and return the log-probability of the "
        "input vector, mask is used to indicate whether the corresponding feature should be "
        "integrated/marginalized out or not. If mask[i] = true, then the ith feature will be "
        "integrated out, otherwise not.")
    .def("init", &SPNetwork::init, "Initialize the SPN, do the following tasks:\n1. Remove "
        "connected sum nodes and product nodes\n2. Compute statistics about the network topology\n"
        "3. Build the bottom-up and top-down visiting order of nodes in SPN")
    .def("set_random_params", &SPNetwork::set_random_params, py::arg("seed"), "Drop the existing "
        "model parameters and initialize using random seed")
    .def("weight_projection", &SPNetwork::weight_projection, py::arg("smooth") = 0.0, "Project "
        "each nonlocally normalized SPN into an SPN with locally normalized weights")
    .def("print", (void (SPNetwork::*)(void)) &SPNetwork::print, "Output the network to stdout")
    .def("sample", (std::vector<double> (SPNetwork::*)(const std::vector<double>&)) &SPNetwork::sample,
        py::arg("input"), "Sample an instantiation from this SPN's distribution, where NaNs are "
        "replaced with a sample from the SPN")
    .def("sample", (std::vector<std::vector<double>> (SPNetwork::*)(const std::vector<std::vector<double>>&))
        &SPNetwork::sample, py::arg("inputs"), "Sample a dataset from this SPN's distribution, "
        "where NaNs are replaced with a sample from the SPN");
}
