#include "../src/SPNNode.h"

#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "node.h"

using namespace SPN;

namespace py = pybind11;

void bind_spn_enums(py::module &m) {
  py::enum_<SPNNodeType>(m, "SPNNodeType", "SPN node type")
    .value("SUMNODE", SPNNodeType::SUMNODE, "Sum node")
    .value("PRODNODE", SPNNodeType::PRODNODE, "Product node")
    .value("VARNODE", SPNNodeType::VARNODE, "Leaf node")
    .export_values();

  py::enum_<VarNodeType>(m, "VarNodeType", "Leaf node type")
    .value("BINNODE", VarNodeType::BINNODE, "Binary/literal node")
    .value("NORMALNODE", VarNodeType::NORMALNODE, "Gaussian node")
    .value("TOPNODE", VarNodeType::TOPNODE, "Top node")
    .value("BOTNODE", VarNodeType::BOTNODE, "Bottom node")
    .value("BERNOULLINODE", VarNodeType::BERNOULLINODE, "Bernoulli node")
    .export_values();
}

void bind_spn_node(py::module &m) {
  class PySPNNode : public SPNNode {
    public:
      using SPNNode::SPNNode;

      SPNNodeType type(void) const override { PYBIND11_OVERLOAD_PURE(SPNNodeType, SPNNode, type, ); }
      std::string type_string(void ) const override { PYBIND11_OVERLOAD_PURE(std::string, SPNNode, type_string, ); }
      std::string string(void) const override { PYBIND11_OVERLOAD_PURE(std::string, SPNNode, string, ); }

    protected:
      std::pair<int, double> sample(double fr) override {
        PYBIND11_OVERLOAD_PURE(PYBIND11_TYPE(std::pair<int, double>), SPNNode, sample, fr);
      }
  };

  py::class_<SPNNode, PySPNNode>(m, "SPNNode", "An SPN generic node")
    .def(py::init<int>(), py::arg("id"), "Constructs a generic SPN node with given ID")
    .def(py::init<int, const std::vector<int>&>(), py::arg("id"), py::arg("scope"), "Constructs a "
        "generic SPN node with given ID and scope.")
    .def("id", &SPNNode::id, "Get SPN node ID")
    .def("num_parents", &SPNNode::num_parents, "Number of parents")
    .def("num_children", &SPNNode::num_children, "Number of children")
    .def("scope", &SPNNode::scope, "Scope of node")
    .def("children", &SPNNode::children, py::return_value_policy::reference, "Children of node")
    .def("parents", &SPNNode::parents, py::return_value_policy::reference, "Parents of node")
    .def("fr", (double (SPNNode::*)(void) const) &SPNNode::fr, "Get forward pass inference")
    .def("fr", (void (SPNNode::*)(double)) &SPNNode::fr, py::arg("v"), "Set forward pass inference")
    .def("dr", (double (SPNNode::*)(void) const) &SPNNode::dr, "Get differential")
    .def("dr", (void (SPNNode::*)(double)) &SPNNode::dr, py::arg("v"), "Set differential")
    .def("add", (void (SPNNode::*)(SPNNode*)) &SPNNode::add, py::arg("child"), py::keep_alive<1, 2>(), "Adds a child to node")
    .def("add", (void (SPNNode::*)(const std::vector<SPNNode*>&)) &SPNNode::add, py::arg("children"), py::keep_alive<1, 2>(), "Adds children to node")
    .def("remove", &SPNNode::remove, py::arg("child"), "Removes child from node")
    .def("set_children", &SPNNode::set_children, py::arg("children"), py::keep_alive<1, 2>(), "Sets the children of node")
    .def("set_parents", &SPNNode::set_parents, py::arg("parents"), py::keep_alive<1, 2>(), "Sets the parents of node")
    .def("add_to_scope", &SPNNode::add_to_scope, py::arg("t"), "Adds variable to scope")
    .def("clear_scope", &SPNNode::clear_scope, "Clears node scope")
    .def("type", &SPNNode::type, "Node type")
    .def("type_string", &SPNNode::type_string, "Node type in string format")
    .def("string", &SPNNode::string, "String node representation");
}

void bind_spn_sum(py::module &m) {
  class PySumNode : public SumNode {
    public:
      using SumNode::SumNode;

      SPNNodeType type(void) const override { PYBIND11_OVERLOAD(SPNNodeType, SumNode, type, ); }
      std::string type_string(void) const override { PYBIND11_OVERLOAD(std::string, SumNode, type_string, ); }
      std::string string(void) const override { PYBIND11_OVERLOAD(std::string, SumNode, string, ); }

    protected:
      std::pair<int, double> sample(double fr) override { PYBIND11_OVERLOAD(PYBIND11_TYPE(std::pair<int, double>), SumNode, sample, fr); }
  };

  py::class_<SumNode, SPNNode, PySumNode>(m, "SumNode", "An SPN sum node")
    .def(py::init<int>(), py::arg("id"), "Constructs a sum node with given ID")
    .def(py::init<int, const std::vector<int>&, const std::vector<double>&>(), py::arg("id"),
        py::arg("scope"), py::arg("weights"), "Constructs a sum node with given scope and array of weights")
    .def("weights", &SumNode::weights, "Returns the array of weights")
    .def("set_weights", &SumNode::set_weights, py::arg("weights"), "Sets the weights of this sum node")
    .def("set_weight", &SumNode::set_weight, py::arg("index"), py::arg("w"), "Sets a weight of this sum node")
    .def("remove_weight", &SumNode::remove_weight, py::arg("i"), "Removes the given weight")
    .def("add_weight", &SumNode::add_weight, py::arg("w"), "Adds a weight to this sum node");
}

void bind_spn_prod(py::module &m) {
  class PyProdNode : public ProdNode {
    public:
      using ProdNode::ProdNode;

      SPNNodeType type(void) const override { PYBIND11_OVERLOAD(SPNNodeType, ProdNode, type, ); }
      std::string type_string(void) const override { PYBIND11_OVERLOAD(std::string, ProdNode, type_string, ); }
      std::string string(void) const override { PYBIND11_OVERLOAD(std::string, ProdNode, string, ); }

    protected:
      std::pair<int, double> sample(double fr) override { PYBIND11_OVERLOAD(PYBIND11_TYPE(std::pair<int, double>), ProdNode, sample, fr); }
  };

  py::class_<ProdNode, SPNNode, PyProdNode>(m, "ProdNode", "An SPN product node")
    .def(py::init<int>(), "Constructs a product node with given ID")
    .def(py::init<int, const std::vector<int>&>(), "Constructs a product node with given ID and scope");
}

void bind_spn_var(py::module &m) {
  class PyVarNode : public VarNode {
    public:
      using VarNode::VarNode;

      SPNNodeType type(void) const override { PYBIND11_OVERLOAD(SPNNodeType, VarNode, type, ); }
      std::string type_string(void) const override { PYBIND11_OVERLOAD(std::string, VarNode, type_string, ); }
      std::string string(void) const override { PYBIND11_OVERLOAD_PURE(std::string, VarNode, string, ); }

      VarNodeType distribution(void) const override { PYBIND11_OVERLOAD_PURE(VarNodeType, VarNode, distribution, ); }
      double prob(double x) const override { PYBIND11_OVERLOAD_PURE(double, VarNode, prob, x); }
      double log_prob(double x) const override { PYBIND11_OVERLOAD_PURE(double, VarNode, log_prob, x); }
      size_t num_param(void) const override { PYBIND11_OVERLOAD_PURE(size_t, VarNode, num_param, ); }

    protected:
      std::pair<int, double> sample(double fr) override {
        PYBIND11_OVERLOAD_PURE(PYBIND11_TYPE(std::pair<int, double>), VarNode, sample, fr);
      }
  };

  py::class_<VarNode, SPNNode, PyVarNode>(m, "VarNode", "An SPN leaf node")
    .def(py::init<int, int>(), py::arg("id"), py::arg("var_index"), "Constructs a leaf node with "
        "given ID and variable index")
    .def("var_index", &VarNode::var_index, "Returns variable index")
    .def("distribution", &VarNode::distribution, "Returns the type of distribution of this leaf")
    .def("prob", &VarNode::prob, py::arg("x"), "Computes the probability density or mass")
    .def("log_prob", &VarNode::log_prob, py::arg("x"), "Computes the log probability density or mass")
    .def("num_param", &VarNode::num_param, "Returns the number of natural statistics of distribution");
}

void bind_spn_bin(py::module &m) {
  class PyBinNode : public BinNode {
    public:
      using BinNode::BinNode;

      std::string type_string(void) const override { PYBIND11_OVERLOAD(std::string, BinNode, type_string, ); }
      std::string string(void) const override { PYBIND11_OVERLOAD(std::string, BinNode, string, ); }

      VarNodeType distribution(void) const override { PYBIND11_OVERLOAD(VarNodeType, BinNode, distribution, ); }
      size_t num_param(void) const override { PYBIND11_OVERLOAD(size_t, BinNode, num_param, ); }
      double prob(double x) const override { PYBIND11_OVERLOAD(double, BinNode, prob, x); }
      double log_prob(double x) const override { PYBIND11_OVERLOAD(double, BinNode, log_prob, x); }

    protected:
      std::pair<int, double> sample(double fr) override { PYBIND11_OVERLOAD(PYBIND11_TYPE(std::pair<int, double>), BinNode, sample, fr); }
  };

  py::class_<BinNode, VarNode, PyBinNode>(m, "BinNode", "A binary/indicator/literal leaf node")
    .def(py::init<int, int, double>(), py::arg("id"), py::arg("var_index"), py::arg("var_value"),
        "Constructs a binary/indicator/literal leaf node given ID, variable index and the value "
        "the variable should take")
    .def("var_value", &BinNode::var_value, "Returns the value the indicator function agrees with");
}

void bind_spn_top(py::module &m) {
  class PyTopNode : public TopNode {
    public:
      using TopNode::TopNode;

      std::string type_string(void) const override { PYBIND11_OVERLOAD(std::string, TopNode, type_string, ); }
      std::string string(void) const override { PYBIND11_OVERLOAD(std::string, TopNode, string, ); }

      VarNodeType distribution(void) const override { PYBIND11_OVERLOAD(VarNodeType, TopNode, distribution, ); }
      size_t num_param(void) const override { PYBIND11_OVERLOAD(size_t, TopNode, num_param, ); }
      double prob(double x) const override { PYBIND11_OVERLOAD(double, TopNode, prob, x); }
      double log_prob(double x) const override { PYBIND11_OVERLOAD(double, TopNode, log_prob, x); }

    protected:
      std::pair<int, double> sample(double fr) override { PYBIND11_OVERLOAD(PYBIND11_TYPE(std::pair<int, double>), TopNode, sample, fr); }
  };

  py::class_<TopNode, VarNode, PyTopNode>(m, "TopNode", "A top (always true) leaf node")
    .def(py::init<int, int>(), py::arg("id"), py::arg("var_index"), "Constructs a top node given "
        "ID and variable index");
}

void bind_spn_bot(py::module &m) {
  class PyBotNode : public BotNode {
    public:
      using BotNode::BotNode;

      std::string type_string(void) const override { PYBIND11_OVERLOAD(std::string, BotNode, type_string, ); }
      std::string string(void) const override { PYBIND11_OVERLOAD(std::string, BotNode, string, ); }

      VarNodeType distribution(void) const override { PYBIND11_OVERLOAD(VarNodeType, BotNode, distribution, ); }
      size_t num_param(void) const override { PYBIND11_OVERLOAD(size_t, BotNode, num_param, ); }
      double prob(double x) const override { PYBIND11_OVERLOAD(double, BotNode, prob, x); }
      double log_prob(double x) const override { PYBIND11_OVERLOAD(double, BotNode, log_prob, x); }

    protected:
      std::pair<int, double> sample(double fr) override { PYBIND11_OVERLOAD(PYBIND11_TYPE(std::pair<int, double>), BotNode, sample, fr); }
  };

  py::class_<BotNode, VarNode, PyBotNode>(m, "BotNode", "A bot (always false) leaf node")
    .def(py::init<int, int>(), py::arg("id"), py::arg("var_index"), "Constructs a bot node given "
        "ID and variable index");
}

void bind_spn_bernoulli(py::module &m) {
  class PyBernoulliNode : public BernoulliNode {
    public:
      using BernoulliNode::BernoulliNode;

      std::string type_string(void) const override { PYBIND11_OVERLOAD(std::string, BernoulliNode, type_string, ); }
      std::string string(void) const override { PYBIND11_OVERLOAD(std::string, BernoulliNode, string, ); }

      VarNodeType distribution(void) const override { PYBIND11_OVERLOAD(VarNodeType, BernoulliNode, distribution, ); }
      size_t num_param(void) const override { PYBIND11_OVERLOAD(size_t, BernoulliNode, num_param, ); }
      double prob(double x) const override { PYBIND11_OVERLOAD(double, BernoulliNode, prob, x); }
      double log_prob(double x) const override { PYBIND11_OVERLOAD(double, BernoulliNode, log_prob, x); }

    protected:
      std::pair<int, double> sample(double fr) override { PYBIND11_OVERLOAD(PYBIND11_TYPE(std::pair<int, double>), BernoulliNode, sample, fr); }
  };

  py::class_<BernoulliNode, VarNode, PyBernoulliNode>(m, "BernoulliNode", "A Bernoulli "
      "distribution leaf node")
    .def(py::init<int, int, double>(), py::arg("id"), py::arg("var_index"), py::arg("p"),
        "Constructs a Bernoulli distribution leaf node given ID, variable index and parameter p")
    .def("p", &BernoulliNode::p, "Returns the parameter p of the underlying Bernoulli")
    .def("set_p", &BernoulliNode::set_p, py::arg("p"), "Sets the parameter p of the underlying Bernoulli");
}

void bind_spn_normal(py::module &m) {
  class PyNormalNode : public NormalNode {
    public:
      using NormalNode::NormalNode;

      std::string type_string(void) const override { PYBIND11_OVERLOAD(std::string, NormalNode, type_string, ); }
      std::string string(void) const override { PYBIND11_OVERLOAD(std::string, NormalNode, string, ); }

      VarNodeType distribution(void) const override { PYBIND11_OVERLOAD(VarNodeType, NormalNode, distribution, ); }
      size_t num_param(void) const override { PYBIND11_OVERLOAD(size_t, NormalNode, num_param, ); }
      double prob(double x) const override { PYBIND11_OVERLOAD(double, NormalNode, prob, x); }
      double log_prob(double x) const override { PYBIND11_OVERLOAD(double, NormalNode, log_prob, x); }

    protected:
      std::pair<int, double> sample(double fr) override { PYBIND11_OVERLOAD(PYBIND11_TYPE(std::pair<int, double>), NormalNode, sample, fr); }
  };

  py::class_<NormalNode, VarNode, PyNormalNode>(m, "NormalNode", "A Gaussian distribution leaf node")
    .def(py::init<int, int, double, double>(), py::arg("id"), py::arg("var_index"),
        py::arg("var_mean"), py::arg("var_var"), "Constructs a Gaussian distribution leaf node "
        "given ID, variable index, variable mean and variance")
    .def("var_mean", &NormalNode::var_mean, "Returns the Gaussian mean")
    .def("var_var", &NormalNode::var_var, "Returns the Gaussian variance")
    .def("set_var_mean", &NormalNode::set_var_mean, py::arg("var_mean"), "Sets the Gaussian mean")
    .def("set_var_var", &NormalNode::set_var_var, py::arg("var_var"), "Sets the Gaussian variance");
}
