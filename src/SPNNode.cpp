//
// Created by Han Zhao on 5/6/15.
//
#include "SPNNode.h"

#include <cassert>
#include <sstream>
#include <limits>
#include <utility>

#include "random.h"

namespace SPN {
    // Friend functions for output
    std::ostream &operator<<(std::ostream &out, const SumNode &node) {
        out << node.string();
        return out;
    }

    std::ostream &operator<<(std::ostream &out, const ProdNode &node) {
        out << node.string();
        return out;
    }

    std::ostream &operator<<(std::ostream &out, const VarNode &node) {
        out << node.string();
        return out;
    }

    std::ostream &operator<<(std::ostream &out, const TopNode &node) {
        out << node.string();
        return out;
    }

    std::ostream &operator<<(std::ostream &out, const BotNode &node) {
        out << node.string();
        return out;
    }

    std::ostream &operator<<(std::ostream &out, const BernoulliNode &node) {
        out << node.string();
        return out;
    }

    std::string SumNode::string() const {
        std::stringstream out;
        out << "sum,";
        out << "id:" << id_ << ",";
        out << "children:[";
        size_t num_child = children_.size();
        if (num_child > 0) {
            for (size_t i = 0; i < num_child - 1; ++i)
                out << children_[i]->id() << ",";
            out << children_[num_child - 1]->id();
        }
        out << "],parents:[";
        size_t num_parent = parents_.size();
        if (num_parent > 0) {
            for (size_t i = 0; i < num_parent - 1; ++i)
                out << parents_[i]->id() << ",";
            out << parents_[num_parent - 1]->id();
        }
        out << "],scope:[";
        size_t size_scope = scope_.size();
        if (size_scope > 0) {
            for (size_t i = 0; i < size_scope - 1; ++i)
                out << scope_[i] << ",";
            out << scope_[size_scope - 1];
        }
        out << "],weights:[";
        if (num_child > 0) {
            for (size_t i = 0; i < num_child - 1; ++i)
                out << weights_[i] << ",";
            out << weights_[num_child - 1];
        }
        out << "]";
        return out.str();
    }

    std::string ProdNode::string() const {
        std::stringstream out;
        out << "product,";
        out << "id:" << id_ << ",";
        out << "children:[";
        size_t num_child = children_.size();
        if (num_child > 0) {
            for (size_t i = 0; i < num_child - 1; ++i)
                out << children_[i]->id() << ",";
            out << children_[num_child - 1]->id();
        }
        out << "],parents:[";
        size_t num_parent = parents_.size();
        if (num_parent > 0) {
            for (size_t i = 0; i < num_parent - 1; ++i)
                out << parents_[i]->id() << ",";
            out << parents_[num_parent - 1]->id();
        }
        out << "],scope:[";
        size_t size_scope = scope_.size();
        if (size_scope > 0) {
            for (size_t i = 0; i < size_scope - 1; ++i)
                out << scope_[i] << ",";
            out << scope_[size_scope - 1];
        }
        out << "]";
        return out.str();
    }

    std::string BinNode::string() const {
        std::stringstream out;
        out << "variable,";
        out << "id:" << id_ << ",";
        out << "children:[";
        size_t num_child = children_.size();
        if (num_child > 0) {
            for (size_t i = 0; i < num_child - 1; ++i)
                out << children_[i]->id() << ",";
            out << children_[num_child - 1]->id();
        }
        out << "],parents:[";
        size_t num_parent = parents_.size();
        if (num_parent > 0) {
            for (size_t i = 0; i < num_parent - 1; ++i)
                out << parents_[i]->id() << ",";
            out << parents_[num_parent - 1]->id();
        }
        out << "],scope:[";
        size_t size_scope = scope_.size();
        if (size_scope > 0) {
            for (size_t i = 0; i < size_scope - 1; ++i)
                out << scope_[i] << ",";
            out << scope_[size_scope - 1];
        }
        out << "],distribution:[";
        out << type_string() << "], value:[";
        out << var_value() << "]";
        return out.str();
    }

    std::string NormalNode::string() const {
        std::stringstream out;
        out << "variable,";
        out << "id:" << id_ << ",";
        out << "children:[";
        size_t num_child = children_.size();
        if (num_child > 0) {
            for (size_t i = 0; i < num_child - 1; ++i)
                out << children_[i]->id() << ",";
            out << children_[num_child - 1]->id();
        }
        out << "],parents:[";
        size_t num_parent = parents_.size();
        if (num_parent > 0) {
            for (size_t i = 0; i < num_parent - 1; ++i)
                out << parents_[i]->id() << ",";
            out << parents_[num_parent - 1]->id();
        }
        out << "],scope:[";
        size_t size_scope = scope_.size();
        if (size_scope > 0) {
            for (size_t i = 0; i < size_scope - 1; ++i)
                out << scope_[i] << ",";
            out << scope_[size_scope - 1];
        }
        out << "],distribution:[";
        out << type_string() << "], mean:[";
        out << var_mean() << "],variance:[";
        out << var_var() << "]";
        return out.str();
    }

    std::string BernoulliNode::string() const {
        std::stringstream out;
        out << "variable,id:" << id_ << ",children:[],parents:[";
        size_t num_parents = parents_.size();
        if (num_parents > 0) {
            for (size_t i = 0; i < num_parents - 1; ++i)
                out << parents_[i]->id() << ",";
            out << parents_[num_parents - 1]->id();
        }
        out << "],scope:[";
        size_t size_scope = scope_.size();
        if (size_scope > 0) {
            for (size_t i = 0; i < size_scope - 1; ++i)
                out << scope_[i] << ",";
            out << scope_[size_scope - 1];
        }
        out << "],distribution,[p:[" << p_ << "]]";
        return out.str();
    }

    std::pair<int, double> SumNode::sample(double fr) {
        size_t n = weights_.size();
        int imax = -1;
        double vmax = -std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < n; ++i) {
            double k = fr + weights_[i] + random::gumbel();
            if (k > vmax) {
                vmax = k;
                imax = i;
            }
        }
        return {imax+1, vmax};
    }
}

