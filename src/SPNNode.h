//
// Created by Han Zhao on 5/6/15.
//

#ifndef SPN_SL_SPNNODE_H
#define SPN_SL_SPNNODE_H

#include "fmath.hpp"

#include <vector>
#include <string>
#include <cmath>
#include <limits>
#include <numeric>
#include <iostream>
#include <cassert>
#include <unordered_map>
#include <algorithm>
#include <utility>

#include <boost/math/constants/constants.hpp>

#include "random.h"

using fmath::log;
using fmath::exp;

namespace SPN {

    enum class SPNNodeType {
        SUMNODE, PRODNODE, VARNODE
    };

    enum class VarNodeType {
        BINNODE, // Binary (Literal) leaf distribution
        NORMALNODE,  // Normal leaf distribution
        TOPNODE, // Top leaf distribution
        BOTNODE, // Bottom leaf distribution
        BERNOULLINODE // Bernoulli leaf distribution
    };

    class SPNNode {
    public:
        // Constructors and destructors
        SPNNode() = default;

        virtual ~SPNNode() = default;

        SPNNode(int id_) : id_(id_) { }

        SPNNode(int id_, const std::vector<int> &scope_) : id_(id_), scope_(scope_) { }

        // Getter
        int id() const { return id_; }

        size_t num_parents() const { return parents_.size(); }

        size_t num_children() const { return children_.size(); }

        const std::vector<int> &scope() const { return scope_; }

        const std::vector<SPNNode *> &children() const { return children_; }

        const std::vector<SPNNode *> &parents() const { return parents_; }

        double fr() const { return fr_; }

        double dr() const { return dr_; }

        // Setter
        void fr(double v) { fr_ = v; }

        void dr(double v) { dr_ = v; }

        inline void add(SPNNode *child) {
            add_child(child);
            child->add_parent(this);
        }

        inline void add(const std::vector<SPNNode*> &children) {
            add_children(children);
            for (auto c : children) c->add_parent(this);
        }

        inline void remove(SPNNode *child) {
            remove_child(child);
            child->remove_parent(this);
        }

        // Shared methods from SPNNode for SumNode, ProdNode and VarNode
        inline void add_child(SPNNode *child) {
            children_.push_back(child);
        }

        // Deprecated
        inline void add_parent(SPNNode *parent) {
            parents_.push_back(parent);
        }

        inline void add_children(const std::vector<SPNNode *> &childs) {
            children_.insert(children_.end(), childs.begin(), childs.end());
        }

        // Deprecated
        inline void add_parents(const std::vector<SPNNode *> &parents) {
            parents_.insert(parents_.end(), parents.begin(), parents.end());
        }

        inline void set_children(const std::vector<SPNNode *> &childs) {
            children_ = childs;
        }

        inline void set_parents(const std::vector<SPNNode *> &parents) {
            parents_ = parents;
        }

        inline void remove_child(SPNNode *child) {
            children_.erase(std::remove(children_.begin(), children_.end(), child),
                            children_.end());
        }

        // Deprecated
        inline void remove_parent(SPNNode *parent) {
            parents_.erase(std::remove(parents_.begin(), parents_.end(), parent),
                           parents_.end());
        }

        inline void add_to_scope(int t) {
            scope_.push_back(t);
        }

        inline void clear_scope() {
            scope_.clear();
        }

        // Declare concrete type.
        virtual SPNNodeType type() const = 0;

        // Type string.
        virtual std::string type_string() const = 0;

        // Print literal string.
        virtual std::string string() const = 0;

        // Friend class
        friend class SPNetwork;

    protected:
        // Sample this node's underlying distribution. Assumes forward pass has already been
        // computed.
        virtual std::pair<int, double> sample(double) = 0;

        // For tracking each node in the SPN
        int id_ = -1;
        // DAG topology recording
        std::vector<SPNNode *> children_;
        std::vector<SPNNode *> parents_;
        // Scope
        std::vector<int> scope_;
        // Evaluation value at input x.
        double fr_ = 0.0;
        // Differentiation value at input x.
        double dr_ = 0.0;
    };


    class SumNode : public SPNNode {
    public:
        // Constructors and destructors
        SumNode() = default;

        virtual ~SumNode() = default;

        SumNode(int id_) : SPNNode(id_) { }

        SumNode(int id_, const std::vector<int> &scope_,
                const std::vector<double> &weights_) : SPNNode(id_, scope_), weights_(weights_) {
            for (size_t i = 0; i < weights_.size(); ++i) {
                values_.push_back(0.0);
            }
        }

        // Additional Getters
        const std::vector<double> &weights() const { return weights_; }

        // Type method
        SPNNodeType type() const override {
            return SPNNodeType::SUMNODE;
        }

        // Type string representation
        std::string type_string() const override {
            return std::string("SumNode");
        }

        // Print literal string.
        std::string string() const override;

        // Setter methods for weights
        inline void set_weights(const std::vector<double> &weights) {
            weights_ = weights;
        }

        inline void set_weight(size_t index, double w) {
            weights_[index] = w;
        }

        // Remove the ith component in the weight vector
        inline void remove_weight(size_t i) {
            weights_.erase(weights_.begin() + i);
        }

        // Add weight
        inline void add_weight(double w) {
            weights_.push_back(w);
            values_.push_back(0.0);
        }
        //
        // Friend function for output
        friend std::ostream &operator<<(std::ostream &, const SumNode &);

        friend class SPNetwork;

    protected:
        std::pair<int, double> sample(double) override;

    private:
        // Avoid the underflow problem in computing the log-probability on a sum node
        // @param log_probs: the log of the probabilities returned by children of the sum node
        inline double weighted_log_sum(std::vector<double> &log_probs) const {
            assert(weights_.size() == log_probs.size());
            double max_logprob = *std::max_element(log_probs.begin(), log_probs.end());
            std::transform(log_probs.begin(), log_probs.end(), log_probs.begin(),
                           [max_logprob](double t) {
                               return exp(t - max_logprob);
                           });
            double sum_exp = 0.0;
            for (size_t i = 0; i < weights_.size(); ++i)
                sum_exp += weights_[i] * log_probs[i];
            return max_logprob + log(sum_exp);
        }

        inline double weighted_log_sum(double *log_probs, size_t size) const {
            double max_logprob = -std::numeric_limits<double>::infinity();
            for (size_t i = 0; i < size; ++i)
                if (log_probs[i] > max_logprob)
                    max_logprob = log_probs[i];
            for (size_t i = 0; i < size; ++i)
                log_probs[i] = exp(log_probs[i] - max_logprob);
            double sum_exp = 0.0;
            for (size_t i = 0; i < size; ++i)
                sum_exp += weights_[i] * log_probs[i];
            return max_logprob + log(sum_exp);
        }

        // Connection weights
        std::vector<double> weights_;

    public:
        // Associated information of each sum node. This is created for the convenience of
        // implementing the streaming learning algorithm.
        std::vector<double> values_;
    };

    class ProdNode : public SPNNode {
    public:
        // Constructors and destructors
        ProdNode() = default;

        virtual ~ProdNode() = default;

        ProdNode(int id_) : SPNNode(id_) { }

        ProdNode(int id_, const std::vector<int> &scope_) : SPNNode(id_, scope_) { }

        // Declare type
        SPNNodeType type() const override {
            return SPNNodeType::PRODNODE;
        }

        // Type string representation
        std::string type_string() const override {
            return std::string("ProdNode");
        }

        // Print literal string.
        std::string string() const override;

        // Friend function for output
        friend std::ostream &operator<<(std::ostream &, const ProdNode &);

        friend class SPNetwork;

    protected:
        inline std::pair<int, double> sample(double) override { return {children_.size()+2, 0}; }

    };


    class VarNode : public SPNNode {
    public:
        // Constructors and destructors
        VarNode() = default;

        virtual ~VarNode() = default;

        VarNode(int id, int var_index) : SPNNode(id), var_index_(var_index) {
            scope_.push_back(var_index);
        }

        // Addtional Getters
        int var_index() const { return var_index_; }

        // Declare type
        SPNNodeType type() const override {
            return SPNNodeType::VARNODE;
        }

        // Type string representation
        std::string type_string() const override {
            return std::string("VarNode");
        }

        // Declare distribution
        virtual VarNodeType distribution() const = 0;

        // Compute probability density/mass.
        virtual double prob(double x) const = 0;

        // Compute log probability density/mass.
        virtual double log_prob(double x) const = 0;

        // Number of natural statistics in the canonical form of the
        // exponential family distribution.
        virtual size_t num_param() const = 0;

        // Friend function for output
        friend std::ostream &operator<<(std::ostream &, const VarNode &);

        friend class SPNetwork;
    protected:
        int var_index_;
    };

    class BinNode : public VarNode {
    public:
        // Default constructor and destructor.
        BinNode() = default;
        virtual ~BinNode() = default;

        // Constructor to set the index of the variable and the value of
        // the binary point mass.
        BinNode(int id, int var_index, double var_value) :
                VarNode(id, var_index), var_value_(var_value) {
        }

        VarNodeType distribution() const override {
            return VarNodeType::BINNODE;
        }

        std::string type_string() const override {
            return std::string("BinNode");
        }

        // The number of sufficient statistics for point pass distribution
        // is 0.
        size_t num_param() const override {
            return 0;
        }

        // Print literal string.
        std::string string() const override;

        double var_value() const {
            return var_value_;
        }

        // Compute the probability mass.
        double prob(double x) const override {
            if (fabs(x - var_value_) < 1e-6) {
                return 1.0;
            } else return 0.0;
        }

        double log_prob(double x) const override {
            if (fabs(x - var_value_) < 1e-6) {
                return 0.0;
            } else return -std::numeric_limits<double>::infinity();
        }

        // Friend function for output
        friend std::ostream &operator<<(std::ostream &, const BinNode &);

        friend class SPNetwork;

    protected:

        inline std::pair<int, double> sample(double) override { return {-var_index_, var_value_}; }

    private:
        // Value of the point taken by the binary random variable, either 0 or 1.
        double var_value_;
    };

    class TopNode : public VarNode {
    public:
        TopNode() = default;
        virtual ~TopNode() = default;

        TopNode(int id, int var_index) : VarNode(id, var_index) {}

        VarNodeType distribution() const override { return VarNodeType::TOPNODE; }

        std::string type_string() const override { return std::string("TopNode"); }

        size_t num_param() const override { return 0; }

        std::string string() const override { return std::string("T"); }

        double prob(double) const override { return 1.0; }

        double log_prob(double) const override { return 0.0; }

        friend std::ostream &operator<<(std::ostream&, const TopNode&);
        friend class SPNetwork;

    protected:

        inline std::pair<int, double> sample(double) override { return {-var_index_, 0.0}; }
    };

    class BotNode : public VarNode {
    public:
        BotNode() = default;
        virtual ~BotNode() = default;

        BotNode(int id, int var_index) : VarNode(id, var_index) {}

        VarNodeType distribution() const override { return VarNodeType::BOTNODE; }

        std::string type_string() const override { return std::string("BotNode"); }

        size_t num_param() const override { return 0; }

        std::string string() const override { return std::string("F"); }

        double prob(double) const override { return 0.0; }

        double log_prob(double) const override { return -std::numeric_limits<double>::infinity(); }

        friend std::ostream &operator<<(std::ostream&, const BotNode&);
        friend class SPNetwork;

    protected:

        inline std::pair<int, double> sample(double) override {
            return {-var_index_, -std::numeric_limits<double>::infinity()};
        }
    };

    class BernoulliNode : public VarNode {
    public:
        BernoulliNode() = default;
        virtual ~BernoulliNode() = default;

        BernoulliNode(int id, int var_index, double p) : VarNode(id, var_index), p_(p) {}

        VarNodeType distribution() const override { return VarNodeType::BERNOULLINODE; }

        std::string type_string() const override { return std::string("BernoulliNode"); }

        size_t num_param() const override { return 1; }

        std::string string() const override;

        inline double p() const { return p_; }

        inline void set_p(double p) { p_ = p; }

        double prob(double x) const override { return fabs(x) < 1e-6 ? p_ : 1.0 - p_; }

        double log_prob(double x) const override { return fabs(x) < 1e-6 ? log(p_) : log(1.0 - p_); }

        friend std::ostream &operator<<(std::ostream&, const BernoulliNode&);
        friend class SPNetwork;

    protected:

        inline std::pair<int, double> sample(double) override {
            return {-var_index_, p_ > (((double) rand()) / ((double) RAND_MAX)) ? 1.0 : 0.0};
        }

    private:
        double p_;
    };

    class NormalNode : public VarNode {
    public:
        // Default constructor and destructor.
        NormalNode() = default;
        virtual ~NormalNode() = default;

        // Constructor to set the mean and the variance of the normal distribution.
        NormalNode(int id, int var_index, double var_mean, double var_var) :
                VarNode(id, var_index), var_mean_(var_mean), var_var_(var_var) {
        }

        VarNodeType distribution() const override {
            return VarNodeType::NORMALNODE;
        }

        std::string type_string() const override {
            return std::string("NormalNode");
        }

        // Print literal string.
        std::string string() const override;

        // Getters and Setters.
        inline double var_mean() const {
            return var_mean_;
        }

        inline double var_var() const {
            return var_var_;
        }

        inline void set_var_mean(double var_mean) {
            var_mean_ = var_mean;
        }

        inline void set_var_var(double var_var) {
            var_var_ = var_var;
        }

        // The sufficient statistics for normal distribution is (x, x^2),
        // i.e., the first and the second order moments.
        size_t num_param() const override {
            return 2;
        }

        // Compute the probability and the log-probability
        double prob(double x) const override {
            return 1.0 / sqrt(2.0 * PI * var_var_) *
                    exp(-(x - var_mean_) * (x - var_mean_) / 2 / var_var_);
        }

        double log_prob(double x) const override {
            return -0.5 * log(2 * PI * var_var_)
                   - 0.5 / var_var_ * (x - var_mean_) * (x - var_mean_);
        }

        // Friend function for output
        friend std::ostream &operator<<(std::ostream &, const NormalNode &);

        friend class SPNetwork;

    protected:

        inline std::pair<int, double> sample(double) override {
            return {-var_index_, random::gaussian(var_mean_, var_var_)};
        }

    private:
        double var_mean_;
        double var_var_;
        static constexpr double PI = 3.1415926535;
    };
}

#endif //SPN_SL_SPNNODE_H
