//
// Created by Han Zhao on 5/7/15.
//
#include "SPNetwork.h"

#include <cstdio>
#include <cmath>

#include <set>
#include <stack>
#include <queue>
#include <random>

#include "mmath.h"

namespace SPN {
    // Constructor of SPN
    SPNetwork::SPNetwork(SPNNode *root) {
        root_ = root;
        init();
    }

    // Destructor of SPN
    SPNetwork::~SPNetwork() {
        // Let pybind11 destroy and garbage collect everything.

        //std::stack<SPNNode*> T, K;
        //std::vector<bool> V(num_nodes_);
        //T.push(root_);
        //K.push(root_);
        //while (!T.empty()) {
            //SPNNode *N = T.top();
            //T.pop();
            //V[N->id_] = true;
            //for (auto C : N->children_)
                //if (!V[C->id_]) {
                    //T.push(C);
                    //K.push(C);
                //}
        //}
        //while (!K.empty()) {
            //SPNNode *N = K.top();
            //K.pop();
            //for (auto C : N->children_) {
                //C->parents_.clear();
                //delete C;
            //}
            //N->children_.clear();
        //}
        //root_ = nullptr;
        //V.clear();
        //id2node_.clear();
        //forward_order_.clear();
        //backward_order_.clear();
        //dist_nodes_.clear();
    }

    double SPNetwork::inference(const std::vector<double> &input, bool verbose) {
        std::vector<std::vector<double>> inputs;
        inputs.push_back(input);
        const std::vector<double> &probs = inference(inputs, verbose);
        return probs[0];
    }

    // Batch mode method for inference
    std::vector<double> SPNetwork::inference(const std::vector<std::vector<double>> &inputs, bool) {
        size_t num_inputs = inputs.size();
        std::vector<bool> mask(inputs[0].size(), false);
        std::vector<double> probs;
        for (size_t n = 0; n < num_inputs; ++n) {
            probs.push_back(fmath::exp(EvalDiff(inputs[n], mask)));
        }
        return probs;
    }

    double SPNetwork::logprob(const std::vector<double> &input, bool verbose) {
        std::vector<std::vector<double>> inputs;
        inputs.push_back(input);
        const std::vector<double> &logps = logprob(inputs, verbose);
        return logps[0];
    }

    std::vector<double> SPNetwork::logprob(const std::vector<std::vector<double>> &inputs, bool) {
        size_t num_inputs = inputs.size();
        std::vector<bool> mask(inputs[0].size(), false);
        std::vector<double> logps;
        for (size_t n = 0; n < num_inputs; ++n) {
            logps.push_back(EvalDiff(inputs[n], mask));
        }
        return logps;
    }

    void SPNetwork::init() {
        // Recursive remove connected sum nodes and product nodes
        std::unordered_set<SPNNode *> visited;
        visited.insert(root_);
        condense_(root_, visited);
        // Re-build mappings and compute network statistics
        compute_statistics_();
        // Check structure
        assert(check_structure_());
        build_order_();
    }

    void SPNetwork::setup_parents_(void) {
        std::queue<SPNNode*> Q;
        std::unordered_set<SPNNode*> visited;
        Q.push(root_);
        visited.insert(root_);
        while (!Q.empty()) {
            SPNNode *N = Q.front();
            Q.pop();
            for (auto C : N->children_)
                if (visited.find(C) == visited.end()) {
                    //C->parents_.push_back(N);
                    C->add_parent(N);
                    Q.push(C);
                    visited.insert(C);
                }
        }
    }

    void SPNetwork::build_order_() {
        // Build top-down order and bottom-up order using topological ordering algorithm.
        std::queue<SPNNode *> active_list;
        // r_parents records the number of parents of a specific node during the topological
        // ordering algorithm.
        size_t r_parents[num_nodes_];
        for (size_t i = 0; i < num_nodes_; ++i) {
            r_parents[i] = id2node_[i]->num_parents();
        }
        SPNNode *pt = nullptr;
        active_list.push(root_);
        assert (r_parents[root_->id_] == 0);
        while (!active_list.empty()) {
            pt = active_list.front();
            backward_order_.push_back(pt);
            if (pt->type() == SPNNodeType::VARNODE) {
                dist_nodes_.push_back((VarNode *) pt);
            }
            for (SPNNode *child : pt->children()) {
                r_parents[child->id_] -= 1;
                if (r_parents[child->id_] == 0) {
                    active_list.push(child);
                }
            }
            active_list.pop();
        }
        assert (backward_order_.size() == num_nodes_);
        for (auto iter = backward_order_.rbegin(); iter != backward_order_.rend(); ++iter) {
            forward_order_.push_back(*iter);
        }
        // Using the forward order to build the scope of each node.
        for (auto node : forward_order_) {
            if (node->type() == SPNNodeType::SUMNODE) {
                assert(node->scope().size() == 0);
                for (int t : node->children()[0]->scope()) {
                    node->add_to_scope(t);
                }
            } else if (node->type() == SPNNodeType::PRODNODE) {
                assert(node->scope().size() == 0);
                for (SPNNode *child : node->children()) {
                    for (int t : child->scope())
                        node->add_to_scope(t);
                }
            } else {
                // SPNNodeType == VARNODE
                assert(node->scope().size() == 1);
            }
        }
        // Check
        std::cout << "Type of root node = " << root_->type_string() << std::endl;
        std::cout << "Size of scope = " << root_->scope().size() << ". Scope of the root node: " << std::endl;
        for (int t : root_->scope())
            std::cout << t << " ";
        std::cout << std::endl;
    }

    void SPNetwork::condense_(SPNNode *node, std::unordered_set<SPNNode *> &visited) {
        if (node->num_children() == 0) return;
        // Post processing
        for (const auto &child : node->children()) {
            // Condense unvisited children
            if (visited.find(child) == visited.end()) {
                visited.insert(child);
                condense_(child, visited);
            }
        }
        // All the sub-SPNs have been condensed
        size_t num_children = node->num_children();
        bool is_grandson = false;
        // Depends on whether current node is SumNode or not
        if (node->type() == SPNNodeType::SUMNODE) {
            double node_weight = 0.0, son_weight = 0.0;
            std::vector<SPNNode *> new_children;
            std::vector<double> new_weights;
            for (size_t i = 0; i < num_children; ++i) {
                if (node->children_[i]->type() == node->type()) {
                    node_weight = ((SumNode *) node)->weights_[i];
                    size_t num_grandsons = node->children_[i]->children_.size();
                    for (size_t j = 0; j < num_grandsons; ++j) {
                        // Check whether the grandson to be added already existed
                        is_grandson = false;
                        son_weight = ((SumNode *) node->children_[i])->weights_[j];
                        for (size_t k = 0; k < new_children.size(); ++k) {
                            if (new_children[k] == node->children_[i]->children_[j]) {
                                // the grandson has already been added, then update the weight directly
                                is_grandson = true;
                                new_weights[k] += node_weight * son_weight;
                                break;
                            }
                        }
                        if (is_grandson) continue;
                        new_children.push_back(node->children_[i]->children_[j]);
                        new_weights.push_back(node_weight * son_weight);
                        // Update parent list of newly added grandson
                        node->children_[i]->children_[j]->add_parent(node);
                    }
                    // Update parent list of children[i]
                    node->children_[i]->remove_parent(node);
                } else {
                    new_children.push_back(node->children_[i]);
                    new_weights.push_back(((SumNode *) node)->weights_[i]);
                }
                // If there is no parent of children_[i], delete it
                if (node->children_[i]->num_parents() == 0) {
                    for (SPNNode *grandson : node->children_[i]->children_)
                        grandson->remove_parent(node->children_[i]);
                    delete node->children_[i];
                }
            }
            // Reset children and weights
            node->set_children(new_children);
            ((SumNode *) node)->set_weights(new_weights);
        } else {
            // One main difference for product node is that we don't need to consider whether a grandson
            // has already been added or not due to the decomposability constraint at the product node.
            // We can make the following claim to simplify the code:
            // For any two branches i, j of a product node p, there is not any edge connection one node
            // from sub-SPN rooted at i to another node from sub-SPN rooted at j.
            std::vector<SPNNode *> new_children;
            for (size_t i = 0; i < num_children; ++i) {
                if (node->children_[i]->type() == node->type()) {
                    // Add all the grandsons to be new children and delete children_[i] from the parent list
                    // of grandsons
                    for (SPNNode *grandson : node->children_[i]->children_) {
                        new_children.push_back(grandson);
                        // Update parent list of grandson
                        grandson->add_parent(node);
                    }
                    // Delete node from the parent list of children_[i]
                    node->children_[i]->remove_parent(node);
                } else {
                    new_children.push_back(node->children_[i]);
                }
                // If current node has no parent, delete it
                if (node->children_[i]->num_parents() == 0) {
                    for (SPNNode *grandson : node->children_[i]->children_)
                        grandson->remove_parent(node->children_[i]);
                    delete node->children_[i];
                }
            }
            // Reset children
            node->set_children(new_children);
        }
    }

    // BFS
    void SPNetwork::compute_statistics_() {
        // Initialize
        size_ = 0;
        height_ = 0;
        num_nodes_ = 0;
        num_edges_ = 0;
        num_var_nodes_ = 0;
        num_sum_nodes_ = 0;
        num_prod_nodes_ = 0;
        id2node_.clear();

        int id = 0;
        std::unordered_set<SPNNode *> visited;
        std::queue<std::pair<SPNNode *, size_t>> forward;
        visited.insert(root_);
        forward.push(std::make_pair(root_, 0));
        switch (root_->type()) {
            case SPNNodeType::SUMNODE:
                num_sum_nodes_ += 1;
                break;
            case SPNNodeType::PRODNODE:
                num_prod_nodes_ += 1;
                break;
            case SPNNodeType::VARNODE:
                num_var_nodes_ += 1;
                break;
        }
        num_nodes_ += 1;
        id2node_.insert({id, root_});
        size_ += 1;
        root_->id_ = id;
        id += 1;
        while (!forward.empty()) {
            auto pair = forward.front();
            forward.pop();
            for (SPNNode *child : pair.first->children()) {
                if (visited.find(child) == visited.end()) {
                    visited.insert(child);
                    forward.push(std::make_pair(child, pair.second + 1));
                    switch (child->type()) {
                        case SPNNodeType::SUMNODE:
                            num_sum_nodes_ += 1;
                            break;
                        case SPNNodeType::PRODNODE:
                            num_prod_nodes_ += 1;
                            break;
                        case SPNNodeType::VARNODE:
                            num_var_nodes_ += 1;
                            break;
                    }
                    num_nodes_ += 1;
                    id2node_.insert({id, child});
                    size_ += 1;
                    child->id_ = id;
                    id += 1;
                    height_ = std::max(height_, pair.second + 1);
                }
                num_edges_ += 1;
                size_ += 1;
            }
        }
        // Sanity check
        assert(num_nodes_ == num_var_nodes_ + num_sum_nodes_ + num_prod_nodes_);
        assert(size_ == num_nodes_ + num_edges_);
        assert(num_nodes_ == id2node_.size());
    }

    // Check that there are no connected nodes of the same type in SPN
    bool SPNetwork::check_structure_() {
        for (SPNNode *pt : bottom_up_order()) {
            for (SPNNode *child : pt->children()) {
                if (pt->type() == child->type()) return false;
            }
        }
        return true;
    }

    // Output
    void SPNetwork::print(std::ostream &out) {
        bfs_traverse([&](SPNNode *node) {
            out << node->string() << '\n';
        });
    }

    template<typename Callable>
    void SPNetwork::bfs_traverse(Callable &&callable) {
        std::queue<SPNNode *> forward;
        std::unordered_set<SPNNode *> visited;
        forward.push(root_);
        visited.insert(root_);
        while (!forward.empty()) {
            SPNNode *node = forward.front();
            forward.pop();
            for (SPNNode *child : node->children()) {
                if (visited.find(child) == visited.end()) {
                    forward.push(child);
                    visited.insert(child);
                }
            }
            callable(node);
        }
    }

    void SPNetwork::set_random_params(uint seed) {
        // Construct random number generator using seed.
        std::mt19937 gen(seed);
        std::uniform_real_distribution<double> dis(0.0, 1.0);
        for (SPNNode *pt : bottom_up_order()) {
            if (pt->type() == SPNNodeType::SUMNODE) {
                double nz = 0.0;
                for (size_t i = 0; i < pt->num_children(); ++i) {
                    ((SumNode *) pt)->set_weight(i, dis(gen));
                }
                for (size_t i = 0; i < pt->num_children(); ++i) {
                    nz += ((SumNode *) pt)->weights()[i];
                }
                // Renormalization
                for (size_t i = 0; i < pt->num_children(); ++i) {
                    ((SumNode *) pt)->set_weight(i, ((SumNode *) pt)->weights()[i] / nz);
                }
            }
        }
    }

    void SPNetwork::weight_projection(double smooth) {
        // First set the values of all the indicators to 1.
        std::vector<double> input(root_->scope_.size(), 0.0);
        std::vector<bool> mask(root_->scope_.size(), true);
        EvalDiff(input, mask);
        // Locally normalization.
        for (SPNNode *pt : top_down_order()) {
            if (pt->type() == SPNNodeType::SUMNODE) {
                // Projection w'_k <- w_k * S_k.
                double ssz = 0.0, sz, max_logp = -std::numeric_limits<double>::infinity();
                for (size_t k = 0; k < pt->num_children(); ++k) {
                    if (pt->children_[k]->fr_ > max_logp) {
                        max_logp = pt->children_[k]->fr_;
                    }
                }
                for (size_t k = 0; k < pt->num_children(); ++k)
                    ssz += ((SumNode*)pt)->weights_[k] * fmath::exp(pt->children_[k]->fr_ - max_logp) + smooth;
                // Local weight normalization.
                for (size_t k = 0; k < pt->num_children(); ++k) {
                    sz = ((SumNode*)pt)->weights_[k] * fmath::exp(pt->children_[k]->fr_ - max_logp) + smooth;
                    ((SumNode *) pt)->set_weight(k, sz / ssz);
                }
            }
        }
    }

    double SPNetwork::EvalDiff(const std::vector<double> &input, const std::vector<bool> &mask) {
        // Bottom-up evaluation pass, process in log-space to avoid numeric issue.
        // Set the value of leaf nodes first
        int var_index, cindex;
        double max_logp, sum_exp, tmp_val;
        // Compute forward values for the rest of internal nodes.
        for (VarNode *pt : dist_nodes_) {
            var_index = pt->var_index();
            if (mask[var_index]) {
                pt->fr_ = 0.0;
            } else {
                pt->fr_ = pt->log_prob(input[var_index]);
            }
        }
        for (SPNNode *pt : forward_order_) {
            if (pt->type() == SPNNodeType::SUMNODE) {
                // Avoid underflow
                max_logp = -std::numeric_limits<double>::infinity();
                for (size_t j = 0; j < pt->children_.size(); ++j)
                    if (pt->children_[j]->fr_ > max_logp)
                        max_logp = pt->children_[j]->fr_;
                if (max_logp == -std::numeric_limits<double>::infinity()) {
                    pt->fr_ = -std::numeric_limits<double>::infinity();
                } else {
                    sum_exp = 0.0;
                    for (size_t j = 0; j < pt->children_.size(); ++j)
                        sum_exp += ((SumNode *) pt)->weights_[j] * fmath::exp(pt->children_[j]->fr_ - max_logp);
                    pt->fr_ = max_logp + fmath::log(sum_exp);
                }
            } else if (pt->type() == SPNNodeType::PRODNODE) {
                pt->fr_ = 0.0;
                for (size_t j = 0; j < pt->children_.size(); ++j)
                    pt->fr_ += pt->children_[j]->fr_;
            }
        }
        // Top-down differentiation pass, process in log-space to avoid the numeric issue.
        root_->dr_ = 0.0;
        for (SPNNode *pt : backward_order_) {
            if (pt == root_) {
                continue;
            }
            pt->dr_ = 0.0;
            max_logp = -std::numeric_limits<double>::infinity();
            for (SPNNode *parent : pt->parents_) {
                // Find the index of current node in his parent.
                cindex = -1;
                for (size_t j = 0; j < parent->children_.size(); ++j) {
                    if (pt == parent->children_[j]) {
                        cindex = j;
                        break;
                    }
                }
                // Determine the shifting size.
                if (parent->type() == SPNNodeType::SUMNODE) {
                    tmp_val = parent->dr_ + fmath::log(((SumNode *) parent)->weights_[cindex]);
                    if (tmp_val > max_logp) {
                        max_logp = tmp_val;
                    }
                } else if (parent->type() == SPNNodeType::PRODNODE) {
                    tmp_val = parent->dr_ + parent->fr_ - pt->fr_;
                    if (tmp_val > max_logp) {
                        max_logp = tmp_val;
                    }
                }
            }
            // Avoid overflow.
            for (SPNNode *parent : pt->parents_) {
                // Find the index of current node in his parent.
                cindex = -1;
                for (size_t j = 0; j < parent->children_.size(); ++j) {
                    if (pt == parent->children_[j]) {
                        cindex = j;
                        break;
                    }
                }
                // Determine the shifting size.
                if (parent->type() == SPNNodeType::SUMNODE) {
                    tmp_val = parent->dr_ + fmath::log(((SumNode *) parent)->weights_[cindex]);
                    pt->dr_ += fmath::exp(tmp_val - max_logp);
                } else if (parent->type() == SPNNodeType::PRODNODE) {
                    tmp_val = parent->dr_ + parent->fr_ - pt->fr_;
                    pt->dr_ += fmath::exp(tmp_val - max_logp);
                }
            }
            pt->dr_ = fmath::log(pt->dr_) + max_logp;
        }
        return root_->fr_;
    }

    void SPNetwork::compute_bottom_up_(const std::vector<std::vector<double>> &inputs,
        const std::vector<std::vector<bool>> &mask, double** outputs) {
        int var_index;
        double max_logp, sum_exp;
        int n = inputs.size();
        assert(inputs.size() == mask.size());
        for (VarNode *pt : dist_nodes_) {
            int j = pt->id_;
            var_index = pt->var_index();
            for (int i = 0; i < n; ++i)
                outputs[i][j] = mask[i][var_index] ? 0.0 : pt->log_prob(inputs[i][var_index]);
        }
        for (SPNNode *pt : forward_order_) {
            for (int i = 0; i < n; ++i) {
                double *cf = outputs[i];
                if (pt->type() == SPNNodeType::SUMNODE) {
                    int n = pt->id_;
                    max_logp = -std::numeric_limits<double>::infinity();
                    for (size_t j = 0; j < pt->children_.size(); ++j) {
                        double f = cf[pt->children_[j]->id_];
                        if (f > max_logp)
                            max_logp = f;
                    }
                    if (max_logp == -std::numeric_limits<double>::infinity()) {
                        cf[n] = -std::numeric_limits<double>::infinity();
                    } else {
                        sum_exp = 0.0;
                        for (size_t j = 0; j < pt->children_.size(); ++j) {
                            double f = cf[pt->children_[j]->id_];
                            sum_exp += dynamic_cast<SumNode*>(pt)->weights_[j] * fmath::exp(f - max_logp);
                        }
                        cf[n] = max_logp + fmath::log(sum_exp);
                    }
                } else if (pt->type() == SPNNodeType::PRODNODE) {
                    double f = 0.0;
                    for (size_t j = 0; j < pt->children_.size(); ++j)
                        f += cf[pt->children_[j]->id_];
                    cf[n] = f;
                }
            }
        }
    }

    void construct_mask_(const std::vector<std::vector<double>>& input,
        std::vector<std::vector<bool>> &mask) {
        size_t n = input.size(), m = input[0].size();
        for (size_t i = 0; i < n; ++i)
          for (size_t j = 0; j < m; ++j)
              mask[i][j] = std::isnan(input[i][j]);
    }

    std::vector<double> SPNetwork::sample(const std::vector<double>& input) {
        std::vector<std::vector<double>> inputs;
        inputs.push_back(input);
        return sample(inputs)[0];
    }

    std::vector<std::vector<double>> SPNetwork::sample(const std::vector<std::vector<double>> &inputs) {
        size_t n = inputs.size();
        std::vector<std::vector<double>> samples(inputs);
        std::vector<std::vector<bool>> mask(n, std::vector<bool>(inputs[0].size(), false));
        std::queue<SPNNode*> Q;
        std::vector<bool> visited(num_nodes_, false);
        size_t s = n*sizeof(double);
        new_matrix(f, double, s, num_nodes_);

        construct_mask_(inputs, mask);
        compute_bottom_up_(inputs, mask, f);

        for (size_t i = 0; i < n; ++i) {
            Q.push(root_);
            while (!Q.empty()) {
                SPNNode *N = Q.front();
                Q.pop();
                visited[N->id_] = true;
                double fr = f[i][N->id_];
                std::pair<int, double> p = N->sample(fr);
                if (p.first > 0) { /* Either sum or product. */
                    if ((size_t) p.first == N->children_.size() + 2) {
                        for (SPNNode *C : N->children_)
                            if (!visited[C->id_]) Q.push(C);
                    } else {
                        SPNNode *C = N->children_[p.first - 1];
                        if (!visited[C->id_]) Q.push(C);
                    }
                } else { /* Leaf node. Sample variable. */
                    int c = -p.first;
                    assert(c >= 0);
                    if (mask[i][c])
                        samples[i][c] = p.second;
                }
            }
            for (auto v : visited) v = false;
        }

        del_matrix(f, n);

        return samples;
    }
}
