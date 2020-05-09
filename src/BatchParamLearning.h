//
// Created by Han Zhao on 5/31/15.
//

#ifndef SPN_EM_PARAMLEARNING_H
#define SPN_EM_PARAMLEARNING_H

#include "SPNetwork.h"

#include <vector>

namespace SPN {
    class BatchParamLearning {
    public:
        BatchParamLearning() = default;

        virtual ~BatchParamLearning() = default;

        virtual void fit(const std::vector<std::vector<double>>&,
                         const std::vector<std::vector<double>>&,
                         SPNetwork&, bool = false) { };

        std::string algo_name() const {
            return algo_name_;
        }

    protected:
        std::string algo_name_ = "AbstractBatchParamLearning";
    };

    // Expectation Maximization algorithm for batch training of SPNs.
    // This is also the Concave-convex procedure (CCCP) based algorithm.
    class ExpectMax : public BatchParamLearning {
    public:
        ExpectMax() : num_iters_(50), lap_lambda_(1), stop_thred_(1e-4)  {
            algo_name_ = "BatchExpectMax";
        }

        ExpectMax(size_t num_iters, double stop_thred, double lap_lambda) :
                num_iters_(num_iters), lap_lambda_(lap_lambda), stop_thred_(stop_thred)  {
            algo_name_ = "BatchExpectMax";
        }

        virtual ~ExpectMax() = default;

        void fit(const std::vector<std::vector<double>> &trains,
                 const std::vector<std::vector<double>> &valids,
                 SPNetwork &spn, bool verbose = false) override;

    private:
        size_t num_iters_;
        double lap_lambda_;
        double stop_thred_;
    };

    // Collapsed Variational Bayes algorithm for batch training of SPNs.
    class CollapsedVB : public BatchParamLearning {
    public:
        CollapsedVB() : num_iters_(50), lrate_(1e-1), stop_thred_(1e-4),
                         prior_scale_(100.0), seed_(42) {
            algo_name_ = "BatchCollapsedVB";
        }

        CollapsedVB(size_t num_iters, double stop_thred, double lrate,
                    double prior_scale, uint seed=42) :
                num_iters_(num_iters), lrate_(lrate), stop_thred_(stop_thred),
                prior_scale_(prior_scale), seed_(seed) {
            algo_name_ = "BatchCollapsedVB";
        }

        virtual ~CollapsedVB() = default;

        void fit(const std::vector<std::vector<double>> &trains,
                 const std::vector<std::vector<double>> &valids,
                 SPNetwork &spn, bool verbose = false) override;

    private:
        size_t num_iters_;
        double lrate_;
        double stop_thred_;
        double prior_scale_;
        uint seed_;
    };

    // Projected gradient descent algorithm for batch training of SPNs.
    class ProjectedGD : public BatchParamLearning {
    public:
        ProjectedGD() : num_iters_(50), proj_eps_(1e-2),
                        stop_thred_(1e-3), lrate_(1e-1),
                        shrink_weight_(8e-1), map_prior_(true),
                        prior_scale_(100.0), seed_(42) {
            algo_name_ = "BatchProjectedGD";
        }

        ProjectedGD(size_t num_iters, double proj_eps, double stop_thred,
                    double lrate, double shrink_weight, bool map_prior=true,
                    double prior_scale=100.0, uint seed=42) :
                num_iters_(num_iters), proj_eps_(proj_eps),
                stop_thred_(stop_thred), lrate_(lrate),
                shrink_weight_(shrink_weight), map_prior_(map_prior),
                prior_scale_(prior_scale), seed_(seed) {
            algo_name_ = "BatchProjectedGD";
        }

        virtual ~ProjectedGD() = default;

        void fit(const std::vector<std::vector<double>> &trains,
                 const std::vector<std::vector<double>> &valids,
                 SPNetwork &spn, bool verbose = false) override;

    private:
        size_t num_iters_;
        double proj_eps_;
        double stop_thred_;
        double lrate_;
        double shrink_weight_;
        bool map_prior_;
        double prior_scale_;
        uint seed_ = 42;
    };

    // L-BFGS algorithm for batch training of SPNs.
    class LBFGS : public BatchParamLearning {
    public:
        LBFGS() : num_iters_(50), proj_eps_(1e-2), stop_thred_(1e-3),
                  lrate_(1e-1), shrink_weight_(8e-1), history_window_(5) {
            algo_name_ = "BatchLBFGS";
        }

        LBFGS(size_t num_iters, double proj_eps, double stop_thred,
              double lrate, double shrink_weight, uint history_window) :
                num_iters_(num_iters), proj_eps_(proj_eps), stop_thred_(stop_thred),
                lrate_(lrate), shrink_weight_(shrink_weight), history_window_(history_window) {
            algo_name_ = "BatchLBFGS";
        }

        virtual ~LBFGS() = default;

        void fit(const std::vector<std::vector<double>> &trains,
                 const std::vector<std::vector<double>> &valids,
                 SPNetwork &spn, bool verbose = false) override;

    private:
        size_t num_iters_;
        double proj_eps_;
        double stop_thred_;
        double lrate_;
        double shrink_weight_;
        uint history_window_;
    };

    // Exponentiated gradient for batch training of SPNs.
    class ExpoGD : public BatchParamLearning {
    public:
        ExpoGD() : num_iters_(50), stop_thred_(1e-3), lrate_(1e-1), shrink_weight_(8e-1) {
            algo_name_ = "BatchExpoGD";
        }

        ExpoGD(size_t num_iters, double stop_thred, double lrate, double shrink_weight) :
                num_iters_(num_iters), stop_thred_(stop_thred), lrate_(lrate), shrink_weight_(shrink_weight) {
            algo_name_ = "BatchExpoGD";
        }

        virtual ~ExpoGD() = default;

        void fit(const std::vector<std::vector<double>> &trains,
                 const std::vector<std::vector<double>> &valids,
                 SPNetwork &spn, bool verbose = false) override;

    private:
        size_t num_iters_;
        double stop_thred_;
        double lrate_;
        double shrink_weight_;
    };

    // Sequential monomial approximation algorithm for batch training of SPNs.
    class SMA : public BatchParamLearning {
    public:
        SMA() : num_iters_(50), stop_thred_(1e-3), lrate_(1e-1), shrink_weight_(8e-1) {
            algo_name_ = "BatchSMA";
        }

        SMA(int num_iters, double stop_thred, double lrate, double shrink_weight)
                : num_iters_(num_iters), stop_thred_(stop_thred), lrate_(lrate), shrink_weight_(shrink_weight) {
            algo_name_ = "BatchSMA";
        }

        virtual ~SMA() = default;

        void fit(const std::vector<std::vector<double>> &trains,
                 const std::vector<std::vector<double>> &valids,
                 SPNetwork &spn, bool verbose = false) override;

    private:
        size_t num_iters_;
        double stop_thred_;
        double lrate_;
        double shrink_weight_;
    };
}


#endif //SPN_EM_PARAMLEARNING_H
