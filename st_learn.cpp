#include <stdio.h>
#include <cmath>
#include <string>

#include <vector>

#include "src/SPNNode.h"
#include "src/SPNetwork.h"
#include "src/utils.h"

class BernoulliNode : public SPN::VarNode {
  public:
    BernoulliNode() = default;
    virtual ~BernoulliNode() = default;

    BernoulliNode(int id, int var_index, double var_value) : SPN::VarNode(id, var_index), var_value_(var_value) {}

    SPN::VarNodeType distribution() const override { return SPN::VarNodeType::BINNODE; }

    std::string type_string() const override {
      return std::string("BernoulliNode");
    }

    size_t num_param() const override { return 0; }

    double var_value() const { return var_value_; }

    double prob(double x) const override {
      printf("x=%f, var_value_=%f\n", x, var_value_);
      if (x == 0) return var_value_;
      return 1.0 - var_value_;
    }

    double log_prob(double x) const override {
      if (x == 0) return log(var_value_);
      return log(1.0 - var_value_);
    }

    std::string string() const override {
      return "no";
    }

    friend class SPN::SPNetwork;

  private:
    double var_value_;
};

void add_child(SPN::SPNNode *pa, SPN::SPNNode *ch) {
  pa->add_child(ch);
  ch->add_parent(pa);
}

void add_wchild(SPN::SumNode *pa, SPN::SPNNode *ch, double w) {
  add_child(pa, ch);
  pa->add_weight(w);
}

int main() {
  SPN::SumNode *root = new SPN::SumNode(0);
  SPN::ProdNode *p1 = new SPN::ProdNode(1), *p2 = new SPN::ProdNode(2), *p3 = new SPN::ProdNode(3);
  SPN::SumNode *s1 = new SPN::SumNode(4), *s2 = new SPN::SumNode(5), *s3 = new SPN::SumNode(6), *s4 = new SPN::SumNode(7);
  //BernoulliNode *x11 = new BernoulliNode(8, 0, 0.25), *x12 = new BernoulliNode(9, 0, 0.65);
  //BernoulliNode *x21 = new BernoulliNode(10, 1, 0.35), *x22 = new BernoulliNode(11, 1, 0.95);
  //SPN::BinNode *x11 = new SPN::BinNode(8, 0, 0.2), *x12 = new SPN::BinNode(9, 0, 0.6);
  //SPN::BinNode *x21 = new SPN::BinNode(10, 1, 0.3), *x22 = new SPN::BinNode(11, 1, 0.9);
  SPN::NormalNode *x11 = new SPN::NormalNode(8, 0, 0.2, 0.3), *x12 = new SPN::NormalNode(9, 0, 0.6, 0.3);
  SPN::NormalNode *x21 = new SPN::NormalNode(10, 1, 0.3, 0.1), *x22 = new SPN::NormalNode(11, 1, 0.9, 1.0);

  add_wchild(root, p1, 0.5);
  add_wchild(root, p2, 0.2);
  add_wchild(root, p3, 0.3);

  add_child(p1, s1);
  add_child(p1, s3);
  add_child(p2, s2);
  add_child(p2, s3);
  add_child(p3, s2);
  add_child(p3, s4);

  add_wchild(s1, x11, 0.3);
  add_wchild(s1, x12, 0.7);

  add_wchild(s2, x11, 0.2);
  add_wchild(s2, x12, 0.8);

  add_wchild(s3, x21, 0.6);
  add_wchild(s3, x22, 0.4);

  add_wchild(s4, x21, 0.9);
  add_wchild(s4, x22, 0.1);

  SPN::SPNetwork S(root);
  S.init();

  std::vector<double> V {1.0, 0.0};
  puts("V = (");
  for (auto it = V.begin(); it != V.end(); ++it) {
    printf("%f,\n", *it);
  }
  puts(")");
  double p = S.inference(V, true);

  std::string v(V.begin(), V.end());
  printf("P(%s) = %f\n", v.c_str(), p);

  SPN::utils::save(&S, "spn.net");

  return 0;
}
