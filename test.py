import random
import numpy as np
import inspect
import zhaospn

def add_child(P, C, w = None):
  D = ([[1., 0.], [1., 1.], [2., 1.], [0., 1.]], [[1., 1.], [0., 0.], [2., 2.]])
  P.add(C)
  if w: P.add_weight(w)
  return P

def add_children(P, C, W = None):
  h = W is not None
  P.add(C)
  for i, c in enumerate(C):
    if h: P.add_weight(W[i])
  return P

def example_SPN(discrete = False):
  C = [(
    zhaospn.spn.BernoulliNode(0, 0, 0.3),
    zhaospn.spn.BernoulliNode(1, 1, 0.5)
  ), (
    zhaospn.spn.BernoulliNode(3, 1, 0.8),
    zhaospn.spn.BernoulliNode(4, 0, 0.4)
  )] if discrete else [(
    zhaospn.spn.NormalNode(0, 0, 0, 1.0),
    zhaospn.spn.NormalNode(1, 1, 1, 0.5)
  ), (
    zhaospn.spn.NormalNode(3, 1, 2, 0.3),
    zhaospn.spn.NormalNode(4, 0, 2, 1.0)
  )]

  S = zhaospn.spn.SPNetwork(
    add_children(zhaospn.spn.SumNode(6), [
      add_children(zhaospn.spn.ProdNode(2), C[0]),
      add_children(zhaospn.spn.ProdNode(5), C[1])
    ], [0.3, 0.7])
  )
  return S

def get_concrete_learners(path):
  L = []
  for (n, l) in inspect.getmembers(path, inspect.isclass):
    if l.__bases__[0].__name__ != 'pybind11_object': L.append((n, l()))
  return (path.__name__, L)

def get_learners():
  return [
    get_concrete_learners(zhaospn.batch),
    get_concrete_learners(zhaospn.online),
    get_concrete_learners(zhaospn.stream)
  ]

def gen_dataset(n, missing = False):
  D = []
  for i in range(n):
    p = random.random()
    x = [random.random(), np.nan if missing else random.random()]
    D = np.append(D, np.random.choice(x, size=2, p=[p, 1.0-p]))
  return D.reshape(n, 2)

def split_data(D, p):
  k = int(len(D)*p)
  return D[:k], D[k:]

def test_learners():
  D = split_data(gen_dataset(200), 0.7)
  S = example_SPN()
  L = get_learners()
  for T in L:
    t = T[0]
    print("======\nType: {}\n======".format(t))
    for p in T[1]:
      n, l = p
      print("  -> Learning with {}:\n---".format(n))
      if 'batch' in t:
        l.fit(D[0], D[1], S, True)
      elif 'online' in t:
        l.fit(D[0], D[1], S, 4, True)
      else:
        for x in D[0]: l.fit(x, S, True)
      S.print()
      print('---')

def test_sampling():
  D = gen_dataset(10000, missing = True)
  S = example_SPN(discrete = True)
  r = np.array(S.sample(D))
  sample_p = r.sum(axis=0)/r.sum()
  true_p = np.array(S.inference([[0, 0], [0, 1], [1, 0], [1, 1]]))
  true_p = np.array((true_p[:2].sum(), true_p[2:].sum()))
  print("True probability: {}\nSample probability: {}".format(true_p, sample_p))

def main():
  test_sampling()

if __name__ == '__main__':
  main()
