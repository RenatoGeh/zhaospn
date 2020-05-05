import inspect
import zhaospn

def add_child(P, C, w = None):
  P.add_child(C)
  C.add_parent(P)
  if w: P.add_weight(w)
  return P

def add_children(P, C, W = None):
  h = W is not None
  P.add_children(C)
  for i, c in enumerate(C):
    c.add_parent(P)
    if h: P.add_weight(W[i])
  return P

def example_SPN():
  S = zhaospn.spn.SPNetwork(
    add_children(zhaospn.spn.SumNode(6), [
      add_children(zhaospn.spn.ProdNode(2), [
        zhaospn.spn.NormalNode(0, 0, 0, 1.0),
        zhaospn.spn.NormalNode(1, 1, 1, 0.5)
      ]),
      add_children(zhaospn.spn.ProdNode(5), [
        zhaospn.spn.NormalNode(3, 1, 2, 0.3),
        zhaospn.spn.NormalNode(4, 0, 2, 1.0)
      ])
    ], [0.3, 0.7])
  )
  S.init()
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

def main():
  S = example_SPN()
  L = get_learners()
  D = ([[1., 0.], [1., 1.], [2., 1.], [0., 1.]], [[1., 1.], [0., 0.], [2., 2.]])
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

if __name__ == '__main__':
  main()
