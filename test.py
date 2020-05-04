import faulthandler
import zhaospn

_P1 = zhaospn.ProdNode(2)
_P2 = zhaospn.ProdNode(5)
_P1.add_children([zhaospn.NormalNode(0, 0, 0, 1.0), zhaospn.NormalNode(1, 1, 1, 0.5)])
_P2.add_children([zhaospn.NormalNode(3, 1, 2.0, 0.3), zhaospn.NormalNode(4, 0, 2, 1.0)])
S = zhaospn.SumNode(6)
S.add_weight(0.3); S.add_weight(0.7)
S.add_children([_P1, _P2])
SPN = zhaospn.SPNetwork(S)
faulthandler.enable()
SPN.init()
