from e3nn.o3 import *
# from _reduce import _wigner_nj
from _reduce import ReducedTensorProducts

irreps = {'i': Irreps('1x1o'), 'j': Irreps('1x1o')}

res = ReducedTensorProducts('ijkl=jikl=ikjl=ijlk',i='1e')