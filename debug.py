from e3nn.o3 import *
from _reduce import _wigner_nj
# from e3nn.o3._reduce import _wigner_nj

irreps = {'i': Irreps('1x1o'), 'j': Irreps('1x1o')}

res = _wigner_nj(*[irreps['i'], irreps['j']])
