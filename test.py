from e3nn.o3 import Irreps
from rtp import ReducedTensorProducts as RTP
from e3nn.o3._reduce import ReducedTensorProducts as realRTP


def test_RTP(formula, **kwargs):
    RTP_res = RTP(formula, **kwargs)
    realRTP_res = realRTP(formula, **kwargs)
    assert RTP_res.irreps_out == RTP_res.irreps_out

if __name__ == 'main':
    test_RTP('ij', i='1e', j='1e')
    test_RTP('ij=ji', i='1e', j='1e')
    test_RTP('ij', i='1e', j='3e')
    test_RTP('ijkl=ijlk', i='1e', j='3e', k='2e', l='1e')
    print('all tests passed')