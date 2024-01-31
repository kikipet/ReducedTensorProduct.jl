import jax
import e3nn_jax._src.reduced_tensor_product as rtp  # uses lru cache
import rtp_jax  # no caching :P
import time

def benchmark(f, n, **kwargs):
    f(**kwargs)
    tot_time = 0
    for _ in range(n):
        start = time.time()
        f(**kwargs)
        tot_time += time.time() - start
    return tot_time / n

irreps = ["0e+1o+2e+3o"]
# irreps = ["0e", "0e+1o", "0e+1o+2e", "0e+1o+2e+3o", "0e+1o+2e+3o+4e"]

# print("JAX w/ caching")
# for irrep in irreps:
#     print(f"Benchmarking {irrep}")
#     runtime = benchmark(rtp.reduced_tensor_product_basis, 10, **{
#         "formula_or_irreps_list": "ijk=jik=kij",
#         "i":irrep,
#     })
#     print(f"Time: {runtime} s")

# print("=====")
print("JAX w/o caching")

for irrep in irreps:
    print(f"Benchmarking {irrep}")
    runtime = benchmark(rtp_jax.reduced_tensor_product_basis, 10, **{
        "formula_or_irreps_list": "ijk=jik=kij",
        "i":irrep,
    })
    print(f"Time: {runtime} s")

print("=====")