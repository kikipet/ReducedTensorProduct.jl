# ReducedTensorProduct.jl

A set of functions to work with representations in the O(3) group. In particular:
* a module (`o3`) supporting implementations of irreducible representations (irreps) (`irreps.jl`)
* functions to find a change-of-basis matrix `Q` to contract reducible representations to a single set of irreps (`rtp.jl`)

## How to use
The file `rtp.jl` defines a module `ReducedTensorProduct` containing two functions `reduced_product` and `reduced_product_dq`, which both can be used to find `Q` as described above. Both can be run as serial functions or in parallel through multithreading. `reduced_product_dq` is the divide-and-conquer version of `reduced_product` and, except for very small cases, the parallelized version of `reduced_product_dq` is generally faster than either the serial `reduced_product_dq` or either version of `reduced_product`.

## Example
```
include("rtp.jl")
irreps_in, irreps_out, Q = ReducedTensorProduct.reduced_product_dq("ijkl=jikl=klij", Dict('i' => "1o"), parallel=true)
# or:
irreps_in, irreps_out, Q = ReducedTensorProduct.reduced_product("ijkl=jikl=klij", Dict('i' => "1o"), parallel=false)
```
