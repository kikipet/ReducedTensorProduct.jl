using Distributed, BenchmarkTools
include("rtp.jl")

irreps = ["0e", "0e+1o", "0e+1o+2e", "0e+1o+2e+3o"]

for irrep in irreps
    println(irrep)
    @btime ReducedTensorProduct.reduced_product("ijk=jik=kij", Dict('i' => $irrep), parallel=false)
    @btime ReducedTensorProduct.reduced_product_dq("ijk=jik=kij", Dict('i' => $irrep), parallel=false)
    @btime ReducedTensorProduct.reduced_product_dq("ijk=jik=kij", Dict('i' => $irrep), parallel=true)
    println("---")
end
