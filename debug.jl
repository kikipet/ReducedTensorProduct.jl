include("rtp.jl")

reduced_tensor_product("ij=-ji", Dict(['i' => o3.Irrep("1o")]))
reduced_tensor_product("ijkl=jikl=ikjl=ijlk", Dict(['i' => o3.Irrep("1e")]))