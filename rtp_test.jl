include("rtp.jl")

function test_antisymmetric_matrix(float_tolerance)
    _, _, Q = reduced_tensor_product("ij=-ji", Dict('i' => "5x0e + 1e"))

    x = torch.randn(2, 5 + 3)
    @assert (wigner.tp(x[1, :], x[2, :]) - torch.einsum('xij,i,j', Q, *x)).abs().max() < float_tolerance # this is a nightmare of types

    @assert (Q + torch.einsum("xij->xji", Q)).abs().max() < float_tolerance
end

function test_reduce_tensor_Levi_Civita_symbol(float_tolerance)
    irreps_in, irreps_out, Q = reduced_tensor_product("ijk=-ikj=-jik", Dict('i' => "1e"))
    @assert irreps_out == o3.Irreps("0e")

    x = torch.randn(3, 3)
    @assert (tp(*x) - torch.einsum("xijk,i,j,k", Q, *x)).abs().max() < float_tolerance

    @assert (Q + torch.einsum("xijk->xikj", Q)).abs().max() < float_tolerance
    @assert (Q + torch.einsum("xijk->xjik", Q)).abs().max() < float_tolerance
end

function test_reduce_tensor_antisymmetric_L2(float_tolerance)
    _, _, Q = reduced_tensor_product("ijk=-ikj=-jik", Dict('i'=>"2e"))

    x = torch.randn(3, 5)
    @assert (tp(*x) - torch.einsum("xijk,i,j,k", Q, *x)).abs().max() < float_tolerance

    @assert (Q + torch.einsum("xijk->xikj", Q)).abs().max() < float_tolerance
    @assert (Q + torch.einsum("xijk->xjik", Q)).abs().max() < float_tolerance
end

function test_reduce_tensor_elasticity_tensor(float_tolerance)
    irreps_in, irreps_out, Q = reduced_tensor_product("ijkl=jikl=klij", i="1e")
    @assert o3.dim(irreps_out) == 21

    x = randn(4, 3)
    @assert (tp(*x) - torch.einsum("xijkl,i,j,k,l", Q, *x)).abs().max() < float_tolerance

    @assert (Q - torch.einsum("xijkl->xjikl", Q)).abs().max() < float_tolerance
    @assert (Q - torch.einsum("xijkl->xijlk", Q)).abs().max() < float_tolerance
    @assert (Q - torch.einsum("xijkl->xklij", Q)).abs().max() < float_tolerance
end

function test_reduce_tensor_elasticity_tensor_parity(float_tolerance)
    irreps_in, irreps_out, Q = reduced_tensor_product("ijkl=jikl=klij", Dict('i' => "1o"))
    @assert o3.dim(irreps_out) == 21
    @assert all(ir.p == 1 for (_, ir) in irreps_out)

    x = randn(4, 3)
    @assert (tp(*x) - torch.einsum("xijkl,i,j,k,l", Q, *x)).abs().max() < float_tolerance

    @assert (Q - torch.einsum("xijkl->xjikl", Q)).abs().max() < float_tolerance
    @assert (Q - torch.einsum("xijkl->xijlk", Q)).abs().max() < float_tolerance
    @assert (Q - torch.einsum("xijkl->xklij", Q)).abs().max() < float_tolerance
end