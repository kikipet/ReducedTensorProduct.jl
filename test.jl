include("rtp_dq.jl")
include("irreps.jl")
using Einsum
using LinearAlgebra


function assert_equivariant()
    irreps_in = "1o"
    _, irreps_out, Q = ReducedTensorProduct.reduced_product_dq("ijkl=jikl=klij", Dict('i' => irreps_in))

    abc = o3.rand_angles(())

    R = o3.D_from_angles_irreps(o3.Irreps(irreps_in), abc...)
    D = o3.D_from_angles_irreps(irreps_out, abc...)

    @einsum Q1[z, a, b, c, d] := Q[z, i, j, k, l] * R[i, a] * R[j, b] * R[k, c] * R[l, d]
    @einsum Q2[w, i, j, k, l] := D[w, z] * Q[z, i, j, k, l]

    @assert isapprox(Q1, Q2, atol=1e-5)
end


function assert_permutation()
    irreps_in = "1o"
    _, irreps_out, Q = ReducedTensorProduct.reduced_product_dq("ijkl=jikl=klij", Dict('i' => irreps_in))

    # 21, 3, 3, 3, 3

    @einsum Q1[z, i, j, k, l] := Q[z, j, i, k, l]
    @assert isapprox(Q1, Q, atol=1e-5)

    @einsum Q2[z, i, j, k, l] := Q[z, k, l, j, i]
    @assert isapprox(Q2, Q, atol=1e-5)
end


function assert_permutation2()
    irreps_in = "1o"
    _, irreps_out, Q = ReducedTensorProduct.reduced_product_dq("ijk=-jik=jki", Dict('i' => irreps_in))

    @einsum Q1[z, i, j, k] := -Q[z, j, i, k]
    @assert isapprox(Q1, Q, atol=1e-5)

    @einsum Q2[z, i, j, k] := Q[z, j, k, i]
    @assert isapprox(Q2, Q, atol=1e-5)
end


function assert_orthogonality()
    irreps_in = "1o"
    _, irreps_out, Q = ReducedTensorProduct.reduced_product_dq("ijkl=jikl=klij", Dict('i' => irreps_in))
    @einsum n2[z, w] := Q[z, i, j, k, l] * Q[w, i, j, k, l]
    @assert isapprox(n2, I(size(Q, 1)), atol=1e-5)
end
