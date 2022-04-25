module Wigner

using LinearAlgebra

struct TP
    op::Tuple
    args::Tuple
end

struct Input
    # start and stop are both inclusive
    tensor::Int
    start::Int
    stop::Int
end

int(x) = floor(Int, x)

function change_basis_real_to_complex(l)
    q = zeros(ComplexF64, 2 * l + 1, 2 * l + 1)
    for m in -l:-1
        q[l + m + 1, l + abs(m) + 1] = sqrt(0.5)
        q[l + m + 1, l - abs(m) + 1] = -sqrt(0.5) * 1im
    end
    q[l + 1, l + 1] = 1
    for m in 1:l
        q[l + m + 1, l + abs(m) + 1] = (-1)^m * sqrt(0.5)
        q[l + m + 1, l - abs(m) + 1] = 1im * (-1)^m * sqrt(0.5)
    end
    q = (-1im)^l * q  # Added factor of 1im^l to make the Clebsch-Gordan coefficients real
    return q
end

function wigner_3j(l1, l2, l3)
    @assert abs(l2 - l3) <= l1 <= l2 + l3
    @assert typeof(l1) == Int && typeof(l2) == Int && typeof(l3) == Int
    C = _so3_clebsch_gordan(l1, l2, l3)
    return C
end

function _so3_clebsch_gordan(l1, l2, l3) ## this is called clebsch_gordan in e3nn_jax
    Q1 = change_basis_real_to_complex(l1)
    Q2 = change_basis_real_to_complex(l2)
    Q3 = change_basis_real_to_complex(l3)
    Q3 = conj(transpose(Q3))
    C = _su2_clebsch_gordan(l1, l2, l3)
    ### begin einsum: C = einsum('ij,kl,mn,ikn->jlm', Q1, Q2, Q3, C)
    s1 = size(Q1)
    s2 = size(Q2)
    s3 = size(Q3)
    C_prod = zeros(s1[2], s2[2], s3[1])
    # C_prod = zeros(ComplexF64, s1[2], s2[2], s3[1])
    for ein_j in 1:s1[2]
        for ein_l in 1:s2[2]
            for ein_m in 1:s3[1]
                ein_total = 0
                for ein_i in 1:s1[1]
                    for ein_k in 1:s2[1]
                        for ein_n in 1:s3[2]
                            ein_total += Q1[ein_i, ein_j] * Q2[ein_k, ein_l] * Q3[ein_m, ein_n] * C[ein_i, ein_k, ein_n]
                        end
                    end
                end
                if abs(ein_total) < 1e-9
                    C_prod[ein_j, ein_l, ein_m] = 0
                else
                    C_prod[ein_j, ein_l, ein_m] = ein_total
                end
            end
        end
    end
    ### end einsum

    # make it real
    @assert all(map(c -> abs(imag(c)) < 1e-5, C_prod))
    C = real(C_prod)

    # normalization
    C = C / norm(C)
    return C
end

function _su2_clebsch_gordan(j1, j2, j3)
    @assert typeof(j1) == Int || typeof(j1) == Float64
    @assert typeof(j2) == Int || typeof(j2) == Float64
    @assert typeof(j3) == Int || typeof(j3) == Float64
    mat = zeros(int(2 * j1 + 1), int(2 * j2 + 1), int(2 * j3 + 1))
    if int(2 * j3) in range(int(2 * abs(j1 - j2)), int(2 * (j1 + j2)), step=2)
        for m1 in [x / 2 for x in range(-int(2 * j1), int(2 * j1), step=2)]
            for m2 in [x / 2 for x in range(-int(2 * j2), int(2 * j2), step=2)]
                if abs(m1 + m2) <= j3
                    mat[int(j1 + m1) + 1, int(j2 + m2) + 1, int(j3 + m1 + m2) + 1] = _su2_clebsch_gordan_coeff((j1, m1), (j2, m2), (j3, m1 + m2))
                end
            end
        end
    end
    return mat
end

function _su2_clebsch_gordan_coeff(idx1, idx2, idx3)
    j1, m1 = idx1
    j2, m2 = idx2
    j3, m3 = idx3

    if m3 != m1 + m2
        return 0
    end
    vmin = int(max(-j1 + j2 + m3, -j1 + m1, 0))
    vmax = int(min(j2 + j3 + m1, j3 - j1 + j2, j3 + m3))

    function f(n)
        @assert n == round(n)
        return convert(Float64, factorial(big(round(Int, n))))
    end

    C = sqrt((2 * j3 + 1) * f(j3 + j1 - j2) * f(j3 - j1 + j2) * f(j1 + j2 - j3) * f(j3 + m3) * f(j3 - m3) /  # noqa: W504
                (f(j1 + j2 + j3 + 1) * f(j1 - m1) * f(j1 + m1) * f(j2 - m2) * f(j2 + m2)))
    S = 0
    for v in vmin:vmax
        S += (-1) ^ (v + j2 + m2) / f(v) * f(j2 + j3 + m1 - v) * f(j1 - m1 + v) / f(j3 - j1 + j2 - v) / f(j3 + m3 - v) / f(v + j1 - j2 - m3)
    end
    C = C * S
    return C
end


function _get_ops(path)
    if typeof(path) == Input
        return
    end
    @assert typeof(path) == TP
    ops = []
    push!(ops, path.op)
    for op in _get_ops(path.args[1])
        push!(ops, op)
    end
    return ops
end

export TP, Input, wigner_3j, _get_ops
end