@everywhere module ReducedTensorProduct

using Distributed
# using SharedArrays
# using Base.Threads
using IterTools
using LinearAlgebra
using Einsum
include("irreps.jl")
include("wigner.jl")
@everywhere using Distributed
@everywhere using IterTools
@everywhere using LinearAlgebra
@everywhere using Einsum
@everywhere include("irreps.jl")
@everywhere include("wigner.jl")


function _wigner_nj(irrepss; normalization="component", filter_ir_mid=nothing)
    """
    Args:
        irrepss (Dict)
    """
    irrepss = [o3.Irreps(irreps) for irreps in irrepss]
    if filter_ir_mid !== nothing
        filter_ir_mid = [o3.Irrep(ir) for ir in filter_ir_mid]
    end

    if length(irrepss) == 1
        irreps = irrepss[1]
        ret = []
        i_dim = o3.dim(irreps)
        e = I(i_dim)
        i = 1
        for mul_ir in irreps
            for _ in 1:mul_ir.mul
                stop = i + o3.dim(mul_ir.ir) - 1
                push!(ret, (mul_ir.ir, e[i:stop, :]))
                i += o3.dim(mul_ir.ir)
            end
        end
        return ret
    end

    irreps_right = irrepss[length(irrepss)]
    irrepss_left = irrepss[1:length(irrepss)-1]
    irrep_left_dims = [o3.dim(irreps) for irreps in irrepss_left]
    ret = []
    for (ir_left, C_left) in _wigner_nj(irrepss_left, normalization=normalization, filter_ir_mid=filter_ir_mid)
        i = 0
        for mul_ir in irreps_right
            for ir_out in ir_left * mul_ir.ir
                if filter_ir_mid !== nothing && !(ir_out in filter_ir_mid)
                    continue
                end

                C = Wigner.wigner_3j(ir_out.l, ir_left.l, mul_ir.ir.l)
                if normalization == "component"
                    C *= o3.dim(ir_out)^0.5
                end
                if normalization == "norm"
                    C *= o3.dim(ir_left)^0.5 * o3.dim(mul_ir.ir)^0.5
                end

                # C = einsum("jk,ijl->ikl", C_left.flatten(1), C)
                C_left_flat = reshape(C_left, size(C_left, 1), :)
                @einsum C_prod[i, k, l] := C_left_flat[j, k] * C[i, j, l]
                C = reshape(C_prod, o3.dim(ir_out), irrep_left_dims..., o3.dim(mul_ir.ir))
                for u in 1:mul_ir.mul
                    E = zeros(o3.dim(ir_out), irrep_left_dims..., o3.dim(irreps_right))
                    start = i + (u-1) * o3.dim(mul_ir.ir) + 1
                    stop = i + u * o3.dim(mul_ir.ir)
                    # access last dimension
                    ndims = length(size(E))
                    E[[1:d for d in size(E)[1:ndims-1]]..., start:stop] = C
                    push!(ret, (ir_out, E))
                end
            end
            i += mul_ir.mul * o3.dim(mul_ir.ir)
        end
    end

    return sort(ret, by = x -> x[1])
end

function is_perm(p)
    return Set(p) == Set(1:length(p))
end

function perm_inverse(p)
    """
    compute the inverse permutation
    """
    return [findfirst(isequal(i), p) for i in 1:length(p)]
end

function perm_compose(p1, p2)
    """
    compute p1 . p2
    """
    @assert is_perm(p1) && is_perm(p2)
    @assert length(p1) == length(p2)
    # p: i |-> p[i]

    # [p1.p2](i) = p1(p2(i)) = p1[p2[i]]
    return [p1[p2[i]] for i in 1:length(p1)]
end

function germinate_formulas(formula)
    formula_split = split(formula, '=')
    formulas = []
    for f in formula_split
        if f[1] == '-'
            push!(formulas, (-1, replace(f, "-" => "")))
        else
            push!(formulas, (1, f))
        end
    end
    s0, f0 = formulas[1]
    @assert s0 == 1

    # `formulas` is a list of (sign, permutation of indices)
    # each formula can be viewed as a permutation of the original formula
    formula_set = Set()
    for (s, f) in formulas
        if length(Set(f)) != length(f) || Set(f) != Set(f0)
            throw(error("$f is not a permutation of $f0"))
        end
        if length(f0) != length(f)
            throw(error("$f0 and $f don't have the same number of indices"))
        end
        indices = (findfirst(isequal(i), f) for i in f0)
        push!(formula_set, (s, vcat(indices...)))
    end

    # they can be composed, for instance if you have ijk=jik=ikj
    # you also have ijk=jki
    # applying all possible compositions creates an entire group
    while true
        n = length(formula_set)
        formula_set = union(formula_set, Set([(s, perm_inverse(p)) for (s, p) in formula_set]))
        formula_set = union(formula_set, Set([
            (s1 * s2, perm_compose(p1, p2))
            for (s1, p1) in formula_set
            for (s2, p2) in formula_set
        ]))
        if length(formula_set) == n
            break  # we break when the set is stable => it is now a group \o/
        end
    end

    return f0, formula_set
end

function orthonormalize(original, ε = 1e-9)
    """orthonomalize vectors

    Parameters
    ----------
    original : `torch.Tensor`
        list of the original vectors :math:`x`

    eps : float
        a small number

    Returns
    -------
    final : `torch.Tensor`
        list of orthonomalized vectors :math:`y`

    matrix : `torch.Tensor`
        the matrix :math:`A` such that :math:`y = A x`
    """
    @assert ndims(original) == 2
    dim = size(original, 2)

    final = []
    matrix = []

    for i in 1:size(original, 1)
        x = original[i, :]
        cx = zeros(size(original, 1))
        cx[i] = 1
        for (j, y) in enumerate(final)
            c = sum(x .* y)
            x = x - c * y
            cx = cx - c * matrix[j]
        end
        if norm(x) > 2 * ε
            c = 1 / norm(x)
            x = c * x
            cx = c * cx
            x[findall(el -> abs(el) < ε, x)] .= 0
            cx[findall(el -> abs(el) < ε, cx)] .= 0
            c = sign(x[findall(el -> el != 0, x)[1, 1]])
            x = c * x
            cx = c * cx
            push!(final, x)
            push!(matrix, cx)
        end
    end

    if length(final) == 0
        final = zeros((0, dim)) # ?
    end
    if length(matrix) == 0
        matrix = zeros((0, length(original))) # ?
    end

    return final, matrix
end

function subformulas(f0, formulas, subset)
    subset_indices = Set(i for i in 1:length(f0) if f0[i] in subset)
    standard_indices = Dict()
    for (i, ndx) in enumerate(subset)
        standard_indices[ndx] = i
    end

    subformulas = Set()
    subformulas_st = Set()
    for (s, f) in formulas
        if all(f0[i] in subset || f[i] == i for i in 1:length(f0))
            f_filtered = filter(x -> x in subset_indices, f)
            f_standard = map(x -> standard_indices[f0[x]], f_filtered)
            push!(subformulas, (s, f))
            push!(subformulas_st, (s, f_standard))
        end
    end

    return subformulas, subformulas_st
end

function find_P_dim(f0, formulas, dims)
    # here we check that each index has one and only one dimension
    for (s, p) in formulas
        f = [f0[i] for i in p]
        for (i, j) in zip(f0, f)
            if i in keys(dims)
                dims[j] = dims[i]
            end
            if j in keys(dims)
                dims[i] = dims[j]
            end
        end
    end

    dims = [dims[i] for i in f0]

    full_base = product([1:d for d in dims]...)  # (0, 0, 0), (0, 0, 1), (0, 0, 2), ... (d1, d2, d3)
    # len(full_base) degrees of freedom in an unconstrained tensor

    # but there is constraints given by the group `formulas`
    # For instance if `ij=-ji`, then 00=-00, 01=-01 and so on
    base = Set()
    for x in full_base
        # T[x] is a coefficient of the tensor T and is related to other coefficient T[y]
        # if x and y are related by a formula
        xs = Set((s, [x[i] for i in p]) for (s, p) in formulas)
        # s * T[x] are all equal for all (s, x) in xs
        # if T[x] = -T[x] it is then equal to 0 and we lose this degree of freedom
        if !((-1, [x...]) in xs)
            # the sign is arbitrary, put both possibilities
            push!(base, Set([
                xs,
                Set((-s, x) for (s, x) in xs)
            ]))
        end
    end

    return length(base) * prod(dims)
end

function find_P(f0, formulas, dims)
    """
    should be called like
    find_P(f0, formulas, Dict('i'=>2))
    """
    # here we check that each index has one and only one dimension
    for (s, p) in formulas
        f = [f0[i] for i in p]
        for (i, j) in zip(f0, f)
            if i in keys(dims) && j in keys(dims) && dims[i] != dims[j]
                throw(error("dimension of $i and $j should be the same"))
            end
            if i in keys(dims)
                dims[j] = dims[i]
            end
            if j in keys(dims)
                dims[i] = dims[j]
            end
        end
    end
    for i in f0
        if !(i in keys(dims))
            throw(error("index $i has no dimension associated to it"))
        end
    end

    dims = [dims[i] for i in f0]

    full_base = product([1:d for d in dims]...)  # (0, 0, 0), (0, 0, 1), (0, 0, 2), ... (d1, d2, d3)
    # len(full_base) degrees of freedom in an unconstrained tensor

    # but there is constraints given by the group `formulas`
    # For instance if `ij=-ji`, then 00=-00, 01=-01 and so on
    base = Set()
    for x in full_base
        # T[x] is a coefficient of the tensor T and is related to other coefficient T[y]
        # if x and y are related by a formula
        xs = Set((s, [x[i] for i in p]) for (s, p) in formulas)
        # s * T[x] are all equal for all (s, x) in xs
        # if T[x] = -T[x] it is then equal to 0 and we lose this degree of freedom
        if !((-1, [x...]) in xs)
            # the sign is arbitrary, put both possibilities
            push!(base, Set([
                xs,
                Set((-s, x) for (s, x) in xs)
            ]))
        end
    end

    # len(base) is the number of degrees of freedom in the tensor.
    base = sort([sort([sort([xs...]) for xs in x]) for x in base])

    # First we compute the change of basis (projection) between full_base and base
    d_sym = length(base)
    P = zeros(d_sym, length(full_base))

    for (i, x) in enumerate(base)
        mapped = map((xs) -> sum([s for (s, x_t) in xs]), x)
        x2 = x[argmax(mapped)]
        for (s, e) in x2
            j = 0
            for (k, d) in zip(e, dims)
                j *= d
                j += (k-1)
            end
            P[i, j+1] = s / length(x2)^0.5
        end
    end
    
    P = reshape(P, d_sym, dims...)
    return P
end

function find_R_old(irreps, filter_ir_mid=nothing, filter_ir_out=nothing)
    Rs = Dict()

    for (ir, base_o3) in _wigner_nj(irreps; filter_ir_mid=filter_ir_mid)
        if filter_ir_out === nothing || ir in filter_ir_out
            if !(ir in keys(Rs))
                Rs[ir] = []
            end
            push!(Rs[ir], base_o3)
        end
    end

    return Rs
end

function find_R(irreps1, irreps2, Q1, Q2, filter_ir_out=nothing)
	Rs = Dict() # dictionary of irreps -> matrix
	irreps_out = []
	k1 = 1
	for mul_ir1 in irreps1
        sub_Q1 = selectdim(Q1, 1, k1:k1 + o3.dim(mul_ir1) - 1)
		sub_Q1 = reshape(sub_Q1, mul_ir1.mul, o3.dim(mul_ir1.ir), :)
		k1 += o3.dim(mul_ir1)
		k2 = 1
		for mul_ir2 in irreps2
            sub_Q2 = selectdim(Q2, 1, k2:k2 + o3.dim(mul_ir2) - 1)
			sub_Q2 = reshape(sub_Q2, mul_ir2.mul, o3.dim(mul_ir2.ir), :)
			k2 += o3.dim(mul_ir2)
			for ir_out in mul_ir1.ir * mul_ir2.ir
				push!(irreps_out, o3.MulIr(mul_ir1.mul * mul_ir2.mul, ir_out))
                cg = Wigner.wigner_3j(mul_ir1.ir.l, mul_ir2.ir.l, ir_out.l)
                ### einsums
                m, i, a = size(sub_Q1)
                n, j, b = size(sub_Q2)
                k = size(cg, 3)
                # @einsum C_tmp[j, k, m, a] := sub_Q1[m, i, a] * cg[i, j, k]
                C_tmp = zeros(j, k, m, a)
                for em in 1:m
                    for ej in 1:j
                        for ek in 1:k
                            c = zeros(a)
                            for ei in 1:i
                                c += view(sub_Q1, em, ei, :) * cg[ei, ej, ek]
                            end
                            C_tmp[ej, ek, em, :] = c
                        end
                    end
                end
                # @einsum C[m, n, k, a, b] := sub_Q2[n, j, b] * C_tmp[j, k, m, a]
                C = zeros(m * n, k, a, b)
                for em in 1:m
                    for en in 1:n
                        for ek = 1:k
                            for ea in 1:a
                                c = zeros(b)
                                for ej in 1:j
                                    c += view(sub_Q2, en, ej, :) * C_tmp[ej, ek, em, ea]
                                end
                                C[em + (en-1) * m, ek, ea, :] = c
                            end
                        end
                    end
                end
                C = reshape(C, size(C, 1), size(C, 2), size(Q1)[2:end]..., size(Q2)[2:end]...)
                if filter_ir_out === nothing || ir_out in filter_ir_out
                    if !(ir_out in keys(Rs))
                        Rs[ir_out] = []
                    end
                    for i in 1:size(C, 1)
                        push!(Rs[ir_out], selectdim(C, 1, i))
                    end
                end
            end
        end
    end
	return o3.simplify(o3.Irreps(sort(irreps_out))), Rs
    # return Rs
end

function find_R_dist(irreps1, irreps2, Q1, Q2, filter_ir_out=nothing)
	Rs = Dict() # dictionary of irreps -> matrix
	irreps_out = []
	k1 = 1
	for mul_ir1 in irreps1
        sub_Q1 = selectdim(Q1, 1, k1:k1 + o3.dim(mul_ir1) - 1)
		sub_Q1 = reshape(sub_Q1, mul_ir1.mul, o3.dim(mul_ir1.ir), :)
		k1 += o3.dim(mul_ir1)
		k2 = 1
		for mul_ir2 in irreps2
            sub_Q2 = selectdim(Q2, 1, k2:k2 + o3.dim(mul_ir2) - 1)
			sub_Q2 = reshape(sub_Q2, mul_ir2.mul, o3.dim(mul_ir2.ir), :)
			k2 += o3.dim(mul_ir2)
			for ir_out in mul_ir1.ir * mul_ir2.ir
				push!(irreps_out, o3.MulIr(mul_ir1.mul * mul_ir2.mul, ir_out))
                cg = Wigner.wigner_3j(mul_ir1.ir.l, mul_ir2.ir.l, ir_out.l)
                ### einsums
                m, i, a = size(sub_Q1)
                n, j, b = size(sub_Q2)
                k = size(cg, 3)
                # @einsum C_tmp[j, k, m, a] := sub_Q1[m, i, a] * cg[i, j, k]
                C_tmp = zeros(j, k, m, a)
                for em in 1:m
                    for ej in 1:j
                        for ek in 1:k
                            c = zeros(a)
                            for ei in 1:i
                                c += view(sub_Q1, em, ei, :) * cg[ei, ej, ek]
                            end
                            C_tmp[ej, ek, em, :] = c
                        end
                    end
                end
                # @einsum C[m, n, k, a, b] := sub_Q2[n, j, b] * C_tmp[j, k, m, a]
                C = zeros(m * n, k, a, b)
                for em in 1:m
                    for en in 1:n
                        for ek = 1:k
                            for ea in 1:a
                                c = zeros(b)
                                for ej in 1:j
                                    c += view(sub_Q2, en, ej, :) * C_tmp[ej, ek, em, ea]
                                end
                                C[em + (en-1) * m, ek, ea, :] = c
                            end
                        end
                    end
                end
                C = reshape(C, size(C, 1), size(C, 2), size(Q1)[2:end]..., size(Q2)[2:end]...)
                if filter_ir_out === nothing || ir_out in filter_ir_out
                    if !(ir_out in keys(Rs))
                        Rs[ir_out] = []
                    end
                    for i in 1:size(C, 1)
                        push!(Rs[ir_out], selectdim(C, 1, i))
                    end
                end
            end
        end
    end
	return o3.simplify(o3.Irreps(sort(irreps_out))), Rs
    # return Rs
end

@everywhere function append2(t1, t2)
    append!(t1[1], t2[1])
    append!(t1[2], t2[2])
    return t1
end
function append2(t1, t2)
    append!(t1[1], t2[1])
    append!(t1[2], t2[2])
    return t1
end

function find_Q_serial(P, Rs, ε=1e-9)
    Q = []
    irreps_out = []

    PP = P * transpose(P)  # (a,a)
    
    for ir in keys(Rs)
        mul = length(Rs[ir])
        # base_o3/R == clebsch-gordan basis
        base_o3 = cat(Rs[ir]..., dims = ndims(Rs[ir][1])+1) # this is slow
        base_o3 = permutedims(base_o3, [ndims(base_o3), 1:ndims(base_o3)-1...])

        R = reshape(base_o3, size(base_o3, 1), size(base_o3, 2), :)  # [multiplicity, ir, input basis] (u,j,omega)

        ### Solve X @ R[:, j] = Y @ P, but keep only X
        RR = view(R, :, 1, :) * transpose(view(R, :, 1, :))  # (u,u)
        RP = view(R, :, 1, :) * transpose(P)  # (u,a)

        prob = cat(cat(RR, -RP, dims=2), cat(transpose(-RP), PP, dims=2), dims=1) # this is slow
        eigenvalues, eigenvectors = eigen(prob)
        eigvec_filtered = eigenvectors[:, map(λ -> λ < ε, eigenvalues)]
        if length(eigvec_filtered) > 0
            X = eigvec_filtered[1:mul, :]  # [solutions, multiplicity]
            proj_s = X * transpose(X)
        else
            proj_s = [0.0;;]
        end

        # look for an X such that Xᵀ * X = Projector
        X, _ = orthonormalize(proj_s, ε)

        for x in X
            C = sum([x[ndx] .* selectdim(base_o3, 1, ndx) for ndx in 1:length(x)])
            # C = torch.einsum("u,ui...->i...", x, base_o3)
            correction = (o3.dim(ir) / sum(C.^2))^0.5
            C = correction * C

            push!(Q, C)
            push!(irreps_out, (1, ir))
        end
    end

    irreps_out = o3.simplify(o3.Irreps(irreps_out))
    Q = vcat(Q...)
    return irreps_out, Q
end

function find_Q_dist(P, Rs, ε=1e-9)
    PP = P * transpose(P)  # (a,a)
    
    irreps_out, Q = @distributed (append2) for ir in [keys(Rs)...]
        mul = length(Rs[ir])
        # base_o3/R == clebsch-gordan basis
        base_o3 = cat(Rs[ir]..., dims = ndims(Rs[ir][1])+1)
        base_o3 = permutedims(base_o3, [ndims(base_o3), 1:ndims(base_o3)-1...])

        R = reshape(base_o3, size(base_o3, 1), size(base_o3, 2), :)  # [multiplicity, ir, input basis] (u,j,omega)

        ### Solve X @ R[:, j] = Y @ P, but keep only X
        RR = view(R, :, 1, :) * transpose(view(R, :, 1, :))  # (u,u)
        RP = view(R, :, 1, :) * transpose(P)  # (u,a)

        prob = cat(cat(RR, -RP, dims=2), cat(transpose(-RP), PP, dims=2), dims=1)
        eigenvalues, eigenvectors = eigen(prob)
        eigvec_filtered = eigenvectors[:, map(λ -> λ < ε, eigenvalues)]
        if length(eigvec_filtered) > 0
            X = eigvec_filtered[1:mul, :]  # [solutions, multiplicity]
            proj_s = X * transpose(X)
        else
            proj_s = [0.0;;]
        end

        # look for an X such that Xᵀ * X = Projector
        X, _ = orthonormalize(proj_s, ε)

        Q_tmp = []
        mul_out = 0

        for x in X
            C = sum([x[ndx] .* selectdim(base_o3, 1, ndx) for ndx in 1:length(x)])
            # C = torch.einsum("u,ui...->i...", x, base_o3)
            correction = (o3.dim(ir) / sum(C.^2))^0.5
            C = correction * C
            push!(Q_tmp, C)
            mul_out += 1
        end
        ([(mul_out, ir)], Q_tmp)
    end

    Q = vcat(Q...)
    irreps_out = o3.simplify(o3.Irreps(irreps_out))
    return irreps_out, Q
end

function reduced_product(formula, irreps, filter_ir_out=nothing, filter_ir_mid=nothing, ε=1e-9; parallel=true)
    """original, serial + no DQ"""
    if filter_ir_out !== nothing
        try
            filter_ir_out = [o3.Irrep(ir) for ir in filter_ir_out]
        catch
            throw(error("filter_ir_out (=$filter_ir_out) must be an iterable of Irrep"))
        end
    end
    if filter_ir_mid !== nothing
        try
            filter_ir_mid = [o3.Irrep(ir) for ir in filter_ir_mid]
        catch
            throw(error("filter_ir_mid (=$filter_ir_mid) must be an iterable of Irrep"))
        end
    end
    
    f0, formulas = germinate_formulas(formula) # create representations for all equivalent index permutations

    # set irrep indices
    irreps = Dict(i => o3.Irreps(irs) for (i, irs) in irreps) # keys: char; values: Irreps
    for i in keys(irreps)
        if length(i) != 1
            throw(error("got an unexpected keyword argument '$i'"))
        end
    end
    for (sign, p) in formulas
        f = join([f0[i] for i in p])
        for (i, j) in zip(f0, f)
            if i in keys(irreps) && j in keys(irreps) && irreps[i] != irreps[j]
                throw(error("irreps of $i and $j should be the same"))
            end
            if i in keys(irreps)
                irreps[j] = irreps[i]
            end
            if j in keys(irreps)
                irreps[i] = irreps[j]
            end
        end
    end
    for i in f0
        if !(i in keys(irreps))
            throw(error("index $i has no irreps associated to it"))
        end
    end
    for i in keys(irreps)
        if !(i in f0)
            throw(error("index $i has an irreps but does not appear in the formula"))
        end
    end

    # permutation basis
    base_perm = find_P(f0, formulas, Dict(i => o3.dim(irs) for (i, irs) in irreps)) # same size as output
    P = reshape(base_perm, size(base_perm, 1), :)  # [permutation basis, input basis] (a,omega)
    # rotation basis
    Rs = find_R_old([irreps[i] for i in f0], filter_ir_mid, filter_ir_out)

    irreps_in = [irreps[i] for i in f0]

    if parallel
        irreps_out, Q = find_Q_dist(P, Rs)
    else
        irreps_out, Q = find_Q_serial(P, Rs)
    end

    return irreps_in, irreps_out, Q
end

function reduced_product_dq(formula, irreps, filter_ir_out=nothing, filter_ir_mid=nothing, ε=1e-9; parallel=true)
    """divide and conquer - multiple"""
    if filter_ir_out !== nothing
        try
            filter_ir_out = [o3.Irrep(ir) for ir in filter_ir_out]
        catch
            throw(error("filter_ir_out (=$filter_ir_out) must be an iterable of Irrep"))
        end
    end
    if filter_ir_mid !== nothing
        try
            filter_ir_mid = [o3.Irrep(ir) for ir in filter_ir_mid]
        catch
            throw(error("filter_ir_mid (=$filter_ir_mid) must be an iterable of Irrep"))
        end
    end
    
    f0, formulas = germinate_formulas(formula) # create representations for all equivalent index permutations

    return _rtp_dq(f0, formulas, irreps, filter_ir_out, filter_ir_mid, ε, parallel=parallel)
end

function _rtp_dq(f0, formulas, irreps, filter_ir_out=nothing, filter_ir_mid=nothing, ε=1e-9; parallel=false) # for caching, f0 => len(f0), irreps => list
    """divide and conquer - remove 1 index at a time"""
    # base case
    if length(f0) == 1
        ir = o3.Irreps(irreps[only(f0)])
        return o3.Irreps(ir), o3.Irreps(ir), I(o3.dim(ir))
    end
    # set irrep indices
    irreps = Dict(i => o3.Irreps(irs) for (i, irs) in irreps) # keys: char; values: Irreps
    for i in keys(irreps)
        if length(i) != 1
            throw(error("got an unexpected keyword argument '$i'"))
        end
    end
    for (sign, p) in formulas
        f = join([f0[i] for i in p])
        for (i, j) in zip(f0, f)
            if i in keys(irreps) && j in keys(irreps) && irreps[i] != irreps[j]
                throw(error("irreps of $i and $j should be the same"))
            end
            if i in keys(irreps)
                irreps[j] = irreps[i]
            end
            if j in keys(irreps)
                irreps[i] = irreps[j]
            end
        end
    end
    for i in f0
        if !(i in keys(irreps))
            throw(error("index $i has no irreps associated to it"))
        end
    end
    for i in keys(irreps)
        if !(i in f0)
            throw(error("index $i has an irreps but does not appear in the formula"))
        end
    end

    ### find optimal subformulas
    best_subindices = nothing
    D_curr = -1
    for subindices in subsets(1:length(f0))
        if length(subindices) > 0 && length(subindices) < length(f0)
            f1 = f0[subindices]
            f2 = f0[[i for i in 1:length(f0) if !(i in subindices)]]
            _, formulas1 = subformulas(f0, formulas, f1)
            _, formulas2 = subformulas(f0, formulas, f2)
            p1 = find_P_dim(f1, formulas1, Dict(i => o3.dim(irreps[i]) for i in f1))
            p2 = find_P_dim(f2, formulas2, Dict(i => o3.dim(irreps[i]) for i in f2))
            if p1 * p2 < D_curr || D_curr == -1
                D_curr = p1 * p2
                best_subindices = subindices[:]
            end
        end
    end
    f1 = f0[best_subindices]
    f2 = f0[[i for i in 1:length(f0) if !(i in best_subindices)]]
    formulas1_orig, formulas1 = subformulas(f0, formulas, f1)
    formulas2_orig, formulas2 = subformulas(f0, formulas, f2)

    ### bases from the full problem
    # permutation basis
    base_perm = find_P(f0, formulas, Dict(i => o3.dim(irs) for (i, irs) in irreps)) # same size as output
    P = reshape(base_perm, size(base_perm, 1), :)  # [permutation basis, input basis] (a,omega)

    ### Qs from subproblems (irrep outputs)
    _, out1, Q1 = _rtp_dq(f1, formulas1, Dict(c => irreps[c] for c in f1), filter_ir_out, filter_ir_mid, ε, parallel=parallel)
    _, out2, Q2 = _rtp_dq(f2, formulas2, Dict(c => irreps[c] for c in f2), filter_ir_out, filter_ir_mid, ε, parallel=parallel)
    
    ### combine Q1 and Q2
    irreps_out, R = find_R(out1, out2, Q1, Q2, filter_ir_out)

    irreps_in = [irreps[i] for i in f0]
    
    ### if all symmetries are already accounted for, find_Q isn't necessary
    # R needs to be turned into an array
    # if size(P, 1) == sum(map(v -> length(v), values(R))...)
    #     return irreps_in, irreps_out, R
    # end

    ### otherwise, take extra global symmetries into account
    if parallel
        irreps_out, Q = find_Q_dist(P, R)
    else
        irreps_out, Q = find_Q_serial(P, R)
    end

    return irreps_in, irreps_out, Q
end

export reduced_product, reduced_product_dq

end