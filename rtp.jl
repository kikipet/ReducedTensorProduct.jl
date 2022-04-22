using IterTools
using LinearAlgebra
include("irreps.jl")
include("wigner.jl")


function _wigner_nj(irrepss; normalization="component", filter_ir_mid=nothing)
    irrepss = [o3.Irreps(irreps) for irreps in irrepss]
    if filter_ir_mid !== nothing
        filter_ir_mid = [o3.Irrep(ir) for ir in filter_ir_mid]
    end

    if length(irrepss) == 1
        irreps = irrepss[1]
        ret = []
        i_dim = o3.dim(irreps)
        # e = [[convert(Float64, ndx1==ndx2) for ndx2 in 1:i_dim] for ndx1 in 1:i_dim]
        e = Array{Float64}(undef, i_dim, i_dim)
        for ndx1 in 1:i_dim
            for ndx2 in 1:i_dim
                e[ndx1, ndx2] = convert(Float64, ndx1 == ndx2)
            end
        end
        i = 1
        for mul_ir in irreps
            for _ in 1:mul_ir.mul
                stop = i + o3.dim(mul_ir.ir) - 1
                push!(ret, (mul_ir.ir, wigner.input(0, i, stop), e[i:stop, :]))
                i += o3.dim(mul_ir.ir)
            end
        end
        return ret
    end

    irreps_right = irrepss[length(irrepss)]
    irrepss_left = irrepss[1:length(irrepss)-1]
    ret = []
    for (ir_left, path_left, C_left) in _wigner_nj(irrepss_left, normalization=normalization, filter_ir_mid=filter_ir_mid)
        i = 0
        for mul_ir in irreps_right
            for ir_out in ir_left * mul_ir.ir
                if filter_ir_mid !== nothing && !(ir_out in filter_ir_mid)
                    continue
                end

                C = wigner.wigner_3j(ir_out.l, ir_left.l, mul_ir.ir.l)
                if normalization == "component"
                    C *= o3.dim(ir_out)^0.5
                end
                if normalization == "norm"
                    C *= o3.dim(ir_left)^0.5 * o3.dim(mul_ir.ir)^0.5
                end

                ### begin einsum: C = einsum("jk,ijl->ikl", C_left.flatten(1), C)
                C_left_flat = reshape(C_left, size(C_left, 1), :)
                s1 = [size(C_left_flat, 1), length(C_left_flat[1])] # temporary
                s2 = size(C_left_flat) # temporary
                Clf_size = size(C_left_flat)
                C_size = size(C)
                @assert Clf_size[1] == C_size[2]
                C_prod = zeros(C_size[1], Clf_size[2], C_size[3])
                for ein_k in 1:Clf_size[2]
                    for ein_i in 1:C_size[1]
                        for ein_l in 1:C_size[3]
                            ein_total = 0
                            for ein_j in 1:C_size[2]
                                ein_total += C_left_flat[ein_j, ein_k] * C[ein_i, ein_j, ein_l]
                            end
                            C_prod[ein_i, ein_k, ein_l] = ein_total
                        end
                    end
                end
                ### end einsum
                C = reshape(C_prod, o3.dim(ir_out), [o3.dim(irreps) for irreps in irrepss_left]..., o3.dim(mul_ir.ir))
                for u in 1:mul_ir.mul
                    E = zeros(o3.dim(ir_out), [o3.dim(irreps) for irreps in irrepss_left]..., o3.dim(irreps_right))
                    start = i + (u-1) * o3.dim(mul_ir.ir) + 1
                    stop = i + u * o3.dim(mul_ir.ir)
                    # access last dimension
                    ndims = length(size(E))
                    E[[1:d for d in size(E)[1:ndims-1]]..., start:stop] = C
                    push!(ret, (ir_out, wigner.tp((ir_left, mul_ir.ir, ir_out), (path_left, wigner.input(length(irrepss_left), start, stop))), E))
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

# examples:
# >>> germinate_formulas('ij')
# ('ij', {(1, (0, 1))})
# >>> germinate_formulas('ij=ji')
# ('ij', {(1, (0, 1)), (1, (1, 0))})
# >>> germinate_formulas('ij=-ji')
# ('ij', {(1, (0, 1)), (-1, (1, 0))})
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
            println("$f is not a permutation of $f0")
            throw(error())
        end
        if length(f0) != length(f)
            println("$f0 and $f don't have the same number of indices")
            throw(error())
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

# >>> Q, ret = reduce_permutation(*germinate_formulas("ij=-ji"), i=2)
# >>> Q
# tensor([[[ 0.0000, -0.7071],
#          [ 0.7071,  0.0000]]])
# >>> ret
# [[(-1, (0, 1)), (1, (1, 0))]]
function reduce_permutation(f0, formulas, dims)
    """
    should be called like
    reduce_permutation(f0, formulas, Dict('i'=>2))
    """
    # here we check that each index has one and only one dimension
    for (s, p) in formulas
        f = string([f0[i] for i in p]...)
        for (i, j) in zip(f0, f)
            if i in keys(dims) && j in keys(dims) && dims[i] != dims[j]
                println("dimension of $i and $j should be the same")
                throw(error())
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
            println("index $i has no dimension associated to it")
            throw(error())
        end
    end

    dims = [dims[i] for i in f0]

    full_base = product([1:d for d in dims]...)  # (0, 0, 0), (0, 0, 1), (0, 0, 2), ... (3, 3, 3)
    # len(full_base) degrees of freedom in an unconstrained tensor

    # but there is constraints given by the group `formulas`
    # For instance if `ij=-ji`, then 00=-00, 01=-01 and so on
    base = Set()
    for x in full_base
        # T[x] is a coefficient of the tensor T and is related to other coefficient T[y]
        # if x and y are related by a formula
        xs = Set([(s, [x[i] for i in p]) for (s, p) in formulas])
        # s * T[x] are all equal for all (s, x) in xs
        # if T[x] = -T[x] it is then equal to 0 and we lose this degree of freedom
        if !((-1, [x...]) in xs)
            # the sign is arbitrary, put both possibilities
            push!(base, Set([
                Set(xs),
                Set([(-s, x) for (s, x) in xs])
            ]))
        end
    end

    # len(base) is the number of degrees of freedom in the tensor.
    base = sort([sort([sort([xs...]) for xs in x]) for x in base])

    # First we compute the change of basis (projection) between full_base and base
    d_sym = length(base)
    Q = zeros(d_sym, length(full_base))
    ret = []

    for (i, x) in enumerate(base)
        mapped = map((xs) -> sum([s for (s, x_t) in xs]), x)
        x2 = x[argmax(mapped)]
        push!(ret, x2)
        for (s, e) in x2
            j = 0
            for (k, d) in zip(e, dims)
                j *= d
                j += (k-1)
            end
            Q[i, j+1] = s / length(x2)^0.5
        end
    end
    
    Q = reshape(Q, d_sym, dims...)
    return Q, ret
end

# >>> orthonormalize(torch.Tensor([[1, 1], [1, 0]]))
# (tensor([[ 0.7071,  0.7071],
#         [ 0.7071, -0.7071]]),
# tensor([[ 0.7071,  0.0000],
#         [-0.7071,  1.4142]]))
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
            # x[map(abs, x) < eps] .= 0
            # cx[map(abs, cx) < eps] .= 0
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

function reduced_tensor_product(formula, irreps, filter_ir_out=nothing, filter_ir_mid=nothing, ε=1e-9)
    if filter_ir_out !== nothing
        try
            filter_ir_out = [o3.Irrep(ir) for ir in filter_ir_out]
        catch
            println("filter_ir_out (=$filter_ir_out) must be an iterable of Irrep")
            throw(error())
        end
    end
    if filter_ir_mid !== nothing
        try
            filter_ir_mid = [o3.Irrep(ir) for ir in filter_ir_mid]
        catch
            println("filter_ir_mid (=$filter_ir_mid) must be an iterable of Irrep")
            throw(error())
        end
    end
    
    f0, formulas = germinate_formulas(formula)

    irreps = Dict(i => o3.Irreps(irs) for (i, irs) in irreps) # keys: char; values: Irreps

    for i in keys(irreps)
        if length(i) != 1
            println("got an unexpected keyword argument '$i'")
            throw(error())
        end
    end

    for (sign, p) in formulas
        f = join([f0[i] for i in p])
        for (i, j) in zip(f0, f)
            if i in keys(irreps) && j in keys(irreps) && irreps[i] != irreps[j]
                throw(RuntimeError("irreps of $i and $j should be the same"))
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
            println("index $i has no irreps associated to it")
            throw(error())
        end
    end

    for i in keys(irreps)
        if !(i in f0)
            println("index $i has an irreps but does not appear in the fomula")
            throw(error())
        end
    end

    base_perm, _ = reduce_permutation(f0, formulas, Dict(i => o3.dim(irs) for (i, irs) in irreps))

    Ps = Dict()

    # what exactly is going on here? i.e. why wigner
    for (ir, path, base_o3) in _wigner_nj([irreps[i] for i in f0]; filter_ir_mid=filter_ir_mid)
        if filter_ir_out === nothing || ir in filter_ir_out
            if !(ir in keys(Ps))
                Ps[ir] = []
            end
            push!(Ps[ir], (path, base_o3))
        end
    end

    outputs = []
    change_of_basis = []
    irreps_out = []

    P = reshape(base_perm, size(base_perm, 1), :)  # [permutation basis, input basis] (a,omega)
    PP = P * transpose(P)  # (a,a)
    
    for ir in keys(Ps)
        mul = length(Ps[ir])
        paths = [path for (path, _) in Ps[ir]]
        base_o3 = cat([R for (_, R) in Ps[ir]]..., dims = ndims(Ps[ir][1][2])+1)
        base_o3 = permutedims(base_o3, [ndims(base_o3), 1:ndims(base_o3)-1...])
        # base_o3 = torch.stack([R for (_, R) in Ps[ir]]) # python

        R = reshape(base_o3, size(base_o3, 1), size(base_o3, 2), :)  # [multiplicity, ir, input basis] (u,j,omega)

        proj_s = []  # list of projectors into vector space
        for j in range(1, o3.dim(ir))
            # Solve X @ R[:, j] = Y @ P, but keep only X
            RR = R[:, j, :] * transpose(R[:, j, :])  # (u,u)
            RP = R[:, j, :] * transpose(P)  # (u,a)

            prob = cat(cat(RR, -RP, dims=2), cat(transpose(-RP), PP, dims=2), dims=1)
            eigenvalues, eigenvectors = eigen(prob)
            eigvec_filtered = eigenvectors[:, map(λ -> λ < ε, eigenvalues)]
            if length(eigvec_filtered) > 0
                X = transpose(eigenvectors[:, map(λ -> λ < ε, eigenvalues)][1:mul])  # [solutions, multiplicity] # doesn't work if X is empty
                push!(proj_s, transpose(X) * X)
            else
                push!(proj_s, [0.0;;])
            end

            break  # do not check all components because too time expensive
        end

        for p in proj_s
            if max(map(abs, p - proj_s[1])...) > ε
                println("found different solutions for irrep $ir")
                throw(error())
            end
        end

        # look for an X such that Xᵀ * X = Projector
        X, _ = orthonormalize(proj_s[1], ε)

        for x in X
            C = sum([x[ndx] .* base_o3[ndx, [1:sz for sz in size(base_o3)[2:ndims(base_o3)]]...] for ndx in 1:length(x)])
            # C = torch.einsum("u,ui...->i...", x, base_o3)
            correction = (o3.dim(ir) / sum(C.^2))^0.5
            C = correction * C

            # anyway correction * v is supposed to be just one number
            push!(outputs, [(correction * v, p) for (v, p) in zip(x, paths) if abs(v) > ε])
            push!(change_of_basis, C)
            push!(irreps_out, (1, ir))
        end
    end

    return change_of_basis # this is, notably, not the tensor product itself
end
