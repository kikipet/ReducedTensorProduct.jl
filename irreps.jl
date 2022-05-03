module o3

using LinearAlgebra
include("wigner.jl")

struct Irrep
    """Irreducible representation of O(3).
    l: degree
    p: parity; 1 if even, -1 if odd"""
    l::Int
    p::Int
end

struct MulIr
    """Represents multiple copies of a single irrep.
    mul: multiplicity
    ir: single Irrep
    """
    mul::Int
    ir::Irrep
end

struct Irreps
    """Represents the sum of multiple irreps."""
    mulirs::Vector{MulIr}
end

function Irrep(l)
    """Creates an Irrep object."""
    if typeof(l) == Irrep
        return l
    end

    if typeof(l) == String || typeof(l) == SubString{String}
        try
            name = strip(l)
            l = parse(Int, name[1:length(name)-1])
            @assert l >= 0
            p = Dict(
                'e' => 1,
                'o' => -1,
                'y' => (-1)^l,
            )[name[length(name)]]
        catch
            println("unable to convert string \"$name\" into an Irrep")
            throw(error())
        end
    elseif typeof(l) == Tuple{Any}
        l, p = l
    end

    @assert typeof(l) == Int && l >= 0
    @assert p in [-1, 1]
    return Irrep(l, p)
end

function dim(I::Irrep)
    """Calculates dimension of a given irrep."""
    return 2 * I.l + 1
end

function dim(I::MulIr)
    """Calculates dimension of a multiple irrep."""
    return I.mul * dim(I.ir)
end

function dim(I::Irreps)
    """Calculates dimension of a direct sum of irreps."""
    d = 0
    for mul_ir in I
        d += mul_ir.mul * dim(mul_ir.ir)
    end
    return d
end

function dim(I::Vector)
    """Calculates dimension of a direct sum of irreps."""
    d = 0
    for mul_ir in I
        d += mul_ir.mul * dim(mul_ir.ir)
    end
    return d
end

function Irreps(irreps)
    """Creates an Irreps object."""
    out = []
    if typeof(irreps) == Irrep
        push!(out, MulIr(1, irreps))
    elseif typeof(irreps) == Irreps
        return irreps
    elseif typeof(irreps) == String
        try
            if strip(irreps) != ""
                for mul_ir in split(irreps, '+')
                    mul_ir = strip(mul_ir)
                    if 'x' in mul_ir
                        split_mul = split(mul_ir, 'x')
                        mul, ir = split(mul_ir, 'x')
                        mul = parse(Int, mul)
                        ir = Irrep(ir)
                    else
                        mul = 1
                        ir = Irrep(mul_ir)
                    end

                    @assert typeof(mul) == Int && mul >= 0
                    push!(out, MulIr(mul, ir))
                end
            end
        catch
            println("Unable to convert string \"$irreps\" into an Irreps")
            throw(error())
        end
    else
        for mul_ir in irreps
            mul = nothing
            ir = nothing

            if typeof(mul_ir) == String
                mul = 1
                ir = Irrep(mul_ir)
            elseif typeof(mul_ir) == Irrep
                mul = 1
                ir = mul_ir
            elseif typeof(mul_ir) == MulIr
                mul = mul_ir.mul
                ir = mul_ir.ir
            elseif length(mul_ir) == 2
                mul, ir = mul_ir
                ir = Irrep(ir)
            end
            if !(typeof(mul) == Int && mul >= 0 && ir !== nothing)
                println("Unable to interpret \"$mul_ir\" as an irrep.")
                throw(error())
            end
            push!(out, MulIr(mul, ir))
        end
    end
    return out
end

function irrep_in(irrep, irreps)
    """Determine whether an irrep is included in a given Irreps object."""
    for mul_ir in irreps
        if irrep == mul_ir.ir
            return true
        end
    end
    return false
end

Base.:+(f::Irrep, g::Irrep) = Irreps(f) + Irreps(g)

function Base.:*(f::Irrep, g::Irrep)
    p = f.p * g.p
    lmin = abs(f.l - g.l)
    lmax = f.l + g.l
    return [Irrep(l, p) for l in lmin:lmax]
end

function Base.:*(f::Int, g::Irrep)
    return Irreps([(f, g)])
end
function Base.:*(f::Irrep, g::Int)
    return Irreps([(g, f)])
end

function Base.:>(f::Irrep, g::Irrep)
    if f.l > g.l
        return true
    end
    return f.l == g.l && f.p > g.p
end

function Base.:<(f::Irrep, g::Irrep)
    if f.l < g.l
        return true
    end
    return f.l == g.l && f.p < g.p
end

function Base.:>=(f::Irrep, g::Irrep)
    if f.l > g.l
        return true
    end
    return f.l == g.l && f.p >= g.p
end

function Base.:<=(f::Irrep, g::Irrep)
    if f.l < g.l
        return true
    end
    return f.l == g.l && f.p <= g.p
end

function Base.:isless(f::Irrep, g::Irrep)
    if f.l < g.l
        return true
    end
    return f.l == g.l && f.p < g.p
end

function Base.:isless(f::MulIr, g::MulIr)
    return f.ir < g.ir
end

function simplify(irreps)
    out = []
    for mul_ir in irreps
        if length(out) > 0 && out[length(out)][2] == mul_ir.ir
            out[length(out)] = (out[length(out)][1] + mul_ir.mul, mul_ir.ir)
        elseif mul_ir.mul > 0
            push!(out, (mul_ir.mul, mul_ir.ir))
        end
    end
    return Irreps(out)
end


function direct_sum(matrices)
    """Direct sum of matrices, put them in the diagonal
    """
    front_indices = size(matrices[1])[1:ndims(matrices)-2]
    m = sum(size(x, ndims(x)-1) for x in matrices)
    n = sum(size(x, ndims(x)) for x in matrices)
    out = zeros(front_indices..., m, n)
    i, j = 1, 1
    for x in matrices
        m = size(x, ndims(x) - 1)
        n = size(x, ndims(x))
        out[[1:f for f in front_indices]..., i:i + m - 1, j:j + n - 1] = x
        i += m
        j += n
    end
    return out
end

function D_from_angles_irrep(irrep, alpha, beta, gamma, k=nothing)
    """Matrix :math:`p^k D^l(\\alpha, \\beta, \\gamma)`
    (matrix) Representation of :math:`O(3)`. :math:`D` is the representation of :math:`SO(3)`, see `wigner_D`.
    Parameters
    ----------
    alpha : `torch.Tensor`
        tensor of shape :math:`(...)`
        Rotation :math:`\\alpha` around Y axis, applied third.
    beta : `torch.Tensor`
        tensor of shape :math:`(...)`
        Rotation :math:`\\beta` around X axis, applied second.
    gamma : `torch.Tensor`
        tensor of shape :math:`(...)`
        Rotation :math:`\\gamma` around Y axis, applied first.
    k : `torch.Tensor`, optional
        tensor of shape :math:`(...)`
        How many times the parity is applied.
    Returns
    -------
    `torch.Tensor`
        tensor of shape :math:`(..., 2l+1, 2l+1)`
    See Also
    --------
    o3.wigner_D
    Irreps.D_from_angles
    """
    if k === nothing
        k = zeros(size(alpha)...)
    end

    alpha = broadcast((α, β, γ, k) -> α, alpha, beta, gamma, k)
    beta = broadcast((α, β, γ, k) -> β, alpha, beta, gamma, k)
    gamma = broadcast((α, β, γ, k) -> γ, alpha, beta, gamma, k)
    k = broadcast((α, β, γ, k) -> k, alpha, beta, gamma, k)
    return Wigner.wigner_D(irrep.l, alpha, beta, gamma) .* reshape([irrep.p^k], 1, 1)
end

function D_from_angles_irreps(irreps, alpha, beta, gamma, k=nothing)
    """Matrix of the representation
    Parameters
    ----------
    alpha : `torch.Tensor`
        tensor of shape :math:`(...)`
    beta : `torch.Tensor`
        tensor of shape :math:`(...)`
    gamma : `torch.Tensor`
        tensor of shape :math:`(...)`
    k : `torch.Tensor`, optional
        tensor of shape :math:`(...)`
    Returns
    -------
    `torch.Tensor`
        tensor of shape :math:`(..., \\mathrm{dim}, \\mathrm{dim})`
    """
    return direct_sum([D_from_angles_irrep(mul_ir.ir, alpha, beta, gamma, k) for mul_ir in irreps for _ in 1:mul_ir.mul])
end

function rand_angles(shape)
    """random rotation angles

    Parameters
    ----------
    *shape : int

    Returns
    -------
    alpha : `torch.Tensor`
        tensor of shape :math:`(\\mathrm{shape})`

    beta : `torch.Tensor`
        tensor of shape :math:`(\\mathrm{shape})`

    gamma : `torch.Tensor`
        tensor of shape :math:`\\mathrm{shape})`
    """
    alpha = 2 * pi * rand(shape...)
    gamma = 2 * pi * rand(shape...)
    beta = acos.(rand(shape...) .* 2 .- 1)
    return alpha, beta, gamma
end

function spherical_harmonics(lmax, p=-1)
    """representation of the spherical harmonics
    Parameters
    ----------
    lmax : int
        maximum :math:`l`
    p : {1, -1}
        the parity of the representation
    Returns
    -------
    `e3nn.o3.Irreps`
        representation of :math:`(Y^0, Y^1, \\dots, Y^{\\mathrm{lmax}})`
    Examples
    --------
    >>> Irreps.spherical_harmonics(3)
    1x0e+1x1o+1x2e+1x3o
    >>> Irreps.spherical_harmonics(4, p=1)
    1x0e+1x1e+1x2e+1x3e+1x4e
    """
    return Irreps([(1, (l, p^l)) for l in 0:lmax])
end

end