module o3

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

end