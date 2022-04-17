module o3

struct Irrep
    l::Int
    p::Int
end

struct MulIr
    mul::Int
    ir::Irrep
end

struct Irreps
    mulirs::Vector{MulIr}
end

function Irrep(l)
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
    return 2 * I.l + 1
end

function dim(I::Irreps)
    d = 0
    for mul_ir in I
        d += mul_ir.mul * dim(mul_ir.ir)
    end
    return d
end

function dim(I::Vector)
    d = 0
    for mul_ir in I
        d += mul_ir.mul * dim(mul_ir.ir)
    end
    return d
end

function Irreps(irreps)
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