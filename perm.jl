using LinearAlgebra

function complete_basis(vecs, eps = 1e-9)
    @assert length(vecs) == 2
    dim = size(vecs)[2] # not sure if this is how things will be formatted

    base = [x / norm(x) for x in vecs]

    expand = []
    for x in ones(dim)
        for y in base + expand
            x -= torch.dot(x, y) * y
        end
        if norm(x) > 2 * eps
            x /= norm(x)
            x[x.abs() < eps] = x.new_zeros(())
            x *= x[x.nonzero()[0, 0]].sign()
            expand += [x]
        end
    end

    if length(expand) > 0
        expand = transpose(hcat(expand))
    else
        expand = zeros(0, dim) # not sure; originally vecs.new_zeros(0, dim)
    end

    return expand
end

function is_perm(p)
    return Set(p) == Set(1:length(p))
end

function identity(n)
    return Vector(1:n)
end

function compose(p1, p2)
    """
    compute p1 . p2
    """
    @assert is_perm(p1) && is_perm(p2)
    @assert length(p1) == length(p2)
    # p: i |-> p[i]

    # [p1.p2](i) = p1(p2(i)) = p1[p2[i]]
    return [p1[p2[i]] for i in 1:length(p1)]
end


function inverse(p)
    """
    compute the inverse permutation
    """
    return [findfirst(isequal(i), p) for i in 1:length(p)]
    # return [p.index(i) for i in 0:length(p)-1]
end


function rand(n)
    i = random.randint(0, math.factorial(n) - 1)
    return from_int(i, n)
end


function from_int(i, n)
    pool = list(range(n))
    p = []
    for _ in range(n)
        j = i % n
        i = i // n
        p.append(pool.pop(j))
        n -= 1
    end
    return tuple(p)
end


function to_int(p)
    n = len(p)
    pool = list(range(n))
    i = 0
    m = 1
    for j in p
        k = pool.index(j)
        i += k * m
        m *= len(pool)
        pool.pop(k)
    end
    return i
end


function group(n)
    return {from_int(i, n) for i in range(math.factorial(n))}
end


function germinate(subset)
    while true
        n = len(subset)
        subset = subset.union([inverse(p) for p in subset])
        subset = subset.union([
            compose(p1, p2)
            for p1 in subset
            for p2 in subset
        ])
        if len(subset) == n
            return subset
        end
    end
end


function is_group(g)
    if length(g) == 0
        return false

    n = len(next(iter(g)))

    for p in g
        @assert len(p) == n, p
    end

    if not identity(n) in g
        return false
    end

    for p in g
        if not inverse(p) in g
            return false
        end
    end

    for p1 in g
        for p2 in g
            if not compose(p1, p2) in g
                return false
            end
        end
    end

    return true
end


function to_cycles(p)
    n = length(p)

    cycles = set()

    for i in range(n)
        c = [i]
        while p[i] != c[0]
            i = p[i]
            c += [i]
        end
        if length(c) >= 2
            i = c.index(min(c))
            c = c[i:] + c[1:i]
            cycles.add(tuple(c))
        end
    end

    return cycles
end


function sign(p)
    s = 1
    for c in to_cycles(p)
        if length(c) % 2 == 0
            s = -s
        end
    end
    return s
end


function standard_representation(p, dtype = nothing, device = nothing)
    """irrep of Sn of dimension n - 1
    """
    A = complete_basis(torch.ones(1, len(p), dtype=dtype, device=device), eps=0.1 / len(p))
    return A @ natural_representation(p) @ A.T
end


function natural_representation(p, dtype = nothing, device = nothing)
    """natural representation of Sn
    """
    n = len(p)
    ip = inverse(p)
    d = torch.zeros(n, n, dtype=dtype, device=device)

    for a in range(n)
        d[a, ip[a]] = 1
    end

    return d
end
