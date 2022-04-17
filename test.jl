using LinearAlgebra

original = [[1, 1], [1, 0]]
epsilon = 1e-9

@assert length(original) == 2
dim = length(original[1])

final = []
matrix = []

for (i, x) in enumerate(original)
    # x = sum_i cx_i original_i
    cx = zeros(length(original))
    cx[i] = 1
    for (j, y) in enumerate(final)
        c = sum(x .* y)
        x = x - c * y
        cx = cx - c * matrix[j]
    end
    if norm(x) > 2 * epsilon
        c = 1 / norm(x)
        x = c * x
        cx = c * cx
        x[findall(el -> abs(el) < epsilon, x)] .= 0
        cx[findall(el -> abs(el) < epsilon, cx)] .= 0
        # x[map(abs, x) < eps] .= 0
        # cx[map(abs, cx) < eps] .= 0
        c = sign(x[findall(el -> el != 0, x)[1, 1]])
        x = c * x
        cx = c * cx
        push!(final, x)
        push!(matrix, cx)
    end
end

# final = transpose(vcat(final...)) if length(final) > 0 else zeros((0, dim)) # ?
# matrix = transpose(vcat(matrix...)) if length(matrix) > 0 else zeros((0, len(original))) # ?
