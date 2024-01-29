module Wigner

using LinearAlgebra
using Einsum

int(x) = floor(Int, x)

function su2_generators(j)
	m = [-j:j-1...]
	if length(m) == 0
		m = Array{Float64}(undef, 0)
	end
	raising = diagm(-1 => -broadcast(sqrt, j .* (j .+ 1) .- m .* (m .+ 1)))

	m = [-j+1:j...]
	if length(m) == 0
		m = Array{Float64}(undef, 0)
	end
	lowering = diagm(1 => broadcast(sqrt, j .* (j .+ 1) .- m .* (m .- 1)))

	m = [-j:j...]
	if length(m) == 0
		m = Array{Float64}(undef, 0)
	end
	return permutedims(cat(
			0.5 * (raising + lowering),  # x (usually)
			diagm(1im * m),  # z (usually)
			-0.5im * (raising - lowering),  # -y (usually)
			dims = 3), [3, 1, 2]) # originally stack(..., dim=0)
	# I hope this is always in 3 dimensions
end

function so3_generators(l)
	X = su2_generators(l)
	Q = change_basis_real_to_complex(l)
	X = cat([conj(transpose(Q)) * X[i, :, :] * Q for i in 1:3]..., dims = 3)
	X = permutedims(X, [3, 1, 2])
	@assert all(broadcast(x -> abs(imag(x)) < 1e-5, X))
	return real(X)
end

function wigner_D(l, alpha, beta, gamma)
	"""Wigner D matrix representation of :math:`SO(3)`.
	It satisfies the following properties:
	* :math:`D(\\text{identity rotation}) = \\text{identity matrix}`
	* :math:`D(R_1 \\circ R_2) = D(R_1) \\circ D(R_2)`
	* :math:`D(R^{-1}) = D(R)^{-1} = D(R)^T`
	* :math:`D(\\text{rotation around Y axis})` has some property that allows us to use FFT in `ToS2Grid`
	Parameters
	----------
	l : int
		:math:`l`
	alpha : `torch.Tensor`
		tensor of shape :math:`(...)`
		Rotation :math:`\\alpha` around Y axis, applied third.
	beta : `torch.Tensor`
		tensor of shape :math:`(...)`
		Rotation :math:`\\beta` around X axis, applied second.
	gamma : `torch.Tensor`
		tensor of shape :math:`(...)`
		Rotation :math:`\\gamma` around Y axis, applied first.
	Returns
	-------
	`torch.Tensor`
		tensor :math:`D^l(\\alpha, \\beta, \\gamma)` of shape :math:`(2l+1, 2l+1)`
	"""
	alpha = broadcast((α, β, γ) -> α, alpha, beta, gamma)
	beta = broadcast((α, β, γ) -> β, alpha, beta, gamma)
	gamma = broadcast((α, β, γ) -> γ, alpha, beta, gamma)
	sizes = size(alpha)
	if size(alpha) == ()
		alpha = [alpha]
		beta = [beta]
		gamma = [gamma]
	end
	alpha = reshape(alpha, sizes..., 1, 1) .% (2 * pi)
	beta = reshape(beta, sizes..., 1, 1) .% (2 * pi)
	gamma = reshape(gamma, sizes..., 1, 1) .% (2 * pi)
	X = so3_generators(l)
	return exp(alpha .* X[2, :, :]) * exp(beta .* X[1, :, :]) * exp(gamma .* X[2, :, :])
end

function change_basis_real_to_complex(l)
	q = zeros(ComplexF64, 2 * l + 1, 2 * l + 1)
	for m in -l:-1
		q[l+m+1, l+abs(m)+1] = sqrt(0.5)
		q[l+m+1, l-abs(m)+1] = -sqrt(0.5) * 1im
	end
	q[l+1, l+1] = 1
	for m in 1:l
		q[l+m+1, l+abs(m)+1] = (-1)^m * sqrt(0.5)
		q[l+m+1, l-abs(m)+1] = 1im * (-1)^m * sqrt(0.5)
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


	# C = einsum('ij,kl,mn,ikn->jlm', Q1, Q2, Q3, C)
	@einsum C_prod[j, l, m] := Q1[i, j] * Q2[k, l] * Q3[m, n] * C[i, k, n]

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
	if int(2 * j3) in range(int(2 * abs(j1 - j2)), int(2 * (j1 + j2)), step = 2)
		for m1 in [x / 2 for x in range(-int(2 * j1), int(2 * j1), step = 2)]
			for m2 in [x / 2 for x in range(-int(2 * j2), int(2 * j2), step = 2)]
				if abs(m1 + m2) <= j3
					mat[int(j1 + m1)+1, int(j2 + m2)+1, int(j3 + m1 + m2)+1] = _su2_clebsch_gordan_coeff((j1, m1), (j2, m2), (j3, m1 + m2))
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
		S += (-1)^(v + j2 + m2) / f(v) * f(j2 + j3 + m1 - v) * f(j1 - m1 + v) / f(j3 - j1 + j2 - v) / f(j3 + m3 - v) / f(v + j1 - j2 - m3)
	end
	C = C * S
	return C
end

end
