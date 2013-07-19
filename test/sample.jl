# Test sample functions

using Distributions
using Base.Test

function est_p(x, K)
	h = zeros(Int, K)
	for xi in x
		h[xi] += 1
	end
	p = h / length(x)
end

#### sample with replacement

n = 10^5
x = sample([10,20,30], n)
@test isa(x, Vector{Int})
@test length(x) == n

h = [sum(x .== 10), sum(x .== 20), sum(x .== 30)]
@test sum(h) == n
ph = h / n
p0 = fill(1/3, 3)
@test_approx_eq_eps ph p0 0.02


#### sample without replacement

# case: K == 2

n = 10^5
x = zeros(Int, 2, n)
for i = 1:n
	v = sample(11:15, 2; replace=false)
	@assert v[1] != v[2]
	x[:,i] = v
end
@test min(x) == 11
@test max(x) == 15

x[:] -= 10  # brings x to 1:5
p0 = fill(1/5, 5)
@test_approx_eq_eps est_p(x, 5) p0 0.02

# case: K == 4 with moderate a (using Fisher-Yates)

n = 10^5
x = zeros(Int, 4, n)
for i = 1 : n
	v = sample(11:20, 4; replace=false)
	sv = sort(v)
	@assert sv[1] < sv[2] < sv[3] < sv[4]
	x[:,i] = v
end
@test min(x) == 11
@test max(x) == 20

x[:] -= 10
p0 = fill(0.1, 10)
@test_approx_eq_eps est_p(x, 10) p0 0.01

# case: K == 4 with very large a (using self-avoid)

n = 10^4
x = zeros(Int, 4, n)
a = 10^7 + 1
b = 2 * 10^7
for i = 1 : n
	v = sample(a:b, 4; replace=false)
	sv = sort(v)
	@assert sv[1] < sv[2] < sv[3] < sv[4]
	x[:,i] = v
end
@test min(x) >= a
@test max(x) <= b


#### weighted sampling

w = [2., 5., 3.]
n = 10^5
x = sample([10,20,30], w, n)

h = [sum(x .== 10), sum(x .== 20), sum(x .== 30)]
@test sum(h) == n
p0 = w / sum(w)
ph = h / n
@test_approx_eq_eps ph p0 0.02

