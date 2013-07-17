# Tests for Multinomial

using Distributions
using Base.Test

p = [0.2, 0.5, 0.3]
nt = 10
d = Multinomial(nt, p)

# Basics

@test dim(d) == 3
@test d.n == nt
@test_approx_eq mean(d) [2., 5., 3.]
@test_approx_eq var(d) [1.6, 2.5, 2.1]
@test_approx_eq cov(d) [1.6 -1.0 -0.6; -1.0 2.5 -1.5; -0.6 -1.5 2.1]

@test insupport(d, [1, 6, 3])
@test !insupport(d, [2, 6, 3])

# random sampling

x = rand(d)
@test isa(x, Vector{Int})
@test sum(x) == nt
@test insupport(d, x)

x = rand(d, 100)
@test isa(x, Matrix{Int})
@test all(sum(x, 1) .== nt)
@test insupport(d, x)

# logpdf

x1 = [1, 6, 3]

@test_approx_eq_eps pdf(d, x1) 0.070875 1.0e-8
@test_approx_eq logpdf(d, x1) log(pdf(d, x1))

x = rand(d, 100)
p = pdf(d, x)
lp = logpdf(d, x)
for i in 1 : size(x, 2)
	@test_approx_eq p[i] pdf(d, x[:,i])
	@test_approx_eq lp[i] logpdf(d, x[:,i])
end
