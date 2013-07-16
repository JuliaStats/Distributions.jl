# Tests for Dirichlet distribution

using Distributions
using Base.Test

d = Dirichlet(3, 2.0)

@test dim(d) == 3
@test d.alpha == [2.0, 2.0, 2.0]
@test d.alpha0 == 6.0

@test_approx_eq mean(d) fill(1.0/3, 3)
@test_approx_eq cov(d) [8 -4 -4; -4 8 -4; -4 -4 8] / (36 * 7)

@test_approx_eq pdf(d, [0.2, 0.3, 0.5]) 3.6
@test_approx_eq pdf(d, [0.4, 0.5, 0.1]) 2.4
@test_approx_eq logpdf(d, [0.2, 0.3, 0.5]) log(3.6)
@test_approx_eq logpdf(d, [0.4, 0.5, 0.1]) log(2.4)

x = rand(d, 100)
p = pdf(d, x)
lp = logpdf(d, x)
for i in 1 : size(x, 2)
	@test_approx_eq p[i] pdf(d, x[:,i])
	@test_approx_eq lp[i] logpdf(d, x[:,i])
end


v = [2.0, 1.0, 3.0]
d = Dirichlet(v)

@test dim(d) == length(v)
@test d.alpha == v
@test d.alpha0 == sum(v)

@test_approx_eq mean(d) v / sum(v)
@test_approx_eq cov(d) [8 -2 -6; -2 5 -3; -6 -3 9] / (36 * 7)

@test_approx_eq pdf(d, [0.2, 0.3, 0.5]) 3.0
@test_approx_eq pdf(d, [0.4, 0.5, 0.1]) 0.24
@test_approx_eq logpdf(d, [0.2, 0.3, 0.5]) log(3.0)
@test_approx_eq logpdf(d, [0.4, 0.5, 0.1]) log(0.24)

x = rand(d, 100)
p = pdf(d, x)
lp = logpdf(d, x)
for i in 1 : size(x, 2)
	@test_approx_eq p[i] pdf(d, x[:,i])
	@test_approx_eq lp[i] logpdf(d, x[:,i])
end

