# Tests for Multinomial

using Distributions
using Base.Test

p = [0.2, 0.5, 0.3]
nt = 10
d = Multinomial(nt, p)

# Basics

@test length(d) == 3
@test d.n == nt
@test_approx_eq mean(d) [2., 5., 3.]
@test_approx_eq var(d) [1.6, 2.5, 2.1]
@test_approx_eq cov(d) [1.6 -1.0 -0.6; -1.0 2.5 -1.5; -0.6 -1.5 2.1]

@test insupport(d, [1, 6, 3])
@test !insupport(d, [2, 6, 3])
@test partype(d) == Float64
@test partype(Multinomial(nt, Vector{Float32}(p))) == Float32

# Conversion
@test typeof(d) == Multinomial{Float64}
@test typeof(Multinomial(nt, Vector{Float32}(p))) == Multinomial{Float32}
@test typeof(convert(Multinomial{Float32}, d)) == Multinomial{Float32}
@test typeof(convert(Multinomial{Float32}, params(d)...)) == Multinomial{Float32}

# random sampling

x = rand(d)
@test isa(x, Vector{Int})
@test sum(x) == nt
@test insupport(d, x)
@test size(x) == size(d)
@test length(x) == length(d)
@test d == typeof(d)(params(d)...)

x = rand(d, 100)
@test isa(x, Matrix{Int})
@test all(sum(x, 1) .== nt)
@test all(insupport(d, x))

x = rand(sampler(d))
@test isa(x, Vector{Int})
@test sum(x) == nt
@test insupport(d, x)

# logpdf

x1 = [1, 6, 3]

@test_approx_eq_eps pdf(d, x1) 0.070875 1.0e-8
@test_approx_eq logpdf(d, x1) log(pdf(d, x1))

x = rand(d, 100)
pv = pdf(d, x)
lp = logpdf(d, x)
for i in 1 : size(x, 2)
	@test_approx_eq pv[i] pdf(d, x[:,i])
	@test_approx_eq lp[i] logpdf(d, x[:,i])
end

# test type stability of logpdf
@test typeof(logpdf(convert(Multinomial{Float32}, d), x1)) == Float32

# suffstats

d0 = d
n0 = 100
x = rand(d0, n0)
w = rand(n0)

ss = suffstats(Multinomial, x)
@test isa(ss, Distributions.MultinomialStats)
@test ss.n == nt
@test ss.scnts == vec(sum(Float64[x[i,j] for i = 1:size(x,1), j = 1:size(x,2)], 2))
@test ss.tw == n0

ss = suffstats(Multinomial, x, w)
@test isa(ss, Distributions.MultinomialStats)
@test ss.n == nt
@test_approx_eq ss.scnts Float64[x[i,j] for i = 1:size(x,1), j = 1:size(x,2)] * w
@test_approx_eq ss.tw sum(w)

# fit

x = rand(d0, 10^5)
@test size(x) == (length(d0), 10^5)
@test all(sum(x, 1) .== nt)

r = fit(Multinomial, x)
@test r.n == nt
@test length(r) == length(p)
@test_approx_eq_eps probs(r) p 0.02

r = fit_mle(Multinomial, x, fill(2.0, size(x,2)))
@test r.n == nt
@test length(r) == length(p)
@test_approx_eq_eps probs(r) p 0.02
