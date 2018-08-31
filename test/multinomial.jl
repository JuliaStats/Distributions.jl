# Tests for Multinomial

using  Distributions
using Test


p = [0.2, 0.5, 0.3]
nt = 10
d = Multinomial(nt, p)

# Basics

@test length(d) == 3
@test d.n == nt
@test mean(d) ≈ [2., 5., 3.]
@test var(d)  ≈ [1.6, 2.5, 2.1]
@test cov(d)  ≈ [1.6 -1.0 -0.6; -1.0 2.5 -1.5; -0.6 -1.5 2.1]

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
@test all(sum(x, dims=1) .== nt)
@test all(insupport(d, x))

x = rand(sampler(d))
@test isa(x, Vector{Int})
@test sum(x) == nt
@test insupport(d, x)

# logpdf

x1 = [1, 6, 3]

@test isapprox(pdf(d, x1), 0.070875, atol=1.0e-8)
@test logpdf(d, x1) ≈ log(pdf(d, x1))

x = rand(d, 100)
pv = pdf(d, x)
lp = logpdf(d, x)
for i in 1 : size(x, 2)
    @test pv[i] ≈ pdf(d, x[:,i])
    @test lp[i] ≈ logpdf(d, x[:,i])
end

# test type stability of logpdf
@test typeof(logpdf(convert(Multinomial{Float32}, d), x1)) == Float32

# test degenerate cases of logpdf
d1 = Multinomial(1, [0.5, 0.5, 0.0])
d2 = Multinomial(0, [0.5, 0.5, 0.0])
x2 = [1, 0, 0]
x3 = [0, 0, 1]
x4 = [1, 0, 1]

@test logpdf(d1, x2) ≈ log(0.5)
@test logpdf(d2, x2) == -Inf
@test logpdf(d1, x3) == -Inf
@test logpdf(d2, x3) == -Inf

# suffstats

d0 = d
n0 = 100
x = rand(d0, n0)
w = rand(n0)

ss = suffstats(Multinomial, x)
@test isa(ss, Distributions.MultinomialStats)
@test ss.n == nt
@test ss.scnts == vec(sum(Float64[x[i,j] for i = 1:size(x,1), j = 1:size(x,2)], dims=2))
@test ss.tw == n0

ss = suffstats(Multinomial, x, w)
@test isa(ss, Distributions.MultinomialStats)
@test ss.n == nt
@test ss.scnts ≈ Float64[x[i,j] for i = 1:size(x,1), j = 1:size(x,2)] * w
@test ss.tw    ≈ sum(w)

# fit

x = rand(d0, 10^5)
@test size(x) == (length(d0), 10^5)
@test all(sum(x, dims=1) .== nt)

r = fit(Multinomial, x)
@test r.n == nt
@test length(r) == length(p)
@test isapprox(probs(r), p, atol=0.02)

r = fit_mle(Multinomial, x, fill(2.0, size(x,2)))
@test r.n == nt
@test length(r) == length(p)
@test isapprox(probs(r), p, atol=0.02)

# behavior for n = 0
d0 = Multinomial(0, p)
@test rand(d0) == [0, 0, 0]
@test pdf(d0, [0, 0, 0]) == 1
@test pdf(d0, [0, 1, 0]) == 0
@test mean(d0) == [0, 0, 0]
@test var(d0) == [0, 0, 0]
@test cov(d0) == zeros(3, 3)
@test entropy(d0) == 0
@test insupport(d0, [0, 0, 0]) == true
@test insupport(d0, [0, 0, 4]) == false
@test length(d0) == 3
@test size(d0) == (3,)
