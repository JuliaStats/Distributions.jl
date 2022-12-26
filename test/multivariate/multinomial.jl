# Tests for Multinomial

using Distributions, Random, StaticArrays
using Test


p = [0.2, 0.5, 0.3]
nt = 10
d = Multinomial(nt, p)
rng = MersenneTwister(123)

@testset "Testing Multinomial with $key" for (key, func) in
    Dict("rand(...)" => [rand, rand],
         "rand(rng, ...)" => [dist -> rand(rng, dist), (dist, n) -> rand(rng, dist, n)])

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
@test typeof(d) == Multinomial{Float64, Vector{Float64}}
@test typeof(Multinomial(nt, Vector{Float32}(p))) == Multinomial{Float32, Vector{Float32}}
@test typeof(convert(Multinomial{Float32}, d)) == Multinomial{Float32, Vector{Float32}}
@test typeof(convert(Multinomial{Float32, Vector{Float32}}, d)) == Multinomial{Float32, Vector{Float32}}
@test typeof(convert(Multinomial{Float32}, params(d)...)) == Multinomial{Float32, Vector{Float32}}
@test typeof(convert(Multinomial{Float32, Vector{Float32}}, params(d)...)) == Multinomial{Float32, Vector{Float32}}
@test convert(Multinomial{Float64}, d) === d
@test convert(Multinomial{Float64, Vector{Float64}}, d) === d

# random sampling

x = func[1](d)
@test isa(x, Vector{Int})
@test sum(x) == nt
@test insupport(d, x)
@test size(x) == size(d)
@test length(x) == length(d)
@test d == typeof(d)(params(d)...)
@test d == deepcopy(d)

x = func[2](d, 100)
@test isa(x, Matrix{Int})
@test all(sum(x, dims=1) .== nt)
@test all(insupport(d, x))

x = func[1](sampler(d))
@test isa(x, Vector{Int})
@test sum(x) == nt
@test insupport(d, x)

# logpdf

x1 = [1, 6, 3]

@test isapprox(pdf(d, x1), 0.070875, atol=1.0e-8)
@test logpdf(d, x1) ≈ log(pdf(d, x1))

x = func[2](d, 100)
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
x = func[2](d0, n0)
w = func[1](n0)

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

x = func[2](d0, 10^5)
@test size(x) == (length(d0), 10^5)
@test all(sum(x, dims=1) .== nt)

r = fit(Multinomial, x)
@test r.n == nt
@test length(r) == length(p)
@test isapprox(probs(r), p, atol=0.02)
r = fit(Multinomial{Float64}, x)
@test r.n == nt
@test length(r) == length(p)
@test isapprox(probs(r), p, atol=0.02)

r = fit_mle(Multinomial, x, fill(2.0, size(x,2)))
@test r.n == nt
@test length(r) == length(p)
@test isapprox(probs(r), p, atol=0.02)

# behavior for n = 0
d0 = Multinomial(0, p)
@test func[1](d0) == [0, 0, 0]
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

# Abstract vector p

@test typeof(Multinomial(nt, SVector{length(p), Float64}(p))) == Multinomial{Float64, SVector{3, Float64}}

end

@testset "Testing Multinomial with $key" for (key, func) in
    Dict("rand!(...)" => (dist, X) -> rand!(dist, X),
         "rand!(rng, ...)" => (dist, X) -> rand!(rng, dist, X))
    # random sampling
    X = Matrix{Int}(undef, length(p), 100)
    x = func(d, X)
    @test x ≡ X
    @test isa(x, Matrix{Int})
    @test all(sum(x, dims=1) .== nt)
    @test all(insupport(d, x))
    pv = pdf(d, x)
    lp = logpdf(d, x)
    for i in 1 : size(x, 2)
        @test pv[i] ≈ pdf(d, x[:,i])
        @test lp[i] ≈ logpdf(d, x[:,i])
    end
end

@testset "Testing Multinomial with $key" for (key, func) in
    Dict("rand!(..., true)" => (dist, X) -> rand!(dist, X, true),
         "rand!(rng, ..., true)" => (dist, X) -> rand!(rng, dist, X, true))
    # random sampling
    X = Vector{Vector{Int}}(undef, 100)
    x = func(d, X)
    @test x ≡ X
    @test all(sum.(x) .== nt)
    @test all(insupport(d, a) for a in x)
end

@testset "Testing Multinomial with $key" for (key, func) in
    Dict("rand!(..., false)" => (dist, X) -> rand!(dist, X, false),
         "rand!(rng, ..., false)" => (dist, X) -> rand!(rng, dist, X, false))
    # random sampling
    X = [Vector{Int}(undef, length(p)) for _ in Base.OneTo(100)]
    x1 = X[1]
    x = func(d, X)
    @test x1 ≡ X[1]
    @test all(sum.(x) .== nt)
    @test all(insupport(d, a) for a in x)
end

repeats = 10
m = Vector{Vector{partype(d)}}(undef, repeats)
rand!(d, m)
@test isassigned(m, 1)
m1=m[1]
rand!(d, m)
@test m1 ≡ m[1]
rand!(d, m, true)
@test m1 ≢ m[1]
m1 = m[1]
rand!(d, m, false)
@test m1 ≡ m[1]

p = [0.2, 0.4, 0.3, 0.1]
nt = 10
d = Multinomial(nt, p)
@test_throws DimensionMismatch rand!(d, m, false)
@test_nowarn rand!(d, m)

p_v = [0.1, 0.4, 0.3, 0.8]
@test_throws DomainError Multinomial(10, p_v)
@test_throws DomainError Multinomial(10, p_v; check_args=true)
Multinomial(10, p_v; check_args=false) # should not warn
