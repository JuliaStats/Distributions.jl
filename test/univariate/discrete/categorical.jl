using Distributions
using Test
using StableRNGs

@testset "Categorical" begin

for p in Any[
    [0.5, 0.5],
    [0.5f0, 0.5f0],
    [1//2, 1//2],
    [0.1, 0.3, 0.2, 0.4],
    [0.15, 0.25, 0.6] ]

    d = Categorical(p)
    k = length(p)
    println("    testing $d as Categorical")

    @test isa(d, Categorical)
    @test probs(d) == p
    @test minimum(d) == 1
    @test maximum(d) == k
    @test extrema(d) == (1, k)
    @test ncategories(d) == k
    @test d == d
    @test d ≈ d

    c = 0.0
    for i = 1:k
        c += p[i]
        @test @inferred(pdf(d, i)) == p[i]
        @test @inferred(pdf(d, float(i))) == p[i]
        @test @inferred(logpdf(d, i)) === log(p[i])
        @test @inferred(logpdf(d, float(i))) === log(p[i])
        @test @inferred(cdf(d, i))  ≈ c
        @test @inferred(cdf(d, i + 0.5)) ≈ c
        @test @inferred(ccdf(d, i)) ≈ 1 - c
        @test @inferred(ccdf(d, i + 0.5)) ≈ 1 - c
        @test @inferred(logcdf(d, i)) ≈ log(c)
        @test @inferred(logcdf(d, i + 0.5)) ≈ log(c)
        @test @inferred(logccdf(d, i)) ≈ log1p(-c)
        @test @inferred(logccdf(d, i + 0.5)) ≈ log1p(-c)
    end

    @test pdf(d, 0) == 0
    @test pdf(d, k+1) == 0
    @test logpdf(d, 0) == -Inf
    @test logpdf(d, k+1) == -Inf
    @test iszero(cdf(d, -Inf))
    @test iszero(cdf(d, 0))
    @test isone(cdf(d, k+1))
    @test isone(cdf(d, Inf))
    @test isnan(cdf(d, NaN))
    @test isone(ccdf(d, -Inf))
    @test isone(ccdf(d, 0))
    @test iszero(ccdf(d, k+1))
    @test iszero(ccdf(d, Inf))
    @test isnan(ccdf(d, NaN))

    @test Base.Fix1(pdf, d).(support(d)) == p
    @test Base.Fix1(pdf, d).(1:k) == p

    @test cf(d, 0) ≈ 1.0
    @test cf(d, 1) ≈ p' * cis.(1:length(p))

    @test mgf(d, 0) ≈ 1.0
    @test mgf(d, 1) ≈ p' * exp.(1:length(p))

    # The test utilities are currently only able to handle Float64s
    if partype(d) === Float64
        test_distr(d, 10^6)
    end
end

d = Categorical(4)
println("    testing $d as Categorical")
@test minimum(d) == 1
@test maximum(d) == 4
@test extrema(d) == (1, 4)
@test probs(d) == [0.25, 0.25, 0.25, 0.25]

p = ones(10^6) * 1.0e-6
@test Distributions.isprobvec(p)

@test convert(Categorical{Float64,Vector{Float64}}, d) === d
for x in (d, probs(d))
    d32 = convert(Categorical{Float32,Vector{Float32}}, d)
    @test d32 isa Categorical{Float32,Vector{Float32}}
    @test probs(d32) == map(Float32, probs(d))
end

@testset "test args... constructor" begin
    @test Categorical(0.3, 0.7) == Categorical([0.3, 0.7])
end

@testset "reproducibility across julia versions" begin
    d = Categorical([0.1, 0.2, 0.7])
    rng = StableRNGs.StableRNG(600)
    @test rand(rng, d, 10) == [3, 1, 1, 2, 3, 2, 3, 3, 2, 3]
end

@testset "comparisons" begin
    d1 = Categorical([0.4, 0.6])
    d2 = Categorical([0.6, 0.4])
    d3 = Categorical([0.2, 0.7, 0.1])

    # Same distribution
    for d in (d1, d2, d3)
        @test d == d
        @test d ≈ d
    end

    # Same support, different probabilities
    @test d2 != d1
    @test !isapprox(d2, d1)
    @test d2 ≈ d1 atol=0.4

    # Different support
    @test d3 != d1
    @test !isapprox(d3, d1)

    # issue #1675
    @test Categorical([0.5, 0.5]) ≈ Categorical([0.5, 0.5])
    @test Categorical([0.5, 0.5]) == Categorical([0.5f0, 0.5f0])
    @test Categorical([0.5, 0.5]) ≈ Categorical([0.5f0, 0.5f0])
end

@testset "issue #832" begin
    priorities = collect(Float64, 1:1000)
    priorities[1:50] .= 1e8

    at = Distributions.AliasTable(priorities)
    iat = rand(at, 16)

    # failure rate of a single sample is sum(51:1000)/50e8 = 9.9845e-5
    # failure rate of 4 out of 16 samples is 1-cdf(Binomial(16, 9.9845e-5), 3) = 1.8074430840897548e-13
    # this test should randomly fail with a probability of 1.8074430840897548e-13
    @test count(==(1e8), priorities[iat]) >= 13
end

end
