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

    c = 0.0
    for i = 1:k
        c += p[i]
        @test pdf(d, i) == p[i]
        @test @inferred(logpdf(d, i)) === log(p[i])
        @test @inferred(logpdf(d, float(i))) === log(p[i])
        @test cdf(d, i)  ≈ c
        @test ccdf(d, i) ≈ 1.0 - c
    end

    @test pdf(d, 0) == 0
    @test pdf(d, k+1) == 0
    @test logpdf(d, 0) == -Inf
    @test logpdf(d, k+1) == -Inf
    @test cdf(d, 0) == 0.0
    @test cdf(d, k+1) == 1.0
    @test ccdf(d, 0) == 1.0
    @test ccdf(d, k+1) == 0.0

    @test pdf.(d, support(d)) == p
    @test pdf.(d, 1:k) == p

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

@test typeof(convert(Categorical{Float32,Vector{Float32}}, d)) == Categorical{Float32,Vector{Float32}}
@test typeof(convert(Categorical{Float32,Vector{Float32}}, d.p)) == Categorical{Float32,Vector{Float32}}

@testset "test args... constructor" begin
    @test Categorical(0.3, 0.7) == Categorical([0.3, 0.7])
end

@testset "reproducibility across julia versions" begin
    d= Categorical([0.1, 0.2, 0.7])
    rng = StableRNGs.StableRNG(600)
    @test rand(rng, d, 10) == [2, 1, 3, 3, 2, 3, 3, 3, 3, 3]
end

end
