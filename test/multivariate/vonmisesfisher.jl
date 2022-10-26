# Tests for Von-Mises Fisher distribution

using Distributions, Random
using LinearAlgebra, Test

using SpecialFunctions

logvmfCp(p::Int, κ::Real) = (p / 2 - 1) * log(κ) - log(besselix(p / 2 - 1, κ)) - κ - p / 2 * log(2π)

safe_logvmfpdf(μ::Vector, κ::Real, x::Vector) = logvmfCp(length(μ), κ) + κ * dot(μ, x)

function gen_vmf_tdata(n::Int, p::Int,
                       rng::Union{AbstractRNG, Missing} = missing)
    if ismissing(rng)
        X = randn(p, n)
    else
        X = randn(rng, p, n)
    end
    for i = 1:n
        X[:,i] = X[:,i] ./ norm(X[:,i])
    end
    return X
end

function test_vmf_rot(p::Int, rng::Union{AbstractRNG, Missing} = missing)
    if ismissing(rng)
        μ = randn(p)
        x = randn(p)
    else
        μ = randn(rng, p)
        x = randn(rng, p)
    end
    κ = norm(μ)
    μ = μ ./ κ

    s = Distributions.VonMisesFisherSampler(μ, κ)
    v = μ - vcat(1, zeros(p-1))
    H = I - 2*v*v'/(v'*v)

    @test Distributions._vmf_rot!(s.v, copy(x)) ≈ (H*x)

end



function test_genw3(κ::Real, ns::Int, rng::Union{AbstractRNG, Missing} = missing)
    p = 3

    if ismissing(rng)
        μ = randn(p)
    else
        μ = randn(rng, p)
    end
    μ = μ ./ norm(μ)

    s = Distributions.VonMisesFisherSampler(μ, float(κ))

    genw3_res = [Distributions._vmf_genw3(rng, s.p, s.b, s.x0, s.c, s.κ) for _ in 1:ns]
    genwp_res = [Distributions._vmf_genwp(rng, s.p, s.b, s.x0, s.c, s.κ) for _ in 1:ns]

    @test isapprox(mean(genw3_res), mean(genwp_res), atol=0.01)
    @test isapprox(std(genw3_res), std(genwp_res), atol=0.01/κ)

    # test mean and stdev against analytical formulas
    coth_κ = coth(κ)
    mean_w = coth_κ - 1/κ
    var_w = 1 - coth_κ^2 + 1/κ^2

    @test isapprox(mean(genw3_res), mean_w, atol=0.01)
    @test isapprox(std(genw3_res), sqrt(var_w), atol=0.01/κ)
end


function test_vonmisesfisher(p::Int, κ::Real, n::Int, ns::Int,
                             rng::Union{AbstractRNG, Missing} = missing)
    if ismissing(rng)
        μ = randn(p)
    else
        μ = randn(rng, p)
    end
    μ = μ ./ norm(μ)

    d = VonMisesFisher(μ, κ)
    @test length(d) == p
    @test meandir(d) == μ
    @test concentration(d) == κ
    @test partype(d) == Float64

    # conversions
    @test convert(VonMisesFisher{partype(d)}, d) === d
    for d32 in (convert(VonMisesFisher{Float32}, d), convert(VonMisesFisher{Float32}, d.μ, d.κ, d.logCκ))
        @test d32 isa VonMisesFisher{Float32}
        @test params(d32) == (map(Float32, μ), Float32(κ))
    end

    θ = κ * μ
    d2 = VonMisesFisher(θ)
    @test length(d2) == p
    @test meandir(d2) ≈ μ
    @test concentration(d2) ≈ κ

    @test d.logCκ ≈ logvmfCp(p, κ) atol=1.0e-12

    X = gen_vmf_tdata(n, p, rng)
    lp0 = zeros(n)
    for i = 1:n
        xi = X[:,i]
        lp0[i] = safe_logvmfpdf(μ, κ, xi)
        @test logpdf(d, xi) ≈ lp0[i]
    end
    @test logpdf(d, X) ≈ lp0

    # sampling
    if ismissing(rng)
        x = rand(d)
    else
        x = rand(rng, d)
    end
    @test norm(x) ≈ 1.0
    @test insupport(d, x)

    if ismissing(rng)
        X = rand(d, n)
    else
        X = rand(rng, d, n)
    end
    for i = 1:n
        @test norm(X[:,i]) ≈ 1.0
        @test insupport(d, X[:,i])
    end

    # MLE
    if ismissing(rng)
        X = rand(d, ns)
    else
        X = rand(rng, d, ns)
    end
    d_est = fit_mle(VonMisesFisher, X)
    @test isa(d_est, VonMisesFisher)
    @test isapprox(d_est.μ, μ, atol=0.01)
    @test isapprox(d_est.κ, κ, atol=κ * 0.01)
    d_est = fit(VonMisesFisher{Float64}, X)
    @test isa(d_est, VonMisesFisher)
    @test isapprox(d_est.μ, μ, atol=0.01)
    @test isapprox(d_est.κ, κ, atol=κ * 0.01)
end


## General testing

@testset "Testing VonMisesFisher argument promotions" begin
    d = VonMisesFisher(Int[1, 0], Float32(5))
    @test d isa VonMisesFisher{Float32}
    d = VonMisesFisher(Int[1, 0], Float64(5))
    @test d isa VonMisesFisher{Float64}
    d = VonMisesFisher(Float64[1, 0], 5)
    @test d isa VonMisesFisher{Float64}
    d = VonMisesFisher(Float64[1, 0], Float32(5))
    @test d isa VonMisesFisher{Float64}
end

n = 1000
ns = 10^6
@testset "Testing VonMisesFisher with $key" for (key, rng) in
    Dict("rand(...)" => missing,
         "rand(rng, ...)" => MersenneTwister(123))

    @testset "Testing VonMisesFisher with $key at ($p, $κ)" for (p, κ) in [(2, 1.0),
                                                                           (2, 5.0),
                                                                           (3, 1.0),
                                                                           (3, 5.0),
                                                                           (5, 2.0),
                                                                           (2, 2),
                                                                           (2, 1000)] # test with large κ
        test_vonmisesfisher(p, κ, n, ns, rng)
        test_vmf_rot(p, rng)
    end

    if !ismissing(rng)
        @testset "Testing genw with $key at (3, $κ)" for κ in [0.1, 0.5, 1.0, 2.0, 5.0]
            test_genw3(κ, ns, rng)
        end
    end
end
