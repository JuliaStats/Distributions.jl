# Tests for Von-Mises Fisher distribution

using Distributions
using LinearAlgebra, Test

using SpecialFunctions

vmfCp(p::Int, κ::Float64) = (κ ^ (p/2 - 1)) / ((2π)^(p/2) * besseli(p/2-1, κ))

safe_vmfpdf(μ::Vector, κ::Float64, x::Vector) = vmfCp(length(μ), κ) * exp(κ * dot(μ, x))

function gen_vmf_tdata(n::Int, p::Int)
    X = randn(p, n)
    for i = 1:n
        X[:,i] = X[:,i] ./ norm(X[:,i])
    end
    return X
end

function test_vonmisesfisher(p::Int, κ::Float64, n::Int, ns::Int)
    μ = randn(p)
    μ = μ ./ norm(μ)

    d = VonMisesFisher(μ, κ)
    @test length(d) == p
    @test meandir(d) == μ
    @test concentration(d) == κ
#    @test d == typeof(d)(params(d)...)
    @test partype(d) == Float64
    # println(d)

    # conversions
    @test typeof(convert(VonMisesFisher{Float32}, d)) == VonMisesFisher{Float32}
    @test typeof(convert(VonMisesFisher{Float32}, d.μ, d.κ, d.logCκ)) == VonMisesFisher{Float32}

    θ = κ * μ
    d2 = VonMisesFisher(θ)
    @test length(d2) == p
    @test meandir(d2) ≈ μ
    @test concentration(d2) ≈ κ

    @test isapprox(d.logCκ, log(vmfCp(p, κ)), atol=1.0e-12)

    X = gen_vmf_tdata(n, p)
    lp0 = zeros(n)
    for i = 1:n
        xi = X[:,i]
        lp0[i] = log(safe_vmfpdf(μ, κ, xi))
        @test logpdf(d, xi) ≈ lp0[i]
    end
    @test logpdf(d, X) ≈ lp0

    # sampling
    x = rand(d)
    @test norm(x) ≈ 1.0

    X = rand(d, n)
    for i = 1:n
        @test norm(X[:,i]) ≈ 1.0
    end

    # MLE
    X = rand(d, ns)
    d_est = fit_mle(VonMisesFisher, X)
    @test isa(d_est, VonMisesFisher)
    @test isapprox(d_est.μ, μ, atol=0.01)
    @test isapprox(d_est.κ, κ, atol=κ * 0.01)
end


## General testing

n = 1000
ns = 10^6
for (p, κ) in [(2, 1.0),
               (2, 5.0),
               (3, 1.0),
               (3, 5.0),
               (5, 2.0)]
    local p

    test_vonmisesfisher(p, κ, n, ns)
end
