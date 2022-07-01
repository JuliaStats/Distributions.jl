# Tests for Von-Mises distribution

using Distributions
using Test

function test_vonmises(μ::Float64, κ::Float64)
    d = VonMises(μ, κ)
    @test length(d) == 1
    @test mean(d) == μ
    @test median(d) == μ
    @test mode(d) == μ
    @test d == typeof(d)(params(d)...,d.I0κx)
    @test d == deepcopy(d)
    @test partype(d) == Float64
    # println(d)

    # conversions
    @test convert(VonMises{Float64}, d) === d
    d32 = convert(VonMises{Float32}, d)
    @test d32 isa VonMises{Float32}
    @test params(d32) == map(Float32, params(d))

    # Support
    @test support(d) == RealInterval(d.μ-π,d.μ+π)
    @test pdf(d, d.μ-2π) == 0.0
    @test pdf(d, d.μ+2π) == 0.0

end


## General testing

for (μ, κ) in [(2.0, 1.0),
               (2.0, 5.0),
               (3.0, 1.0),
               (3.0, 5.0),
               (5.0, 2.0)]

    test_vonmises(μ, κ)
end

@testset "VonMises with large kappa" begin
    d = VonMises(1.1, 1000)
    @test var(d) ≈ 0.0005001251251957198
    @test entropy(d) ≈ -2.034688918525470
    @test cf(d, 2.5) ≈ -0.921417 + 0.38047im atol=1e-6
    @test pdf(d, 0.5) ≈ 1.758235814051e-75
    @test logpdf(d, 0.5) ≈ -172.1295710466005
    @test cdf(d, 1.0) ≈ 0.000787319 atol=1e-9
end
