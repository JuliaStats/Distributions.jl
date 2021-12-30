using Distributions
using Test

@testset "Exgaussian" begin
    μ = 400
    σ = 40
    τ = 200

    @test_throws ArgumentError Exgaussian(100, 0, 100)
    @test_throws ArgumentError Exgaussian(100, 1, 0)

    d = Exgaussian(μ,σ,τ)
    @test d == typeof(d)(params(d)...)
    @test d == deepcopy(d)

    @test mean(d) == μ + τ
    @test var(d) == σ^2 + τ^2
    @test std(d) == √var(d)

    t = 500
    @test pdf(d,t) ≈ 0.0030607  atol = 1e-5
    @test cdf(d,t) ≈ 0.38164  atol = 1e-5

    # Check mgf
    del = 0.0000001
    @test (mgf(d,del/2) - mgf(d,-del/2))/del  ≈ μ + τ  atol = 1e-5

end
 