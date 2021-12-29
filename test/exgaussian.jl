using Distributions
using Test

@testset "Exgaussian" begin
    μ = 400
    σ = 40
    τ = 200

    d = Exgaussian(μ,σ,τ)
    @test d == typeof(d)(params(d)...)
    @test d == deepcopy(d)

    @test mean(d) == μ + τ
    @test var(d) == σ^2 + τ^2

    t = 500
    @test pdf(d,t) ≈ 0.0030607  atol = 1e-5
    @test cdf(d,t) ≈ 0.38164  atol = 1e-5
end
 