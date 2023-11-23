using Distributions
using Test


@testset "non-central T-Distributions" begin
    ν = 50
    λ = -5

    d = NoncentralT(ν, λ)
    @test_throws MethodError NoncentralT(ν, complex(λ))
    @test d == typeof(d)(params(d)...)
    @test d == deepcopy(d)
    @test mean(d) ≈ -5.0766 atol = 1e-5

    λ = 5
    d = NoncentralT(ν, λ)
    @test mean(d) ≈ 5.0766 atol=1e-5

    λ = 0
    d = NoncentralT(ν, λ)
    @test var(d) == var(TDist(ν))
end
