using Distributions
if VERSION >= v"0.7.0-DEV"
    using Test
else
    using Base.Test
end

@testset "non-central T-Distributions" begin
    ν = 50
    λ = -5

    d = NoncentralT(ν, λ)
    @test_throws MethodError NoncentralT(ν, complex(λ))
    @test d == typeof(d)(params(d)...)
    @test mean(d) ≈ -5.0766 atol = 1e-5

    λ = 5
    d = NoncentralT(ν, λ)
    @test mean(d) ≈ 5.0766 atol=1e-5

    λ = 0
    d = NoncentralT(ν, λ)
    @test var(d) == var(TDist(ν))
end
