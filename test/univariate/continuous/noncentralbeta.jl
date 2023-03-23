using Distributions
using Test


@testset "NoncentralBeta" begin 
    α = 2.0; β = 3.0; λ = 1.0
    d = NoncentralBeta(α, β, λ)
    @test isapprox(mean(d), 0.4466, atol=1e-4)
    @test isapprox(var(d), 0.0416, atol=1e-4)
    λ = 0.0
    d = NoncentralBeta(α, β, λ)
    d1 = Beta(α, β)
    @test isapprox(mean(d), mean(d1))
    @test isapprox(var(d), var(d1))
end
