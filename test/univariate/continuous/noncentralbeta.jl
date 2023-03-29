using Distributions
using Test


@testset "NoncentralBeta" begin 
    α = 11.0 ; β = 8.0; λ = 5.0
    d = NoncentralBeta(α, β, λ)
    @test isapprox(mean(d), 0.6259, atol=1e-4)
    @test isapprox(var(d), 0.0111, atol=1e-4)
    λ = 0.0
    d = NoncentralBeta(α, β, λ)
    d1 = Beta(α, β)
    @test isapprox(mean(d), mean(d1))
    @test isapprox(var(d), var(d1))
end
