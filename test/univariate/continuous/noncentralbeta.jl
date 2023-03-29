using Distributions
using Test


@testset "NoncentralBeta" begin 
    α = 11.54 ; β = 8.89; λ = 5/4
    d = NoncentralBeta(α, β, λ)
    @test isapprox(mean(d), 0.5772, atol=1e-4)
    @test isapprox(var(d), 0.0113, atol=1e-4)
    λ = 0.0
    d = NoncentralBeta(α, β, λ)
    d1 = Beta(α, β)
    @test isapprox(mean(d), mean(d1))
    @test isapprox(var(d), var(d1))
end
