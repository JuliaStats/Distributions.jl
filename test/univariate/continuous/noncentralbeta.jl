using Distributions
using Test
using HypergeometricFunctions


let 
    α = 2.0; β = 3.0; λ = 1.0
    d = NoncentralBeta(α, β, λ)
    @test isapprox(mean(d), 0.4466, atol=1e-4)
    @test isapprox(var(d), 0.0416, atol=1e-4)
end