@testset "NormalInverseGaussian random repeatable and basic metrics" begin
    rng = Random.MersenneTwister(42)
    rng2 = copy(rng)
    µ = 0.0
    α = 1.0
    β = 0.5
    δ = 3.0
    g = sqrt(α^2 - β^2)
    d = NormalInverseGaussian(μ, α, β, δ)
    v1 = rand(rng, d)
    v2 = rand(rng, d)
    v3 = rand(rng2, d)
    @test v1 ≈ v3
    @test v1 ≉ v2

    @test mean(d) ≈ µ + β * δ / g
    @test var(d) ≈ δ * α^2 / g^3
    @test skewness(d) ≈ 3β/(α*sqrt(δ*g))
end
