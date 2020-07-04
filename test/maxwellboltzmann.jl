using Test, Distributions

@testset "MaxwellBoltzmann" begin
    d1 = MaxwellBoltzmann()
    @test mean(d1) ≈ 2 * sqrt(2 / π)
    @test mode(d1) ≈ sqrt(2)
    d2 = MaxwellBoltzmann(2.0)
    @test pdf(d2, 1.0) ≈ sqrt(2 / π) * exp(-1.0 / (2 * d2.a^2)) / d2.a^3
    d3 = MaxwellBoltzmann(298.15, 6.635914e-26) # mass of Ar-40 in kg at room temp
    @test pdf(d3, 500.0) ≈ sqrt(2 / π) * 500.0^2 * exp(-500.0^2 / (2 * d3.a^2)) / d3.a^3
    @test dof(d1) == 3
    @test params(d1) == (1.0,)
    @test params(d2) == (2.0,)
end
