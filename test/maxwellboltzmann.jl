using Test
using Distributions

@testset "MaxwellBoltzmann" begin
    d = MaxwellBoltzmann()
    @test mean(d) = 2 * sqrt(2 / π)
    @test mode(d) = sqrt(2)
end
