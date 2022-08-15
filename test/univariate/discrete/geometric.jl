using Distributions
using Test
using FiniteDifferences

@testset "Geometric mgf and k vs k-1 parametrization #1604" begin
    d = Geometric(0.2)
    @test mgf(d, 0) == 1
    @test cf(d, 0) == 1

    fdm = central_fdm(5, 1)
    @test fdm(Base.Fix1(mgf, d), 0) ≈ mean(d)
    @test fdm(Base.Fix1(cf, d), 0) ≈ mean(d) * im
end
