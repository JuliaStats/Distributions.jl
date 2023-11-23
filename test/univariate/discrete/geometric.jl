using Distributions
using Test
using FiniteDifferences

@testset "Geometric mgf and k vs k-1 parametrization #1604" begin
    d = Geometric(0.2)
    @test mgf(d, 0) ≈ 1
    @test cf(d, 0) ≈ 1

    fdm1 = central_fdm(5, 1)
    @test fdm1(Base.Fix1(mgf, d), 0) ≈ mean(d)
    @test fdm1(Base.Fix1(cf, d), 0) ≈ mean(d) * im

    fdm2 = central_fdm(5, 2)
    m2 = var(d) + mean(d)^2
    @test fdm2(Base.Fix1(mgf, d), 0) ≈ m2
    @test fdm2(Base.Fix1(cf, d), 0) ≈ -m2
    test_cgf(Geometric(0.1), (1f-1, -1e6))
    test_cgf(Geometric(0.5), (1f-1, -1e6))
end
