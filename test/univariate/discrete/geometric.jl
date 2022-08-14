using Distributions
using Test
using ForwardDiff

@testset "Geometric mgf and k vs k-1 parametrization #1604" begin
    d = Geometric(0.2)
    @test mgf(d, 0) ≈ 1
    @test ForwardDiff.derivative(Base.Fix1(mgf, d), 0) ≈ mean(d)
end
