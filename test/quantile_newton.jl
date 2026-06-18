using Distributions
using Test


d = Normal()
@test Distributions.quantile_newton(d, 0.5) == quantile(d, 0.5)
@test Distributions.cquantile_newton(d, 0.5) == cquantile(d, 0.5)

# issues #1571 and #2061
@testset "InverseGaussian quantile convergence" begin
    cases = (
        (InverseGaussian(1.0, 0.25), 0.999996),
        (InverseGaussian(2.8853900817779268), 0.9999996485182184),
    )

    @testset "p=$p" for (d, p) in cases
        x = quantile(d, p)
        @test isfinite(x)
        @test isapprox(cdf(d, x), p; atol=1e-10)
    end
end
