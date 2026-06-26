using Distributions
using Test


d = Normal()
@test Distributions.quantile_newton(d, 0.5) == quantile(d, 0.5)
@test Distributions.cquantile_newton(d, 0.5) == cquantile(d, 0.5)

# issue #1571
@testset "InverseGaussian quantile convergence (#1571)" begin
    d = InverseGaussian(1.0, 0.25)
    p = 0.999996
    x = quantile(d, p)
    @test isfinite(x)
    @test cdf(d, x) ≈ p
end

# issue #2061
@testset "InverseGaussian quantile convergence (#2061)" begin
    d = InverseGaussian(2.8853900817779268)
    p = 0.9999996485182184
    x = quantile(d, p)
    @test isfinite(x)
    @test cdf(d, x) ≈ p
end

# issue #1898: large-σ InverseGaussian, `p` far in the tail. `2 * cdf(Normal(), 5) - 1` is the
# value `erf(5 / sqrt(2))` from the original report.
@testset "InverseGaussian quantile convergence (#1898)" begin
    d = InverseGaussian(1.187997687788096, 60.467382225458564)
    p = 2 * cdf(Normal(), 5) - 1
    x = quantile(d, p)
    @test isfinite(x)
    @test cdf(d, x) ≈ p
end

# the Roots defaults are type-aware: `Float32` must converge (and preserve the element type)
# without the absolute tolerance floor that the hand-rolled loops needed to avoid an infinite loop.
@testset "Float32 quantile round-trip" begin
    d = InverseGaussian(2.0f0, 3.0f0)
    for p in (0.1f0, 0.5f0, 0.9f0, 0.999f0)
        x = quantile(d, p)
        @test x isa Float32
        @test isfinite(x)
        @test cdf(d, x) ≈ p
    end
end
