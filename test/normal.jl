using Test, Distributions
@testset "Normal" begin
    @test isa(convert(Normal{Float64}, Float16(0), Float16(1)),
              Normal{Float64})
    @test Inf === logpdf(Normal(0, 0), 0)
    @test -Inf === logpdf(Normal(), Inf)
    @test iszero(logcdf(Normal(0, 0), 0))
    @test iszero(logcdf(Normal(), Inf))
    @test -Inf === logccdf(Normal(0, 0), 0)
    @test iszero(logccdf(Normal(eps(), 0), 0))
    @test -Inf === quantile(Normal(), 0)
    @test iszero(quantile(Normal(), 0.5))
    @test Inf === quantile(Normal(), 1)
    @test -Inf === quantile(Normal(0, 0), 0)
    @test iszero(quantile(Normal(0, 0), 0.75))
    @test Inf === quantile(Normal(0, 0), 1)
    @test -Inf === quantile(Normal(0.25, 0), 0)
    @test 0.25 == quantile(Normal(0.25, 0), 0.95)
    @test Inf === quantile(Normal(0.25, 0), 1)
    @test Inf === cquantile(Normal(), 0)
    @test iszero(cquantile(Normal(), 0.5))
    @test -Inf === cquantile(Normal(), 1)
    @test Inf === cquantile(Normal(0, 0), 0)
    @test iszero(cquantile(Normal(0, 0), 0.75))
    @test -Inf === cquantile(Normal(0, 0), 1)
    @test Inf === cquantile(Normal(0.25, 0), 0)
    @test 0.25 == cquantile(Normal(0.25, 0), 0.95)
    @test -Inf === cquantile(Normal(0.25, 0), 1)
    @test -Inf === invlogcdf(Normal(), -Inf)
    @test isnan(invlogcdf(Normal(), NaN))
    @test Inf === invlogccdf(Normal(), -Inf)
    @test isnan(invlogccdf(Normal(), NaN))
    # test for #996 being fixed
    let d = Normal(0, 1), x = 1.0, ∂x = 2.0
        @inferred cdf(d, ForwardDiff.Dual(x, ∂x)) ≈ ForwardDiff.Dual(cdf(d, x), ∂x * pdf(d, x))
    end
end
