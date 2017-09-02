using Distributions, Base.Test

@testset "EmpiricalUnivariateDistribution" begin
    n = 100
    r = MersenneTwister(123)
    @testset "$d data" for (d, x) in (("discrete"  , rand( r, 1:10, n)),
                                 ("continuous", randn(r,       n)))
        X = EmpiricalUnivariateDistribution(x)

        @testset "test function: $f" for f in (mean, var, std, skewness, kurtosis, median, entropy)
            @test f(X) ≈ f(x)
        end

        ecdfx = StatsBase.ecdf(x)
        @testset "cdf" for t in linspace(-10, 10, 100)
            @test cdf(X, t) == ecdfx(t)
            @test cdf(X, t) == mean(x -> x <= t, x)
        end

        @testset "quantile" for q in linspace(0, 1, 100)
            @test quantile(X, q) == quantile(x, q)
        end

        @testset "pdf" begin
            @test sum(t -> pdf(X, t), unique(x)) ≈ 1
        end
    end
end
