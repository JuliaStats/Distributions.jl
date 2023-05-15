using Test, Distributions
using Random
using StatsBase

@testset "OrderStatistic" begin
    @testset "basic" begin
        for dist in [Uniform(), Normal(), DiscreteUniform(10)], n in [1, 2, 10], i in 1:n
            d = OrderStatistic(dist, n, i)
            @test d isa OrderStatistic
            if dist isa DiscreteUnivariateDistribution
                @test d isa DiscreteUnivariateDistribution
            else
                @test d isa ContinuousUnivariateDistribution
            end
            @test d.dist === dist
            @test d.n == n
            @test d.rank == i
        end
        @test_throws ArgumentError OrderStatistic(Normal(), 0, 1)
        OrderStatistic(Normal(), 0, 1; check_args=false)
        @test_throws ArgumentError OrderStatistic(Normal(), 10, 11)
        OrderStatistic(Normal(), 10, 11; check_args=false)
        @test_throws ArgumentError OrderStatistic(Normal(), 10, 0)
        OrderStatistic(Normal(), 10, 0; check_args=false)
    end

    @testset "params" begin
        for dist in [Uniform(), Normal(), DiscreteUniform(10)], n in [1, 2, 10], i in 1:n
            d = OrderStatistic(dist, n, i)
            @test params(d) == (params(dist)..., n, i)
            @test partype(d) === partype(dist)
        end
    end

    @testset "support" begin
        n = 10
        for i in 1:10
            d1 = OrderStatistic(Uniform(), n, i)
            @test minimum(d1) == 0
            @test maximum(d1) == 1
            @test insupport(d1, 0)
            @test insupport(d1, 0.5)
            @test insupport(d1, 1)
            @test !insupport(d1, -eps())
            @test !insupport(d1, 1 + eps())
            @test !hasfinitesupport(d1)
            @test islowerbounded(d1)
            @test isupperbounded(d1)

            d2 = OrderStatistic(Normal(), n, i)
            @test minimum(d2) == -Inf
            @test maximum(d2) == Inf
            @test insupport(d2, -Inf)
            @test insupport(d2, 0)
            @test insupport(d2, Inf)
            @test !hasfinitesupport(d2)
            @test !islowerbounded(d2)
            @test !isupperbounded(d2)

            d3 = OrderStatistic(DiscreteUniform(1, 10), n, i)
            @test minimum(d3) == 1
            @test maximum(d3) == 10
            @test insupport(d3, 1)
            @test insupport(d3, 5)
            @test insupport(d3, 10)
            @test !insupport(d3, 0)
            @test !insupport(d3, 11)
            @test hasfinitesupport(d3)
            @test islowerbounded(d3)
            @test isupperbounded(d3)
        end
    end

    @testset "pdf/logpdf" begin
        @testset "continuous" begin
            # test against the exact formula computed using BigFloats
            @testset for T in (Float32, Float64)
                @testset for dist in
                             [Uniform(T(-2), T(1)), Normal(T(3), T(2)), Exponential(T(10))],
                    n in [1, 10, 100],
                    i in 1:n

                    d = OrderStatistic(dist, n, i)
                    c = factorial(big(n)) / factorial(big(i - 1)) / factorial(big(n - i))
                    # since density is concentrated around the i/n quantile, sample a point
                    # nearby it
                    x = quantile(dist, clamp(i / n + (rand() - 1//2) / 10, 0, 1))
                    p = cdf(dist, big(x))
                    pdf_exp = c * p^(i - 1) * (1 - p)^(n - i) * pdf(dist, big(x))
                    @test @inferred(T, pdf(d, x)) ≈ T(pdf_exp)
                    @test @inferred(T, logpdf(d, x)) ≈ T(log(pdf_exp))
                end
            end
        end
        @testset "discrete" begin
            # test check that the pdf is the difference of the CDF at adjacent points
            @testset for dist in
                         [DiscreteUniform(10, 30), Poisson(100.0), Binomial(20, 0.3)],
                n in [1, 10, 100],
                i in 1:n

                d = OrderStatistic(dist, n, i)
                xs = quantile(dist, 0.01):quantile(dist, 0.99)
                for x in xs
                    p = @inferred pdf(d, x)
                    lp = @inferred logpdf(d, x)
                    @test lp ≈ logdiffcdf(d, x, x - 1)
                    @test p ≈ exp(lp)
                end
            end
        end
    end

    @testset "distribution normalizes to 1" begin
        @testset for dist in [
                Uniform(-2, 1),
                Normal(2, 3),
                Exponential(5),
                DiscreteUniform(10, 40),
                Poisson(100),
            ],
            n in [1, 10, 20],
            i in 1:n

            d = OrderStatistic(dist, n, i)
            Distributions.expectation(one, d) ≈ 1
        end
    end

    @testset "cdf/logcdf/ccdf/logccdf/quantile/cquantile" begin
        # test against the exact formula computed using BigFloats
        @testset for T in (Float32, Float64)
            dists = [
                (Uniform(T(-2), T(1)), Uniform(big(-2), big(1))),
                (Normal(T(3), T(2)), Normal(big(3), big(2))),
                (Exponential(T(10)), Exponential(big(10))),
                (DiscreteUniform(1, 10), DiscreteUniform(1, 10)),
                (Poisson(T(20)), Poisson(big(20))),
            ]
            @testset for (dist, bigdist) in dists, n in [10, 100], i in 1:n
                dist isa DiscreteDistribution && T !== Float64 && continue
                d = OrderStatistic(dist, n, i)
                # since density is concentrated around the i/n quantile, sample a point
                # nearby it
                x = quantile(dist, clamp(i / n + (rand() - 1//2) / 10, 1e-4, 1 - 1e-4))
                p = cdf(bigdist, big(x))
                cdf_exp = sum(i:n) do j
                    c = binomial(big(n), big(j))
                    return c * p^j * (1 - p)^(n - j)
                end
                @test @inferred(T, cdf(d, x)) ≈ T(cdf_exp)
                @test cdf(d, maximum(d)) ≈ one(T)
                @test cdf(d, minimum(d) - 1) ≈ zero(T)
                @test @inferred(T, logcdf(d, x)) ≈ T(log(cdf_exp))
                @test logcdf(d, maximum(d)) ≈ zero(T)
                @test logcdf(d, minimum(d) - 1) ≈ -Inf
                @test @inferred(T, ccdf(d, x)) ≈ T(1 - cdf_exp)
                @test ccdf(d, maximum(d)) ≈ zero(T)
                @test ccdf(d, minimum(d) - 1) ≈ one(T)
                @test @inferred(T, logccdf(d, x)) ≈ T(log(1 - cdf_exp))
                @test logccdf(d, maximum(d)) ≈ -Inf
                @test logccdf(d, minimum(d) - 1) ≈ zero(T)
                q = cdf(d, x)
                if dist isa DiscreteDistribution
                    # for discrete distributions, tiny numerical error can cause the wrong
                    # integer value to be returned.
                    q -= sqrt(eps(T))
                end
                xq = @inferred(T, quantile(d, q))
                xqc = @inferred(T, cquantile(d, 1 - q))
                @test xq ≈ xqc
                @test isapprox(xq, T(x); atol=1e-4) ||
                    (dist isa DiscreteDistribution && xq < x)
            end
        end
    end

    @testset "rand" begin
        @testset for T in [Float32, Float64],
            dist in [Uniform(T(-2), T(1)), Normal(T(1), T(2))]

            d = OrderStatistic(dist, 10, 5)
            rng = Random.default_rng()
            Random.seed!(rng, 42)
            x = @inferred(rand(rng, d))
            xs = @inferred(rand(rng, d, 10))
            S = eltype(rand(dist))
            @test typeof(x) === S
            @test eltype(xs) === S
            @test length(xs) == 10

            Random.seed!(rng, 42)
            x2 = rand(rng, d)
            xs2 = rand(rng, d, 10)
            @test x2 == x
            @test xs2 == xs
        end

        ndraws = 100_000
        nchecks = 4 * 2 * 111  # NOTE: update if the below number of tests changes
        α = (0.01 / nchecks) / 2  # multiple correction
        tol = quantile(Normal(), 1 - α)

        @testset for dist in [Uniform(), Exponential(), Poisson(20), Binomial(20, 0.3)]
            @testset for n in [1, 10, 100], i in 1:n
                d = OrderStatistic(dist, n, i)
                x = rand(d, ndraws)
                m, v = mean_and_var(x)
                if dist isa Uniform
                    # Arnold (2008). A first course in order statistics. Eqs 2.2.20-21
                    m_exact = i / (n + 1)
                    v_exact = m_exact * (1 - m_exact) / (n + 2)
                elseif dist isa Exponential
                    # Arnold (2008). A first course in order statistics. Eqs 4.6.6-7
                    m_exact = sum(k -> inv(n - k + 1), 1:i)
                    v_exact = sum(k -> inv((n - k + 1)^2), 1:i)
                elseif dist isa DiscreteUnivariateDistribution
                    # estimate mean and variance with explicit sum, Eqs 3.2.6-7 from
                    # Arnold (2008). A first course in order statistics.
                    xs = 0:quantile(dist, 0.9999)
                    m_exact = sum(x -> ccdf(d, x), xs)
                    v_exact = 2 * sum(x -> x * ccdf(d, x), xs) + m_exact - m_exact^2
                end
                # compute asymptotic sample standard deviation
                mean_std = sqrt(v_exact / ndraws)
                var_std = sqrt((moment(x, 4) - v_exact^2) / ndraws)
                @test m ≈ m_exact atol = (tol * mean_std)
                @test v ≈ v_exact atol = (tol * var_std)
            end
        end
    end
end
