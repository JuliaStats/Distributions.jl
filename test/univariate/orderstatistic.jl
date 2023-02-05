using Test, Distributions
using QuadGK: quadgk
using Random

@testset "OrderStatistic" begin
    @testset "continuous" begin
        @testset "basic" begin
            for dist in [Uniform(), Normal()], n in [1, 2, 10], i in 1:n
                d = OrderStatistic(dist, n, i)
                @test d isa OrderStatistic
                @test d isa ContinuousUnivariateDistribution
                @test d.dist === dist
                @test d.n == n
                @test d.i == i
            end
            @test_throws ArgumentError OrderStatistic(Normal(), 0, 1)
            OrderStatistic(Normal(), 0, 1; check_args=false)
            @test_throws ArgumentError OrderStatistic(Normal(), 10, 11)
            OrderStatistic(Normal(), 10, 11; check_args=false)
            @test_throws ArgumentError OrderStatistic(Normal(), 10, 0)
            OrderStatistic(Normal(), 10, 0; check_args=false)
        end

        @testset "params" begin
            for dist in [Uniform(), Normal()], n in [1, 2, 10], i in 1:n
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
            end
        end

        @testset "cdf" begin
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
                    @test pdf(d, x) ≈ T(pdf_exp) atol = sqrt(eps(T))
                end
            end
        end

        @testset "pdf/logpdf" begin
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

        @testset "pdf integrates to 1" begin
            @testset for dist in [Uniform(-2, 1), Normal(2, 3), Exponential(5)],
                n in [1, 10, 20],
                i in 1:n

                d = OrderStatistic(dist, n, i)
                @test quadgk(x -> pdf(d, x), minimum(d), maximum(d))[1] ≈ 1
            end
        end

        @testset "cdf/logcdf/ccdf/logccdf/quantile/cquantile" begin
            # test against the exact formula computed using BigFloats
            @testset for T in (Float32, Float64)
                @testset for dist in
                             [Uniform(T(-2), T(1)), Normal(T(3), T(2)), Exponential(T(10))],
                    n in [1, 10, 100],
                    i in 1:n

                    d = OrderStatistic(dist, n, i)
                    # since density is concentrated around the i/n quantile, sample a point
                    # nearby it
                    x = quantile(dist, clamp(i / n + (rand() - 1//2) / 10, 0, 1))
                    p = cdf(dist, big(x))
                    cdf_exp = sum(i:n) do j
                        c = binomial(big(n), big(j))
                        return c * p^j * (1 - p)^(n - j)
                    end
                    @test @inferred(T, cdf(d, x)) ≈ T(cdf_exp)
                    @test @inferred(T, logcdf(d, x)) ≈ T(log(cdf_exp))
                    @test @inferred(T, ccdf(d, x)) ≈ T(1 - cdf_exp)
                    @test @inferred(T, logccdf(d, x)) ≈ T(log(1 - cdf_exp))
                    q = cdf(d, x)
                    @test @inferred(T, quantile(d, q)) ≈ T(x)
                    @test @inferred(T, cquantile(d, 1 - q)) ≈ T(x)
                end
            end
        end

        @testset "rand" begin
            @testset for T in [Float32, Float64]
                dist = Uniform(T(-2), T(1))
                d = OrderStatistic(dist, 10, 5)
                rng = Random.default_rng()
                @inferred T rand(rng, d)
                @inferred Vector{T} rand(rng, d, 10)
            end

            ndraws = 10_000
            @testset "Uniform()" begin
                # test against known mean and variance of order statistics
                @testset for n in [1, 10, 100], i in 1:n
                    d = OrderStatistic(Uniform(), n, i)
                    x = rand(d, ndraws)
                    m, v = mean_and_var(x)
                    m_exact = i//(n + 1)
                    v_exact = (m * (1 - m) / (n + 2))
                    # compute asymptotic sample standard deviation
                    mean_std = sqrt(v_exact / ndraws)
                    var_std = sqrt((moment(x, 4) - v_exact^2) / ndraws)
                    @test m ≈ m_exact atol = (3 * mean_std)
                    @test v ≈ v_exact atol = (4 * var_std)
                end
            end

            @testset "Exponential()" begin
                # test against known mean and variance of order statistics
                @testset for n in [1, 10, 100], i in 1:n
                    d = OrderStatistic(Exponential(), n, i)
                    x = rand(d, ndraws)
                    m, v = mean_and_var(x)
                    m_exact = sum(r -> inv(n - r + 1), 1:i)
                    v_exact = sum(r -> inv((n - r + 1)^2), 1:i)
                    # compute asymptotic sample standard deviation
                    mean_std = sqrt(v_exact / ndraws)
                    var_std = sqrt((moment(x, 4) - v_exact^2) / ndraws)
                    @test m ≈ m_exact atol = (3 * mean_std)
                    @test v ≈ v_exact atol = (4 * var_std)
                end
            end
        end
    end
end
