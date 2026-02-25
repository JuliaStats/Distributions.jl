using Distributions, LinearAlgebra, Random, SpecialFunctions, Statistics, Test

@testset "JointOrderStatistics" begin
    Random.seed!(123)

    @testset "check_args" begin
        dist = Normal()
        JointOrderStatistics(dist, 2, 1:2)
        JointOrderStatistics(dist, 3, 2:3)
        JointOrderStatistics(dist, 5, [2, 3])
        @test_throws DomainError JointOrderStatistics(dist, 0, 1:2)
        @test_throws DomainError JointOrderStatistics(dist, 2, 2:3)
        @test_throws DomainError JointOrderStatistics(dist, 3, 0:3)
        @test_throws DomainError JointOrderStatistics(dist, 5, 3:-1:2)
        @test_throws DomainError JointOrderStatistics(dist, 5, 2:1:1)
        @test_throws DomainError JointOrderStatistics(dist, 0, [1, 2])
        @test_throws DomainError JointOrderStatistics(dist, 2, [2, 3])
        @test_throws DomainError JointOrderStatistics(dist, 3, [0, 1, 2, 3])
        @test_throws DomainError JointOrderStatistics(dist, 5, Int[])
        @test_throws DomainError JointOrderStatistics(dist, 5, (3, 2))
        @test_throws DomainError JointOrderStatistics(dist, 5, (3, 3))
        JointOrderStatistics(dist, 0, 1:2; check_args=false)
        JointOrderStatistics(dist, 2, 2:3; check_args=false)
        JointOrderStatistics(dist, 3, 0:3; check_args=false)
        JointOrderStatistics(dist, 5, 3:-1:2; check_args=false)
        JointOrderStatistics(dist, 5, 2:1:1; check_args=false)
        JointOrderStatistics(dist, 0, [1, 2]; check_args=false)
        JointOrderStatistics(dist, 2, [2, 3]; check_args=false)
        JointOrderStatistics(dist, 3, [0, 1, 2, 3]; check_args=false)
        JointOrderStatistics(dist, 5, Int[]; check_args=false)
        JointOrderStatistics(dist, 5, (3, 2); check_args=false)
        JointOrderStatistics(dist, 5, (3, 3); check_args=false)
    end

    @testset for T in [Float32, Float64],
        dist in [Uniform(T(2), T(10)), Exponential(T(10)), Normal(T(100), T(10))],
        n in [16, 40],
        r in [
            1:n,
            ([i, j] for j in 2:n for i in 1:min(10, j - 1))...,
            vcat(2:4, (n - 10):(n - 5)),
            (2, n ÷ 2, n - 5),
        ]

        d = JointOrderStatistics(dist, n, r)

        @testset "basic" begin
            @test d isa JointOrderStatistics
            @test d.dist === dist
            @test d.n === n
            @test d.ranks === r
            @test length(d) == length(r)
            @test params(d) == (params(dist)..., d.n, d.ranks)
            @test partype(d) === partype(dist)
            @test eltype(d) === eltype(dist)

            length(r) == n && @test JointOrderStatistics(dist, n) == d
        end

        @testset "support" begin
            @test minimum(d) == fill(minimum(dist), length(r))
            @test maximum(d) == fill(maximum(dist), length(r))
            x = sort(rand(dist, length(r)))
            x2 = sort(rand(dist, length(r) + 1))
            @test insupport(d, x)
            if length(x) > 1
                @test !insupport(d, reverse(x))
                @test !insupport(d, x[1:(end - 1)])
            end
            @test !insupport(d, x2)
            @test !insupport(d, fill(NaN, length(x)))
        end

        @testset "pdf/logpdf" begin
            x = convert(Vector{T}, sort(rand(dist, length(r))))
            @test @inferred(logpdf(d, x)) isa T
            @test @inferred(pdf(d, x)) isa T

            if length(r) == 1
                @test logpdf(d, x) ≈ logpdf(OrderStatistic(dist, n, r[1]), x[1])
                @test pdf(d, x) ≈ pdf(OrderStatistic(dist, n, r[1]), x[1])
            elseif length(r) == 2
                i, j = r
                xi, xj = x
                lc = T(
                    logfactorial(n) - logfactorial(i - 1) - logfactorial(n - j) -
                    logfactorial(j - i - 1),
                )
                lp = (
                    lc +
                    (i - 1) * logcdf(dist, xi) +
                    (n - j) * logccdf(dist, xj) +
                    (j - i - 1) * logdiffcdf(dist, xj, xi) +
                    logpdf(dist, xi) +
                    logpdf(dist, xj)
                )
                @test logpdf(d, x) ≈ lp
                @test pdf(d, x) ≈ exp(lp)
            elseif collect(r) == 1:n
                @test logpdf(d, x) ≈ sum(Base.Fix1(logpdf, d.dist), x) + loggamma(T(n + 1))
                @test pdf(d, x) ≈ exp(logpdf(d, x))
            end

            @testset "no density for vectors out of support" begin
                # check unsorted vectors have 0 density
                x2 = copy(x)
                x2[1], x2[2] = x2[2], x2[1]
                @test logpdf(d, x2) == T(-Inf)
                @test pdf(d, x2) == zero(T)

                x3 = copy(x)
                x3[end-1], x3[end] = x3[end], x3[end-1]
                @test logpdf(d, x3) == T(-Inf)
                @test pdf(d, x3) == zero(T)

                # check out of support of original distribution
                if islowerbounded(dist)
                    x4 = copy(x)
                    x4[1] = minimum(dist) - 1
                    @test logpdf(d, x4) == T(-Inf)
                    @test pdf(d, x4) == zero(T)
                end
            end
        end
    end

    @testset "rand" begin
        @testset for T in [Float32, Float64]
            dist = Uniform(T(-2), T(1))
            d = JointOrderStatistics(dist, 10, 1:10)
            S = typeof(rand(dist))

            v = rand(d)
            @test v isa Vector{S}
            @test insupport(d, v)
            @test size(v) == (10,)

            rng = Random.default_rng()
            Random.seed!(rng, 42)
            x = @inferred(rand(rng, d, 20))
            @test x isa Matrix{S}
            @test size(x) == (10, 20)
            @test all(xi -> insupport(d, xi), eachcol(x))

            Random.seed!(rng, 42)
            x2 = rand(rng, d, 20)
            @test x2 == x
        end

        ndraws = 300_000
        dists = [Uniform(), Exponential(), DiscreteUniform(1, 10), Binomial(10, 0.3)]

        @testset "marginal mean and standard deviation" begin
            n = 20
            rs = [1:n, [1, n], vcat(1:7, 12:17)]
            @testset for dist in dists, r in rs
                d = JointOrderStatistics(dist, n, r)
                x = rand(d, ndraws)
                @test all(xi -> insupport(d, xi), eachcol(x))

                m = mean(x; dims=2)
                v = var(x; mean=m, dims=2)
                if dist isa Uniform
                    # Arnold (2008). A first course in order statistics. eq 2.2.20-21
                    m_exact = r ./ (n + 1)
                    v_exact = @. (m_exact * (1 - m_exact) / (n + 2))
                elseif dist isa Exponential
                    # Arnold (2008). A first course in order statistics. eq 4.6.6-7
                    m_exact = [sum(k -> inv(n - k + 1), 1:i) for i in r]
                    v_exact = [sum(k -> inv((n - k + 1)^2), 1:i) for i in r]
                elseif Distributions.value_support(typeof(dist)) === Discrete
                    isbounded(dist) || error("Discrete distribution $dist is not bounded")
                    # analytically compute mean and variance using marginal distributions of order statistics
                    x_support = support(dist)
                    probs = stack(
                        map(r) do r_i
                            dist_orderstat = OrderStatistic(dist, n, r_i)
                            return pdf.(Ref(dist_orderstat), x_support)
                        end,
                    )
                    m_exact = probs' * x_support
                    v_exact = (probs' * x_support .^ 2) .- m_exact .^ 2
                end
                # compute asymptotic sample standard deviation
                mean_std = @. sqrt(v_exact / ndraws)
                m4 = dropdims(mapslices(xi -> moment(xi, 4), x; dims=2); dims=2)
                var_std = @. sqrt((m4 - v_exact^2) / ndraws)

                nchecks = length(r)
                α = (0.01 / nchecks) / 2  # multiple correction
                tol = quantile(Normal(), 1 - α)
                for k in eachindex(m, m_exact, v, v_exact, mean_std, var_std)
                    @test m[k] ≈ m_exact[k] atol = (tol * mean_std[k])
                    @test v[k] ≈ v_exact[k] atol = (tol * var_std[k])
                end
            end
        end

        @testset "pairwise correlations" begin
            n = 100
            rs = [    # good mixture of r values with gaps and no gaps
                1:n,
                vcat(1:10, (div(n, 2) - 5):(div(n, 2) + 5), (n - 9):n),
                vcat(10:20, (n - 19):(n - 10)),
                (1, n),
            ]

            nchecks = length(dists) * sum(rs) do r
                m = length(r)
                return div(m * (m - 1), 2)
            end
            α = (0.01 / nchecks) / 2  # multiple correction
            tol = quantile(Normal(), 1 - α) / sqrt(ndraws)

            @testset for dist in dists, r in rs
                d = JointOrderStatistics(dist, n, r)
                x = rand(d, ndraws)
                @test all(xi -> insupport(d, xi), eachcol(x))

                m = length(r)

                if Distributions.value_support(typeof(dist)) === Continuous
                    xcor = cor(x; dims=2)
                    if dist isa Uniform
                        # Arnold (2008). A first course in order statistics. Eq 2.3.16
                        s = @. n - r + 1
                        xcor_exact = Symmetric(
                            sqrt.((r .* collect(s)') ./ (collect(r)' .* s))
                        )
                    elseif dist isa Exponential
                        # Arnold (2008).  A first course in order statistics. Eq 4.6.8
                        v = [sum(k -> inv((n - k + 1)^2), 1:i) for i in r]
                        xcor_exact = Symmetric(sqrt.(v ./ v'))
                    end
                    for ii in 1:m, ji in (ii + 1):m
                        i = r[ii]
                        j = r[ji]
                        ρ = xcor[ii, ji]
                        ρ_exact = xcor_exact[ii, ji]
                        # use variance-stabilizing transformation, recommended in §3.6 of
                        # Van der Vaart, A. W. (2000). Asymptotic statistics (Vol. 3).
                        @test atanh(ρ) ≈ atanh(ρ_exact) atol = tol
                    end
                elseif isbounded(dist)
                    # for discrete distributions, asymptotic joint distribution of order statistics is not necessarily
                    # multivariate normal, so we can't use the above check. Instead, we use the asymptotic distribution
                    # of the sample covariance matrix.
                    xcov = cov(x; dims=2)
                    x_support = support(dist)
                    probs = stack(
                        map(r) do r_i
                            dist_orderstat = OrderStatistic(dist, n, r_i)
                            return pdf.(Ref(dist_orderstat), x_support)
                        end,
                    )
                    m_exact = probs' * x_support
                    var_exact = probs' * x_support .^ 2 .- m_exact .^ 2
                    kurt_marginal = probs' * (x_support .- m_exact') .^ 4

                    xcov_exact = zeros(length(r), length(r))
                    x4_central_exact = zeros(length(r), length(r))
                    for ii in 1:m
                        i = r[ii]
                        xcov_exact[ii, ii] = var_exact[ii]
                        x4_central_exact[ii, ii] = kurt_marginal[ii]
                        for ji in (ii + 1):m
                            j = r[ji]
                            dist_pairwise = JointOrderStatistics(dist, n, (i, j))
                            for (xi, xj) in Iterators.product(x_support, x_support)
                                xi <= xj || continue
                                pij = pdf(dist_pairwise, [xi, xj])
                                zi = xi - m_exact[ii]
                                zj = xj - m_exact[ji]
                                xcov_exact[ii, ji] += pij * zi * zj
                                x4_central_exact[ii, ji] += pij * (zi * zj)^2
                            end
                        end
                    end
                    xcov_exact = Symmetric(xcov_exact, :U)
                    x4_central_exact = Symmetric(x4_central_exact, :U)
                    # asymptotic variance of sample covariance from Theorem 1.2.17 of
                    # Muirhead (1982), Aspects of Multivariate Statistical Theory
                    xcov_var_scaled = x4_central_exact .- xcov_exact .^ 2
                    @test all((abs.(xcov_exact - xcov) ./ sqrt.(xcov_var_scaled)) .< tol)
                end
            end
        end
    end
end
