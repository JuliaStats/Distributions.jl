# Tests for Dirichlet distribution

using  Distributions
using Test, Random, LinearAlgebra
using ChainRulesCore
using ChainRulesTestUtils
using FiniteDifferences

Random.seed!(34567)

rng = MersenneTwister(123)

@testset "Testing Dirichlet with $key" for (key, func) in
    Dict("rand(...)" => [rand, rand],
         "rand(rng, ...)" => [dist -> rand(rng, dist), (dist, n) -> rand(rng, dist, n)])

    for T in (Int, Float64)
        d = Dirichlet(3, T(2))

        @test length(d) == 3
        @test eltype(d) === T
        @test d.alpha == [2, 2, 2]
        @test d.alpha0 == 6

        @test mean(d) ≈ fill(1/3, 3)
        @test mode(d) ≈ fill(1/3, 3)
        @test cov(d)  ≈ [8 -4 -4; -4 8 -4; -4 -4 8] / (36 * 7)
        @test var(d)  ≈ diag(cov(d))

        r = Vector{Float64}(undef, 3)
        Distributions.dirichlet_mode!(r, d.alpha, d.alpha0)
        @test r ≈ fill(1/3, 3)

        @test pdf(Dirichlet([1, 1]), [0, 1]) ≈ 1
        @test pdf(Dirichlet([1f0, 1f0]), [0f0, 1f0]) ≈ 1
        @test typeof(pdf(Dirichlet([1f0, 1f0]), [0f0, 1f0])) === Float32

        @test iszero(pdf(d, [-1, 1, 0]))
        @test iszero(pdf(d, [0, 0, 1]))
        @test pdf(d, [0.2, 0.3, 0.5]) ≈ 3.6
        @test pdf(d, [0.4, 0.5, 0.1]) ≈ 2.4
        @test logpdf(d, [0.2, 0.3, 0.5]) ≈ log(3.6)
        @test logpdf(d, [0.4, 0.5, 0.1]) ≈ log(2.4)

        x = func[2](d, 100)
        p = pdf(d, x)
        lp = logpdf(d, x)
        for i in 1 : size(x, 2)
            @test lp[i] ≈ logpdf(d, x[:,i])
            @test p[i]  ≈ pdf(d, x[:,i])
        end

        v = [2, 1, 3]
        d = Dirichlet(T.(v))

        @test eltype(d) === T
        @test Dirichlet([2, 1, 3]).alpha == d.alpha

        @test length(d) == length(v)
        @test d.alpha == v
        @test d.alpha0 == sum(v)
        @test d == Dirichlet{T}(params(d)...)
        @test d == deepcopy(d)

        @test mean(d) ≈ v / sum(v)
        @test cov(d)  ≈ [8 -2 -6; -2 5 -3; -6 -3 9] / (36 * 7)
        @test var(d)  ≈ diag(cov(d))

        @test pdf(d, [0.2, 0.3, 0.5]) ≈ 3
        @test pdf(d, [0.4, 0.5, 0.1]) ≈ 0.24
        @test logpdf(d, [0.2, 0.3, 0.5]) ≈ log(3)
        @test logpdf(d, [0.4, 0.5, 0.1]) ≈ log(0.24)

        x = func[2](d, 100)
        p = pdf(d, x)
        lp = logpdf(d, x)
        for i in 1 : size(x, 2)
            @test p[i]  ≈ pdf(d, x[:,i])
            @test lp[i] ≈ logpdf(d, x[:,i])
        end

        # Sampling

        x = func[1](d)
        @test isa(x, Vector{Float64})
        @test length(x) == 3

        x = func[2](d, 10)
        @test isa(x, Matrix{Float64})
        @test size(x) == (3, 10)

        v = [2, 1, 3]
        d = Dirichlet(Float32.(v))
        @test eltype(d) === Float32

        x = func[1](d)
        @test isa(x, Vector{Float32})
        @test length(x) == 3

        x = func[2](d, 10)
        @test isa(x, Matrix{Float32})
        @test size(x) == (3, 10)


        # Test MLE

        v = [2, 1, 3]
        d = Dirichlet(v)

        n = 10000
        x = func[2](d, n)
        x = x ./ sum(x, dims=1)

        r = fit_mle(Dirichlet, x)
        @test r.alpha ≈ d.alpha atol=0.25
        r = fit(Dirichlet{Float32}, x)
        @test r.alpha ≈ d.alpha atol=0.25

        # r = fit_mle(Dirichlet, x, fill(2.0, n))
        # @test r.alpha ≈ d.alpha atol=0.25
    end
end

@testset "Dirichlet: entropy" begin
    α = exp.(rand(2))
    @test entropy(Dirichlet(α)) ≈ entropy(Beta(α...))

    N = 10
    @test entropy(Dirichlet(N, 1)) ≈ -loggamma(N)
    @test entropy(Dirichlet(ones(N))) ≈ -loggamma(N)
end

@testset "Dirichlet: ChainRules (length=$n)" for n in (2, 10)
    alpha = rand(n)
    d = Dirichlet(alpha)

    @testset "constructor $T" for T in (Dirichlet, Dirichlet{Float64})
        # Avoid issues with finite differencing if values in `alpha` become negative or zero
        # by using forward differencing
        test_frule(T, alpha; fdm=forward_fdm(5, 1))
        test_rrule(T, alpha; fdm=forward_fdm(5, 1))
    end

    @testset "_logpdf" begin
        # `x1` is in the support, `x2` isn't
        x1 = rand(n)
        x1 ./= sum(x1)
        x2 = x1 .+ 1

        # Use special finite differencing method that tries to avoid moving outside of the
        # support by limiting the range of the points around the input that are evaluated
        fdm = central_fdm(5, 1; max_range=1e-9)

        for x in (x1, x2)
            # We have to adjust the tolerance since the finite differencing method is rough
            test_frule(Distributions._logpdf, d, x; fdm=fdm, rtol=1e-5, nans=true)
            test_rrule(Distributions._logpdf, d, x; fdm=fdm, rtol=1e-5, nans=true)
        end
    end
end
