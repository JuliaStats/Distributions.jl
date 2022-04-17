# Tests for Dirichlet distribution

using  Distributions
using Test, Random, LinearAlgebra
using ChainRulesCore
using ChainRulesTestUtils

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

@testset "Dirichlet differentiation $n" for n in (2, 10)
    alpha = rand(n)
    Δalpha = randn(n)
    d, ∂d = ChainRulesCore.frule((nothing, Δalpha), Dirichlet, alpha)
    ChainRulesTestUtils.test_frule(Dirichlet ⊢ ChainRulesCore.NoTangent(), alpha ⊢ Δalpha)
    _, dp = ChainRulesCore.rrule(Dirichlet, alpha)
    ChainRulesTestUtils.test_rrule(Dirichlet{Float64} ⊢ ChainRulesCore.NoTangent(), alpha)
    x = rand(n)
    x ./= sum(x)
    Δx = 0.05 * rand(n)
    Δx .-= mean(Δx)
    # such that x ∈ Δ, x + Δx ∈ Δ
    ChainRulesTestUtils.test_frule(Distributions._logpdf ⊢ ChainRulesCore.NoTangent(), d, x ⊢ Δx)
    @testset "finite diff f/r-rule logpdf" begin
        for _ in 1:10
            x = rand(n)
            x ./= sum(x)
            Δx = 0.005 * rand(n)
            Δx .-= mean(Δx)
            if insupport(d, x + Δx) && insupport(d, x - Δx)
                y, pullback = ChainRulesCore.rrule(Distributions._logpdf, d, x)
                yf, Δy = ChainRulesCore.frule(
                    (
                        ChainRulesCore.NoTangent(),
                        map(zero, ChainRulesTestUtils.rand_tangent(d)),
                        Δx,
                    ),
                    Distributions._logpdf,
                    d, x,
                )
                y2 = Distributions._logpdf(d, x + Δx)
                y1 = Distributions._logpdf(d, x - Δx)
                @test isfinite(y)
                @test y == yf
                @test Δy ≈ y2 - y atol=5e-3
                _, ∂d, ∂x = pullback(1.0)
                @test y2 - y1 ≈ dot(2Δx, ∂x) atol=5e-3 rtol=1e-6
                # mutating alpha only to compute a new y, changing only this term and not the others in Dirichlet
                Δalpha = 0.03 * rand(n)
                Δalpha .-= mean(Δalpha)
                @assert all(>=(0), alpha + Δalpha)
                d.alpha .+= Δalpha
                ya = Distributions._logpdf(d, x)
                # resetting alpha
                d.alpha .-= Δalpha
                @test ya - y ≈ dot(Δalpha, ∂d.alpha) atol=5e-5 rtol=1e-6
            end
        end
    end
end
