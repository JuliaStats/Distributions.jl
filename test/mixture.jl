using Distributions, Random
using Test
using ForwardDiff: Dual


# Core testing procedure

function test_mixture(g::UnivariateMixture, n::Int, ns::Int,
                      rng::Union{AbstractRNG, Missing} = missing)
    if g isa UnivariateGMM
        T = eltype(g.means)
    else
        T = eltype(typeof(g))
    end
    X = zeros(T, n)
    for i = 1:n
        X[i] = rand(g)
    end

    K = ncomponents(g)
    pr = @inferred(probs(g))
    @assert length(pr) == K

    # mean
    mu = 0.0
    for k = 1:K
        mu += pr[k] * mean(component(g, k))
    end
    @test @inferred(mean(g)) ≈ mu

    # evaluation of cdf
    cf = zeros(T, n)
    for k = 1:K
        c_k = component(g, k)
        for i = 1:n
            cf[i] += pr[k] * cdf(c_k, X[i])
        end
    end

    for i = 1:n
        @test @inferred(cdf(g, X[i])) ≈ cf[i]
    end
    @test Base.Fix1(cdf, g).(X) ≈ cf

    # evaluation
    P0 = zeros(T, n, K)
    LP0 = zeros(T, n, K)
    for k = 1:K
        c_k = component(g, k)
        for i = 1:n
            x_i = X[i]
            P0[i,k] = pdf(c_k, x_i)
            LP0[i,k] = logpdf(c_k, x_i)
        end
    end

    mix_p0 = P0 * pr
    mix_lp0 = log.(mix_p0)

    for i = 1:n
        @test @inferred(pdf(g, X[i])) ≈ mix_p0[i]
        @test @inferred(logpdf(g, X[i])) ≈ mix_lp0[i]
        @test @inferred(componentwise_pdf(g, X[i])) ≈ vec(P0[i,:])
        @test @inferred(componentwise_logpdf(g, X[i])) ≈ vec(LP0[i,:])
    end

    @test @inferred(map(Base.Fix1(pdf, g), X)) ≈ mix_p0
    @test @inferred(map(Base.Fix1(logpdf, g), X)) ≈ mix_lp0
    @test @inferred(componentwise_pdf(g, X)) ≈ P0
    @test @inferred(componentwise_logpdf(g, X)) ≈ LP0

    # quantile
    αs = float(partype(g))[0.0; 0.49; 0.5; 0.51; 1.0]
    for α in αs
        @test cdf(g, @inferred(quantile(g, α))) ≈ α
    end
    @test @inferred(median(g)) ≈ quantile(g, 1//2)

    # sampling
    # sampling does not work with `Float32` since `AliasTable` does not support `Float32`
    # Ref: https://github.com/JuliaStats/StatsBase.jl/issues/158
    if T <: AbstractFloat && eltype(probs(g)) === Float64
        if ismissing(rng)
            Xs = rand(g, ns)
        else
            Xs = rand(rng, g, ns)
        end
        @test isa(Xs, Vector{T})
        @test length(Xs) == ns
        @test isapprox(mean(Xs), mean(g), atol=0.01)
    end
end

function test_mixture(g::MultivariateMixture, n::Int, ns::Int,
                      rng::Union{AbstractRNG, Missing} = missing)
    X = zeros(length(g), n)
    for i = 1:n
        if ismissing(rng)
            X[:, i] = rand(g)
        else
            X[:, i] = rand(rng, g)
        end
    end

    K = ncomponents(g)
    pr = @inferred(probs(g))
    @assert length(pr) == K

    # mean
    mu = zeros(length(g))
    for k = 1:K
        mu .+= pr[k] .* mean(component(g, k))
    end
    @test @inferred(mean(g)) ≈ mu

    # evaluation
    P0 = zeros(n, K)
    LP0 = zeros(n, K)
    for k = 1:K
        c_k = component(g, k)
        for i = 1:n
            x_i = X[:,i]
            P0[i,k] = pdf(c_k, x_i)
            LP0[i,k] = logpdf(c_k, x_i)
        end
    end

    mix_p0 = P0 * pr
    mix_lp0 = log.(mix_p0)

    for i = 1:n
        x_i = X[:,i]
        @test @inferred(pdf(g, x_i)) ≈ mix_p0[i]
        @test @inferred(logpdf(g, x_i)) ≈ mix_lp0[i]
        @test @inferred(componentwise_pdf(g, x_i)) ≈ vec(P0[i,:])
        @test @inferred(componentwise_logpdf(g, x_i)) ≈ vec(LP0[i,:])
    end
#=
    @show g
    @show size(X)
    @show size(mix_p0)
=#
    @test @inferred(pdf(g, X)) ≈ mix_p0
    @test @inferred(logpdf(g, X)) ≈ mix_lp0
    @test @inferred(componentwise_pdf(g, X)) ≈ P0
    @test @inferred(componentwise_logpdf(g, X)) ≈ LP0

    # sampling
    if ismissing(rng)
        Xs = rand(g, ns)
    else
        Xs = rand(rng, g, ns)
    end
    @test isa(Xs, Matrix{Float64})
    @test size(Xs) == (length(g), ns)
    @test isapprox(vec(mean(Xs, dims=2)), mean(g), atol=0.1)
    @test isapprox(cov(Xs, dims=2)      , cov(g) , atol=0.1)
    @test isapprox(var(Xs, dims=2)      , var(g) , atol=0.1)
end

function test_params(g::AbstractMixtureModel)
    C = eltype(g.components)
    pars = params(g)
    mm = MixtureModel(C, pars...)
    @test g.prior == mm.prior
    @test g.components == mm.components
    @test g == deepcopy(g)
end

function test_params(g::UnivariateGMM)
    pars = params(g)
    mm = UnivariateGMM(pars...)
    @test g == mm
    @test g == deepcopy(g)
end

# Tests

@testset "Testing Mixtures with $key" for (key, rng) in
    Dict("rand(...)" => missing,
         "rand(rng, ...)" => MersenneTwister(123))

    @testset "Testing UnivariateMixture" begin
        g_u = MixtureModel([Normal(), Normal()])
        @test isa(g_u, MixtureModel{Univariate, Continuous, <:Normal})
        @test ncomponents(g_u) == 2
        test_mixture(g_u, 1000, 10^6, rng)
        test_params(g_u)
        @test minimum(g_u) == -Inf
        @test maximum(g_u) == Inf
        @test extrema(g_u) == (-Inf, Inf)
        @test @inferred(median(g_u)) === 0.0
        @test @inferred(quantile(g_u, 0.5f0)) === 0.0

        g_u = MixtureModel(Normal{Float64}, [(0.0, 1.0), (2.0, 1.0), (-4.0, 1.5)], [0.2, 0.5, 0.3])
        @test isa(g_u, MixtureModel{Univariate,Continuous,<:Normal})
        @test ncomponents(g_u) == 3
        test_mixture(g_u, 1000, 10^6, rng)
        test_params(g_u)
        @test minimum(g_u) == -Inf
        @test maximum(g_u) == Inf
        @test extrema(g_u) == (-Inf, Inf)

        g_u = MixtureModel(Normal{Float32}, [(0f0, 1f0), (0f0, 2f0)], [0.4f0, 0.6f0])
        @test isa(g_u, MixtureModel{Univariate,Continuous,<:Normal})
        @test ncomponents(g_u) == 2
        test_mixture(g_u, 1000, 10^6, rng)
        test_params(g_u)
        @test minimum(g_u) == -Inf
        @test maximum(g_u) == Inf
        @test extrema(g_u) == (-Inf, Inf)
        @test @inferred(median(g_u)) === 0f0
        @test @inferred(quantile(g_u, 0.5f0)) === 0f0

        g_u = MixtureModel([TriangularDist(-1,2,0),TriangularDist(-.5,3,1),TriangularDist(-2,0,-1)])
        @test minimum(g_u) ≈ -2.0
        @test maximum(g_u) ≈ 3.0
        @test extrema(g_u) == (minimum(g_u), maximum(g_u))
        @test insupport(g_u, 2.5) == true
        @test insupport(g_u, 3.5) == false

        μ = [0.0, 2.0, -4.0]; σ = [1.0, 1.2, 1.5]; p = [0.2, 0.5, 0.3]
        for T = [Float64, Dual]
            g_u = @inferred UnivariateGMM(map(Dual, μ), map(Dual, σ),
                                          Categorical(map(Dual, p)))
            @test isa(g_u, UnivariateGMM)
            @test ncomponents(g_u) == 3
            test_mixture(g_u, 1000, 10^6, rng)
            test_params(g_u)
            @test minimum(g_u) == -Inf
            @test maximum(g_u) == Inf
            @test extrema(g_u) == (-Inf, Inf)
        end

        # https://github.com/JuliaStats/Distributions.jl/issues/1121
        @test @inferred(logpdf(UnivariateGMM(μ, σ, Categorical(p)), 42)) isa Float64

        @testset "Product 0 NaN in mixtures" begin
            distributions = [
                Normal(-1.0, 0.3),
                Normal(0.0, 0.5),
                Normal(3.0, 1.0),
                Normal(NaN, 1.0),
            ]
            priors = [0.25, 0.25, 0.5, 0.0]
            gmm_normal = MixtureModel(distributions, priors)
            for x in rand(10)
                result = pdf(gmm_normal, x)
                @test !isnan(result)
            end
        end
    end

    @testset "Testing MultivariatevariateMixture" begin
        g_m = MixtureModel(
            IsoNormal[ MvNormal([0.0, 0.0], I),
                       MvNormal([0.2, 1.0], I),
                       MvNormal([-0.5, -3.0], 1.6 * I) ],
            [0.2, 0.5, 0.3])
        @test isa(g_m, MixtureModel{Multivariate, Continuous, IsoNormal})
        @test length(components(g_m)) == 3
        @test length(g_m) == 2
        @test insupport(g_m, [0.0, 0.0]) == true
        test_mixture(g_m, 1000, 10^6, rng)
        test_params(g_m)

        u1 =  Uniform()
        u2 =  Uniform(1.0, 2.0)
        utot =Uniform(0.0, 2.0)

        # mixture supposed to be a uniform on [0.0,2.0]
        unif_mixt =  MixtureModel([u1,u2])
        @test var(utot) ≈  var(unif_mixt)
        @test mean(utot) ≈ mean(unif_mixt)
        for x in -1.0:0.5:2.5
            @test cdf(utot,x) ≈ cdf(utot,x)
        end
    end

    # issue #1501
    @testset "quantile of mixture with single component" begin
        for T in (Float32, Float64)
            d = MixtureModel([Normal{T}(T(1), T(0))])
            for p in (0.2, 0.2f0, 1//3)
                @test @inferred(quantile(d, p)) == 1
                @test @inferred(median(d)) == 1
            end
        end
    end
end
