using Distributions
using Base.Test

# Core testing procedure

function test_mixture(g::UnivariateMixture, n::Int, ns::Int)
    X = zeros(n)
    for i = 1:n
        X[i] = rand(g)
    end

    K = ncomponents(g)
    pr = probs(g)
    @assert length(pr) == K

    # mean
    mu = 0.0
    for k = 1:K
        mu += pr[k] * mean(component(g, k))
    end
    @test_approx_eq mean(g) mu

    # evaluation
    P0 = zeros(n, K)
    LP0 = zeros(n, K)
    for k = 1:K
        c_k = component(g, k)
        for i = 1:n
            x_i = X[i]
            P0[i,k] = pdf(c_k, x_i)
            LP0[i,k] = logpdf(c_k, x_i)
        end
    end

    mix_p0 = P0 * pr
    mix_lp0 = log(mix_p0)

    for i = 1:n
        @test_approx_eq pdf(g, X[i]) mix_p0[i]
        @test_approx_eq logpdf(g, X[i]) mix_lp0[i]
        @test_approx_eq componentwise_pdf(g, X[i]) vec(P0[i,:])
        @test_approx_eq componentwise_logpdf(g, X[i]) vec(LP0[i,:])
    end

    @test_approx_eq pdf(g, X) mix_p0
    @test_approx_eq logpdf(g, X) mix_lp0
    @test_approx_eq componentwise_pdf(g, X) P0
    @test_approx_eq componentwise_logpdf(g, X) LP0

    # sampling
    Xs = rand(g, ns)
    @test isa(Xs, Vector{Float64})
    @test length(Xs) == ns
    @test_approx_eq_eps mean(Xs) mean(g) 0.01
end

function test_mixture(g::MultivariateMixture, n::Int, ns::Int)
    X = zeros(length(g), n)
    for i = 1:n
        X[:,i] = rand(g)
    end

    K = ncomponents(g)
    pr = probs(g)
    @assert length(pr) == K

    # mean
    mu = 0.0
    for k = 1:K
        mu += pr[k] * mean(component(g, k))
    end
    @test_approx_eq mean(g) mu

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
    mix_lp0 = log(mix_p0)

    for i = 1:n
        x_i = X[:,i]
        @test_approx_eq pdf(g, x_i) mix_p0[i]
        @test_approx_eq logpdf(g, x_i) mix_lp0[i]
        @test_approx_eq componentwise_pdf(g, x_i) vec(P0[i,:])
        @test_approx_eq componentwise_logpdf(g, x_i) vec(LP0[i,:])
    end

    @test_approx_eq pdf(g, X) mix_p0
    @test_approx_eq logpdf(g, X) mix_lp0
    @test_approx_eq componentwise_pdf(g, X) P0
    @test_approx_eq componentwise_logpdf(g, X) LP0

    # sampling
    Xs = rand(g, ns)
    @test isa(Xs, Matrix{Float64})
    @test size(Xs) == (length(g), ns)
    @test_approx_eq_eps vec(mean(Xs, 2)) mean(g) 0.01
end



# Tests

println("    testing UnivariateMixture")

g_u = MixtureModel(Normal, [(0.0, 1.0), (2.0, 1.0), (-4.0, 1.5)], [0.2, 0.5, 0.3])
@test isa(g_u, MixtureModel{Univariate, Continuous, Normal})
@test ncomponents(g_u) == 3
test_mixture(g_u, 1000, 10^6)

g_u = UnivariateGMM([0.0, 2.0, -4.0], [1.0, 1.2, 1.5], Categorical([0.2, 0.5, 0.3]))
@test isa(g_u, UnivariateGMM)
@test ncomponents(g_u) == 3
test_mixture(g_u, 1000, 10^6)

println("    testing MultivariateMixture")
g_m = MixtureModel(
    IsoNormal[ MvNormal([0.0, 0.0], 1.0),
               MvNormal([0.2, 1.0], 1.0),
               MvNormal([-0.5, -3.0], 1.6) ],
    [0.2, 0.5, 0.3])
@test isa(g_m, MixtureModel{Multivariate, Continuous, IsoNormal})
@test length(components(g_m)) == 3
@test length(g_m) == 2
test_mixture(g_m, 1000, 10^6)
