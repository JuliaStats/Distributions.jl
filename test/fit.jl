# Testing:
#
#  - computation of sufficient statistics
#  - distribution fitting (i.e. estimation)
#

using Distributions
using OffsetArrays
using Test, Random, LinearAlgebra


n0 = 100
N = 10^5

rng = MersenneTwister(123)

const funcs = ([rand,rand], [dist -> rand(rng, dist), (dist, n) -> rand(rng, dist, n)])

@testset "Testing fit for DiscreteUniform" begin
    for func in funcs
        w = func[1](n0)

        x = func[2](DiscreteUniform(10, 15), n0)
        d = fit(DiscreteUniform, x)
        @test isa(d, DiscreteUniform)
        @test minimum(d) == minimum(x)
        @test maximum(d) == maximum(x)

        d = fit(DiscreteUniform, func[2](DiscreteUniform(10, 15), N))
        @test minimum(d) == 10
        @test maximum(d) == 15
    end
end


@testset "Testing fit for Bernoulli" begin
    for rng in ((), (rng,)), D in (Bernoulli, Bernoulli{Float64}, Bernoulli{Float32})
        v = rand(rng..., n0)
        z = rand(rng..., Bernoulli(0.7), n0)
        for x in (z, OffsetArray(z, -n0 ÷ 2)), w in (v, OffsetArray(v, -n0 ÷ 2))
            ss = @inferred suffstats(D, x)
            @test ss isa Distributions.BernoulliStats
            @test ss.cnt0 == n0 - count(t->t != 0, z)
            @test ss.cnt1 == count(t->t != 0, z)

            ss = @inferred suffstats(D, x, w)
            @test ss isa Distributions.BernoulliStats
            @test ss.cnt0 ≈ sum(v[z .== 0])
            @test ss.cnt1 ≈ sum(v[z .== 1])

            d = @inferred fit(D, x)
            @test d isa D
            @test mean(d) ≈ count(t->t != 0, z) / n0

            d = @inferred fit(D, x, w)
            @test d isa D
            @test mean(d) ≈ sum(v[z .== 1]) / sum(v)
        end

        z = rand(rng..., Bernoulli(0.7), N)
        for x in (z, OffsetArray(z, -N ÷ 2))
            d = @inferred fit(D, x)
            @test d isa D
            @test mean(d) ≈ 0.7 atol=0.01
        end
    end
end

@testset "Testing fit for Beta" begin
    for func in funcs, dist in (Beta, Beta{Float64})
        d = fit(dist, func[2](dist(1.3, 3.7), N))
        @test isa(d, dist)
        @test isapprox(d.α, 1.3, atol=0.1)
        @test isapprox(d.β, 3.7, atol=0.1)

        d = fit_mle(dist, func[2](dist(1.3, 3.7), N))
        @test isa(d, dist)
        @test isapprox(d.α, 1.3, atol=0.1)
        @test isapprox(d.β, 3.7, atol=0.1)

    end
end

@testset "Testing fit for Binomial" begin
    for rng in ((), (rng,)), D in (Binomial, Binomial{Float64}, Binomial{Float32})
        v = rand(rng..., n0)
        z = rand(rng..., Binomial(100, 0.3), n0)
        for x in (z, OffsetArray(z, -n0 ÷ 2)), w in (v, OffsetArray(v, -n0 ÷ 2))
            ss = @inferred suffstats(D, (100, x))
            @test ss isa Distributions.BinomialStats
            @test ss.ns ≈ sum(z)
            @test ss.ne == n0
            @test ss.n == 100

            ss = @inferred suffstats(D, (100, x), w)
            @test ss isa Distributions.BinomialStats
            @test ss.ns ≈ dot(z, v)
            @test ss.ne ≈ sum(v)
            @test ss.n == 100

            d = @inferred fit(D, (100, x))
            @test d isa D
            @test ntrials(d) == 100
            @test succprob(d) ≈ sum(z) / (n0 * 100)

            d = @inferred fit(D, (100, x), w)
            @test d isa D
            @test ntrials(d) == 100
            @test succprob(d) ≈ dot(z, v) / (sum(v) * 100)
        end

        z = rand(rng..., Binomial(100, 0.3), N)
        for x in (z, OffsetArray(z, -N ÷ 2))
            d = @inferred fit(D, 100, x)
            @test d isa D
            @test ntrials(d) == 100
            @test succprob(d) ≈ 0.3 atol=0.01
        end
    end
end

# Categorical

@testset "Testing fit for Categorical" begin
    for func in funcs
        p = [0.2, 0.5, 0.3]
        x = func[2](Categorical(p), n0)
        w = func[1](n0)

        ss = suffstats(Categorical, (3, x))
        h = Float64[count(v->v == i, x) for i = 1 : 3]
        @test isa(ss, Distributions.CategoricalStats)
        @test ss.h ≈ h

        d = fit(Categorical, (3, x))
        @test isa(d, Categorical)
        @test ncategories(d) == 3
        @test probs(d) ≈ h / sum(h)

        d2 = fit(Categorical, x)
        @test isa(d2, Categorical)
        @test probs(d2) == probs(d)

        ss = suffstats(Categorical, (3, x), w)
        h = Float64[sum(w[x .== i]) for i = 1 : 3]
        @test isa(ss, Distributions.CategoricalStats)
        @test ss.h ≈ h

        d = fit(Categorical, (3, x), w)
        @test isa(d, Categorical)
        @test probs(d) ≈ h / sum(h)

        d = fit(Categorical, suffstats(Categorical, 3, x, w))
        @test isa(d, Categorical)
        @test probs(d) ≈ (h / sum(h))

        d = fit(Categorical, func[2](Categorical(p), N))
        @test isa(d, Categorical)
        @test isapprox(probs(d), p, atol=0.01)
    end
end

@testset "Testing fit for Cauchy" begin
    @test fit(Cauchy, collect(-4.0:4.0)) === Cauchy(0.0, 2.0)
    @test fit(Cauchy{Float64}, collect(-4.0:4.0)) === Cauchy(0.0, 2.0)
end

@testset "Testing fit for Exponential" begin
    for func in funcs, dist in (Exponential, Exponential{Float64})
        w = func[1](n0)
        x = func[2](dist(0.5), n0)

        ss = suffstats(dist, x)
        @test isa(ss, Distributions.ExponentialStats)
        @test ss.sx ≈ sum(x)
        @test ss.sw == n0

        ss = suffstats(dist, x, w)
        @test isa(ss, Distributions.ExponentialStats)
        @test ss.sx ≈ dot(x, w)
        @test ss.sw == sum(w)

        d = fit(dist, x)
        @test isa(d, dist)
        @test scale(d) ≈ mean(x)

        d = fit(dist, x, w)
        @test isa(d, dist)
        @test scale(d) ≈ dot(x, w) / sum(w)

        d = fit(dist, func[2](dist(0.5), N))
        @test isa(d, dist)
        @test isapprox(scale(d), 0.5, atol=0.01)
    end
end

@testset "Testing fit for Normal" begin
    for func in funcs, dist in (Normal, Normal{Float64})
        μ = 11.3
        σ = 3.2
        w = func[1](n0)

        x = func[2](dist(μ, σ), n0)

        ss = suffstats(dist, x)
        @test isa(ss, Distributions.NormalStats)
        @test ss.s  ≈ sum(x)
        @test ss.m  ≈ mean(x)
        @test ss.s2 ≈ sum((x .- ss.m).^2)
        @test ss.tw ≈ n0

        ss = suffstats(dist, x, w)
        @test isa(ss, Distributions.NormalStats)
        @test ss.s  ≈ dot(x, w)
        @test ss.m  ≈ dot(x, w) / sum(w)
        @test ss.s2 ≈ dot((x .- ss.m).^2, w)
        @test ss.tw ≈ sum(w)

        d = fit(dist, x)
        @test isa(d, dist)
        @test d.μ ≈ mean(x)
        @test d.σ ≈ sqrt(mean((x .- d.μ).^2))

        d = fit(dist, x, w)
        @test isa(d, dist)
        @test d.μ ≈ dot(x, w) / sum(w)
        @test d.σ ≈ sqrt(dot((x .- d.μ).^2, w) / sum(w))

        d = fit(dist, func[2](dist(μ, σ), N))
        @test isa(d, dist)
        @test isapprox(d.μ, μ, atol=0.1)
        @test isapprox(d.σ, σ, atol=0.1)
    end
end

@testset "Testing fit for Normal with known moments" begin
    import Distributions.NormalKnownMu, Distributions.NormalKnownSigma
    μ = 11.3
    σ = 3.2

    for func in funcs

        w = func[1](n0)
        x = func[2](Normal(μ, σ), n0)

        ss = suffstats(NormalKnownMu(μ), x)
        @test isa(ss, Distributions.NormalKnownMuStats)
        @test ss.μ == μ
        @test ss.s2 ≈ sum(abs2.(x .- μ))
        @test ss.tw ≈ n0

        ss = suffstats(NormalKnownMu(μ), x, w)
        @test isa(ss, Distributions.NormalKnownMuStats)
        @test ss.μ == μ
        @test ss.s2 ≈ dot((x .- μ).^2, w)
        @test ss.tw ≈ sum(w)

        d = fit_mle(Normal, x; mu=μ)
        @test isa(d, Normal)
        @test d.μ == μ
        @test d.σ ≈ sqrt(mean((x .- d.μ).^2))

        d = fit_mle(Normal, x, w; mu=μ)
        @test isa(d, Normal)
        @test d.μ == μ
        @test d.σ ≈ sqrt(dot((x .- d.μ).^2, w) / sum(w))


        ss = suffstats(NormalKnownSigma(σ), x)
        @test isa(ss, Distributions.NormalKnownSigmaStats)
        @test ss.σ == σ
        @test ss.sx ≈ sum(x)
        @test ss.tw ≈ n0

        ss = suffstats(NormalKnownSigma(σ), x, w)
        @test isa(ss, Distributions.NormalKnownSigmaStats)
        @test ss.σ == σ
        @test ss.sx ≈ dot(x, w)
        @test ss.tw ≈ sum(w)

        d = fit_mle(Normal, x; sigma=σ)
        @test isa(d, Normal)
        @test d.σ == σ
        @test d.μ ≈ mean(x)

        d = fit_mle(Normal, x, w; sigma=σ)
        @test isa(d, Normal)
        @test d.σ == σ
        @test d.μ ≈ dot(x, w) / sum(w)
    end
end

@testset "Testing fit for Uniform" begin
    for rng in ((), (rng,)), D in (Uniform, Uniform{Float64}, Uniform{Float32})
        z = rand(rng..., Uniform(1.2, 5.8), n0)
        for x in (z, OffsetArray(z, -n0 ÷ 2))
            d = fit(D, x)
            @test d isa D
            @test 1.2 <= minimum(d) <= maximum(d) <= 5.8
            @test minimum(d) == partype(d)(minimum(z))
            @test maximum(d) == partype(d)(maximum(z))
        end

        z = rand(rng..., Uniform(1.2, 5.8), N)
        for x in (z, OffsetArray(z, -N ÷ 2))
            d = fit(D, x)
            @test d isa D
            @test 1.2 <= minimum(d) <= maximum(d) <= 5.8
            @test minimum(d) ≈ 1.2 atol=0.02
            @test maximum(d) ≈ 5.8 atol=0.02
        end
    end
end

@testset "Testing fit for Gamma" begin
    for func in funcs, dist in (Gamma, Gamma{Float64})
        x = func[2](dist(3.9, 2.1), n0)
        w = func[1](n0)

        ss = suffstats(dist, x)
        @test isa(ss, Distributions.GammaStats)
        @test ss.sx    ≈ sum(x)
        @test ss.slogx ≈ sum(log.(x))
        @test ss.tw    ≈ n0

        ss = suffstats(dist, x, w)
        @test isa(ss, Distributions.GammaStats)
        @test ss.sx    ≈ dot(x, w)
        @test ss.slogx ≈ dot(log.(x), w)
        @test ss.tw    ≈ sum(w)

        d = fit(dist, func[2](dist(3.9, 2.1), N))
        @test isa(d, dist)
        @test isapprox(shape(d), 3.9, atol=0.1)
        @test isapprox(scale(d), 2.1, atol=0.2)
    end
end

@testset "Testing fit for Geometric" begin
    for func in funcs, dist in (Geometric, Geometric{Float64})
        x = func[2](dist(0.3), n0)
        w = func[1](n0)

        ss = suffstats(dist, x)
        @test isa(ss, Distributions.GeometricStats)
        @test ss.sx ≈ sum(x)
        @test ss.tw ≈ n0

        ss = suffstats(dist, x, w)
        @test isa(ss, Distributions.GeometricStats)
        @test ss.sx ≈ dot(x, w)
        @test ss.tw ≈ sum(w)

        d = fit(dist, x)
        @test isa(d, dist)
        @test succprob(d) ≈ inv(1. + mean(x))

        d = fit(dist, x, w)
        @test isa(d, dist)
        @test succprob(d) ≈ inv(1. + dot(x, w) / sum(w))

        d = fit(dist, func[2](dist(0.3), N))
        @test isa(d, dist)
        @test isapprox(succprob(d), 0.3, atol=0.01)
    end
end

@testset "Testing fit for Laplace" begin
    for func in funcs, dist in (Laplace, Laplace{Float64})
        d = fit(dist, func[2](dist(5.0, 3.0), N + 1))
        @test isa(d, dist)
        @test isapprox(location(d), 5.0, atol=0.03)
        @test isapprox(scale(d)   , 3.0, atol=0.03)
    end
end

@testset "Testing fit for Pareto" begin
    for func in funcs, dist in (Pareto, Pareto{Float64})
        x = func[2](dist(3., 7.), N)
        d = fit(dist, x)

        @test isa(d, dist)
        @test isapprox(shape(d), 3., atol=0.1)
        @test isapprox(scale(d), 7., atol=0.1)
    end
end

@testset "Testing fit for Poisson" begin
    for func in funcs, dist in (Poisson, Poisson{Float64})
        x = func[2](dist(8.2), n0)
        w = func[1](n0)

        ss = suffstats(dist, x)
        @test isa(ss, Distributions.PoissonStats)
        @test ss.sx ≈ sum(x)
        @test ss.tw ≈ n0

        ss = suffstats(dist, x, w)
        @test isa(ss, Distributions.PoissonStats)
        @test ss.sx ≈ dot(x, w)
        @test ss.tw ≈ sum(w)

        d = fit(dist, x)
        @test isa(d, dist)
        @test mean(d) ≈ mean(x)

        d = fit(dist, x, w)
        @test isa(d, dist)
        @test mean(d) ≈ dot(Float64[xx for xx in x], w) / sum(w)

        d = fit(dist, func[2](dist(8.2), N))
        @test isa(d, dist)
        @test isapprox(mean(d), 8.2, atol=0.2)
    end
end

@testset "Testing fit for InverseGaussian" begin
    for func in funcs, dist in (InverseGaussian, InverseGaussian{Float64})
        x = rand(dist(3.9, 2.1), n0)
        w = func[1](n0)

        ss = suffstats(dist, x)
        @test isa(ss, Distributions.InverseGaussianStats)
        @test ss.sx    ≈ sum(x)
        @test ss.sinvx ≈ sum(1 ./ x)
        @test ss.sw    ≈ n0

        ss = suffstats(dist, x, w)
        @test isa(ss, Distributions.InverseGaussianStats)
        @test ss.sx    ≈ dot(x, w)
        @test ss.sinvx ≈ dot(1 ./ x, w)
        @test ss.sw    ≈ sum(w)

        d = fit(dist, rand(dist(3.9, 2.1), N))
        @test isa(d, dist)
        @test isapprox(mean(d), 3.9, atol=0.1)
        @test isapprox(shape(d), 2.1, atol=0.1)

        d = fit_mle(dist, rand(dist(3.9, 2.1), N))
        @test isapprox(mean(d), 3.9, atol=0.1)
        @test isapprox(shape(d), 2.1, atol=0.1)
    end
end

@testset "Testing fit for Rayleigh" begin
    for func in funcs, dist in (Rayleigh, Rayleigh{Float64})
        x = func[2](dist(3.6), N)
        d = fit(dist, x)

        @test isa(d, dist)
        @test isapprox(mode(d), 3.6, atol=0.1)

        # Test automatic differentiation
        f(x) = mean(fit(Rayleigh, x))
        @test all(ForwardDiff.gradient(f, x) .>= 0)
    end
end

@testset "Testing fit for Weibull" begin
    for func in funcs, dist in (Weibull, Weibull{Float64})
        d = fit(dist, func[2](dist(8.1, 4.3), N))
        @test isa(d, dist)
        @test isapprox(d.α, 8.1, atol = 0.1)
        @test isapprox(d.θ, 4.3, atol = 0.1)

    end
end
