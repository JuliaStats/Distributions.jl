# Testing censored distributions

module TestCensored

using Distributions, Test
using Distributions: Censored

function _as_mixture(d::Censored)
    d0 = d.uncensored
    dtrunc = if d0 isa DiscreteUniform || d0 isa Poisson
        truncated(
            d0,
            d.lower === nothing ? -Inf : floor(d.lower) + 1,
            d.upper === nothing ? Inf : ceil(d.upper) - 1,
        )
    elseif d0 isa ContinuousDistribution
        truncated(
            d0,
            d.lower === nothing ? -Inf : nextfloat(float(d.lower)),
            d.upper === nothing ? Inf : prevfloat(float(d.upper)),
        )
    else
        error("truncation to open interval not implemented for $d0")
    end
    prob_lower = d.lower === nothing ? 0 : cdf(d0, d.lower)
    prob_upper = if d.upper === nothing
        0
    elseif d0 isa ContinuousDistribution
        ccdf(d0, d.upper)
    else
        ccdf(d0, d.upper) + pdf(d0, d.upper)
    end
    prob_interval = 1 - (prob_lower + prob_upper)
    components = Distribution[dtrunc]
    probs = [prob_interval]
    if prob_lower > 0
        # workaround for MixtureModel currently not supporting mixtures of discrete and
        # continuous components
        push!(components, d0 isa DiscreteDistribution ? Dirac(d.lower) : Normal(d.lower, 0))
        push!(probs, prob_lower)
    end
    if prob_upper > 0
        push!(components, d0 isa DiscreteDistribution ? Dirac(d.upper) : Normal(d.upper, 0))
        push!(probs, prob_upper)
    end
    return MixtureModel(map(identity, components), probs)
end

@testset "censored" begin
    d0 = Normal(0, 1)
    @test_throws ArgumentError censored(d0, 1, -1)

    # bound argument constructors
    d = censored(d0, -1, 1.0)
    @test d isa Censored
    @test d.lower === -1.0
    @test d.upper === 1.0

    d = censored(d0, nothing, -1)
    @test d isa Censored
    @test d.lower === nothing
    @test d.upper == -1

    d = censored(d0, 1, nothing)
    @test d isa Censored
    @test d.upper === nothing
    @test d.lower == 1

    d = censored(d0, nothing, nothing)
    @test d === d0

    # bound keyword constructors
    d = censored(d0; lower=-2, upper=1.5)
    @test d isa Censored
    @test d.lower === -2.0
    @test d.upper === 1.5

    d = censored(d0; upper=true)
    @test d isa Censored
    @test d.lower === nothing
    @test d.upper === true

    d = censored(d0; lower=-3)
    @test d isa Censored
    @test d.upper === nothing
    @test d.lower === -3

    d = censored(d0)
    @test d === d0
end

@testset "Censored" begin
    @testset "basic" begin
        # check_args
        @test_throws ArgumentError Censored(Normal(0, 1), 2, 1)
        @test_throws ArgumentError Censored(Normal(0, 1), 2, 1; check_args=true)
        Censored(Normal(0, 1), 2, 1; check_args=false)
        Censored(Normal(0, 1), nothing, 1; check_args=true)
        Censored(Normal(0, 1), 2, nothing; check_args=true)

        d = Censored(Normal(0.0, 1.0), -1, 2)
        @test d isa Censored
        @test eltype(d) === Float64
        @test params(d) === (params(Normal(0.0, 1.0))..., -1, 2)
        @test partype(d) === Float64
        @test @inferred extrema(d) == (-1, 2)
        @test @inferred islowerbounded(d)
        @test @inferred isupperbounded(d)
        @test @inferred insupport(d, 0.1)
        @test insupport(d, -1)
        @test insupport(d, 2)
        @test !insupport(d, -1.1)
        @test !insupport(d, 2.1)
        @test sprint(show, "text/plain", d) == "Censored($(Normal(0.0, 1.0)); lower=-1, upper=2)"

        d = Censored(Cauchy(0, 1), nothing, 2)
        @test d isa Censored
        @test eltype(d) === Base.promote_type(eltype(Cauchy(0, 1)), Int)
        @test params(d) === (params(Cauchy(0, 1))..., nothing, 2)
        @test partype(d) === Float64
        @test extrema(d) == (-Inf, 2.0)
        @test @inferred !islowerbounded(d)
        @test @inferred isupperbounded(d)
        @test @inferred insupport(d, 0.1)
        @test insupport(d, -3)
        @test insupport(d, 2)
        @test !insupport(d, 2.1)
        @test sprint(show, "text/plain", d) == "Censored($(Cauchy(0.0, 1.0)); upper=2)"

        d = Censored(Gamma(1, 2), 2, nothing)
        @test d isa Censored
        @test eltype(d) === Base.promote_type(eltype(Gamma(1, 2)), Int)
        @test params(d) === (params(Gamma(1, 2))..., 2, nothing)
        @test partype(d) === Float64
        @test extrema(d) == (2.0, Inf)
        @test @inferred islowerbounded(d)
        @test @inferred !isupperbounded(d)
        @test @inferred insupport(d, 2.1)
        @test insupport(d, 2.0)
        @test !insupport(d, 1.9)
        @test sprint(show, "text/plain", d) == "Censored($(Gamma(1, 2)); lower=2)"

        d = Censored(Binomial(10, 0.2), -1.5, 9.5)
        @test extrema(d) === (0.0, 9.5)
        @test @inferred islowerbounded(d)
        @test @inferred isupperbounded(d)
        @test @inferred !insupport(d, -1.5)
        @test insupport(d, 0)
        @test insupport(d, 9.5)
        @test !insupport(d, 10)

        @test censored(Censored(Normal(), 1, nothing), nothing, 2) == Censored(Normal(), 1, 2)
        @test censored(Censored(Normal(), nothing, 1), -1, nothing) == Censored(Normal(), -1, 1)
        @test censored(Censored(Normal(), 1, 2), 1.5, 2.5) == Censored(Normal(), 1.5, 2.0)
        @test censored(Censored(Normal(), 1, 3), 1.5, 2.5) == Censored(Normal(), 1.5, 2.5)
        @test censored(Censored(Normal(), 1, 2), 0.5, 2.5) == Censored(Normal(), 1.0, 2.0)
        @test censored(Censored(Normal(), 1, 2), 0.5, 1.5) == Censored(Normal(), 1.0, 1.5)

        @test censored(Censored(Normal(), nothing, 1), nothing, 1) == Censored(Normal(), nothing, 1)
        @test censored(Censored(Normal(), nothing, 1), nothing, 2) == Censored(Normal(), nothing, 1)
        @test censored(Censored(Normal(), nothing, 1), nothing, 1.5) == Censored(Normal(), nothing, 1)
        @test censored(Censored(Normal(), nothing, 1.5), nothing, 1) == Censored(Normal(), nothing, 1)

        @test censored(Censored(Normal(), 1, nothing), 1, nothing) == Censored(Normal(), 1, nothing)
        @test censored(Censored(Normal(), 1, nothing), 2, nothing) == Censored(Normal(), 2, nothing)
        @test censored(Censored(Normal(), 1, nothing), 1.5, nothing) == Censored(Normal(), 1.5, nothing)
        @test censored(Censored(Normal(), 1.5, nothing), 1, nothing) == Censored(Normal(), 1.5, nothing)
    end

    @testset "Uniform" begin
        d0 = Uniform(0, 10)
        bounds = [
            (nothing, 8),
            (-Inf, 8),
            (nothing, Inf),
            (2, nothing),
            (2, Inf),
            (-Inf, nothing),
            (2, 8),
            (3.5, nothing),
            (3.5, Inf),
            (-Inf, Inf),
        ]
        @testset "lower = $(lower === nothing ? "nothing" : lower), upper = $(upper === nothing ? "nothing" : upper)" for (lower, upper) in bounds
            d = censored(d0, lower, upper)
            dmix = _as_mixture(d)
            l, u = extrema(d)
            if lower === nothing || !isfinite(lower)
                @test l == minimum(d0)
            else
                @test l == lower
            end
            if upper === nothing || !isfinite(upper)
                @test u == maximum(d0)
            else
                @test u == upper
            end
            @testset for f in [cdf, logcdf, ccdf, logccdf]
                @test @inferred(f(d, l)) ≈ f(dmix, l) atol=1e-8
                @test @inferred(f(d, l - 0.1)) ≈ f(dmix, l - 0.1) atol=1e-8
                @test @inferred(f(d, u)) ≈ f(dmix, u) atol=1e-8
                @test @inferred(f(d, u + 0.1)) ≈ f(dmix, u + 0.1) atol=1e-8
                @test @inferred(f(d, 5)) ≈ f(dmix, 5)
            end
            @testset for f in [mean, var]
                @test @inferred(f(d)) ≈ f(dmix)
            end
            @test @inferred(median(d)) ≈ clamp(median(d0), l, u)
            @inferred quantile(d, 0.5)
            @test Base.Fix1(quantile, d).(0:0.01:1) ≈ clamp.(Base.Fix1(quantile, d0).(0:0.01:1), l, u)
            # special-case pdf/logpdf/loglikelihood since when replacing Dirac(μ) with
            # Normal(μ, 0), they are infinite
            if lower === nothing || !isfinite(lower)
                @test @inferred(pdf(d, l)) ≈ pdf(d0, l)
                @test @inferred(logpdf(d, l)) ≈ logpdf(d0, l)
            else
                @test @inferred(pdf(d, l)) ≈ cdf(d0, l)
                @test @inferred(logpdf(d, l)) ≈ logcdf(d0, l)
            end
            if upper === nothing || !isfinite(upper)
                @test @inferred(pdf(d, u)) ≈ pdf(d0, u)
                @test @inferred(logpdf(d, u)) ≈ logpdf(d0, u)
            else
                @test @inferred(pdf(d, u)) ≈ ccdf(d0, u)
                @test @inferred(logpdf(d, u)) ≈ logccdf(d0, u)
            end
            # rand
            x = rand(d, 10_000)
            @test all(x -> insupport(d, x), x)
            # loglikelihood
            @test @inferred(loglikelihood(d, x)) ≈ sum(x -> logpdf(d, x), x)
            @test loglikelihood(d, [x; -1]) == -Inf
            # entropy
            @test @inferred(entropy(d)) ≈ mean(x -> -logpdf(d, x), x) atol = 1e-1
        end
    end

    @testset "Normal" begin
        d0 = Normal()
        bounds = [(nothing, 0.2), (-0.1, nothing), (-0.1, 0.2)]
        @testset "lower = $(lower === nothing ? "nothing" : lower), upper = $(upper === nothing ? "nothing" : upper)" for (lower, upper) in bounds
            d = censored(d0, lower, upper)
            dmix = _as_mixture(d)
            l, u = extrema(d)
            @testset for f in [cdf, logcdf, ccdf, logccdf]
                @test f(d, l) ≈ f(dmix, l) atol=1e-8
                @test f(d, l - 0.1) ≈ f(dmix, l - 0.1) atol=1e-8
                @test f(d, u) ≈ f(dmix, u) atol=1e-8
                @test f(d, u + 0.1) ≈ f(dmix, u + 0.1) atol=1e-8
                @test f(d, 5) ≈ f(dmix, 5)
            end
            @testset for f in [mean, var]
                @test f(d) ≈ f(dmix)
            end
            @test median(d) ≈ clamp(median(d0), l, u)
            @test Base.Fix1(quantile, d).(0:0.01:1) ≈ clamp.(Base.Fix1(quantile, d0).(0:0.01:1), l, u)
            # special-case pdf/logpdf/loglikelihood since when replacing Dirac(μ) with
            # Normal(μ, 0), they are infinite
            if lower === nothing
                @test pdf(d, l) ≈ pdf(d0, l)
                @test logpdf(d, l) ≈ logpdf(d0, l)
            else
                @test pdf(d, l) ≈ cdf(d0, l)
                @test logpdf(d, l) ≈ logcdf(d0, l)
            end
            if upper === nothing
                @test pdf(d, u) ≈ pdf(d0, u)
                @test logpdf(d, u) ≈ logpdf(d0, u)
            else
                @test pdf(d, u) ≈ ccdf(d0, u)
                @test logpdf(d, u) ≈ logccdf(d0, u)
            end
            # rand
            x = rand(d, 10_000)
            @test all(x -> insupport(d, x), x)
            # loglikelihood
            @test loglikelihood(d, x) ≈ sum(x -> logpdf(d, x), x)
            # entropy
            @test entropy(d) ≈ mean(x -> -logpdf(d, x), x) atol = 1e-1
        end
    end

    @testset "DiscreteUniform" begin
        d0 = DiscreteUniform(0, 10)
        bounds = [
            (nothing, 8),
            (-Inf, 8),
            (nothing, Inf),
            (2, nothing),
            (2, Inf),
            (-Inf, nothing),
            (2, 8),
            (3.5, nothing),
            (3.5, Inf),
            (-Inf, Inf),
        ]
        @testset "lower = $(lower === nothing ? "nothing" : lower), upper = $(upper === nothing ? "nothing" : upper)" for (lower, upper) in bounds
            d = censored(d0, lower, upper)
            dmix = _as_mixture(d)
            @test extrema(d) == extrema(dmix)
            l, u = extrema(d)
            @testset for f in [pdf, logpdf, cdf, logcdf, ccdf, logccdf]
                @test @inferred(f(d, l)) ≈ f(dmix, l) atol=1e-8
                @test @inferred(f(d, l - 0.1)) ≈ f(dmix, l - 0.1) atol=1e-8
                @test @inferred(f(d, u)) ≈ f(dmix, u) atol=1e-8
                @test @inferred(f(d, u + 0.1)) ≈ f(dmix, u + 0.1) atol=1e-8
                @test @inferred(f(d, 5)) ≈ f(dmix, 5)
            end
            @testset for f in [mean, var]
                @test @inferred(f(d)) ≈ f(dmix)
            end
            @test @inferred(median(d)) ≈ clamp(median(d0), l, u)
            @inferred quantile(d, 0.5)
            @test Base.Fix1(quantile, d).(0:0.01:1) ≈ clamp.(Base.Fix1(quantile, d0).(0:0.01:1), l, u)
            # rand
            x = rand(d, 10_000)
            @test all(x -> insupport(d, x), x)
            # loglikelihood
            @test @inferred(loglikelihood(d, x)) ≈ loglikelihood(dmix, x)
            # mean, std
            μ = @inferred mean(d)
            xall = unique(x)
            @test μ ≈ sum(x -> pdf(d, x) * x, xall)
            @test mean(x) ≈ μ atol = 1e-1
            v = @inferred var(d)
            @test v ≈ sum(x -> pdf(d, x) * abs2(x - μ), xall)
            @test std(x) ≈ sqrt(v) atol = 1e-1
            # entropy
            @test @inferred(entropy(d)) ≈ sum(x -> pdf(d, x) * -logpdf(d, x), xall)
        end
    end

    @testset "Poisson" begin
        d0 = Poisson(20)
        bounds = [(nothing, 12), (2, nothing), (2, 12), (8, nothing)]
        @testset "lower = $(lower === nothing ? "nothing" : lower), upper = $(upper === nothing ? "nothing" : upper)" for (lower, upper) in bounds
            d = censored(d0, lower, upper)
            dmix = _as_mixture(d)
            @test extrema(d) == extrema(dmix)
            l, u = extrema(d)
            @testset for f in [pdf, logpdf, cdf, logcdf, ccdf, logccdf]
                @test f(d, l) ≈ f(dmix, l) atol=1e-8
                @test f(d, l - 0.1) ≈ f(dmix, l - 0.1) atol=1e-8
                @test f(d, u) ≈ f(dmix, u) atol=1e-8
                @test f(d, u + 0.1) ≈ f(dmix, u + 0.1) atol=1e-8
                @test f(d, 5) ≈ f(dmix, 5)
            end
            @test median(d) ≈ clamp(median(d0), l, u)
            @test Base.Fix1(quantile, d).(0:0.01:0.99) ≈ clamp.(Base.Fix1(quantile, d0).(0:0.01:0.99), l, u)
            x = rand(d, 100)
            @test loglikelihood(d, x) ≈ loglikelihood(dmix, x)
            # rand
            x = rand(d, 10_000)
            @test all(x -> insupport(d, x), x)
            # mean, std
            @test mean(x) ≈ mean(x) atol = 1e-1
            @test std(x) ≈ std(x) atol = 1e-1            
        end
    end

    @testset "mixed types are still type-inferrible" begin
        bounds = [(nothing, 8), (2, nothing), (2, 8)]
        @testset "lower = $(lower === nothing ? "nothing" : lower), upper = $(upper === nothing ? "nothing" : upper), uncensored partype=$T0, partype=$T" for (lower, upper) in bounds,
                T in (Int, Float32, Float64), T0 in (Int, Float32, Float64)
            d0 = Uniform(T0(0), T0(10))
            d = censored(d0, lower === nothing ? nothing : T(lower), upper === nothing ? nothing : T(upper))
            l, u = extrema(d)
            @testset for f in [pdf, logpdf, cdf, logcdf, ccdf, logccdf]
                @inferred f(d, 3)
                @inferred f(d, 4f0)
                @inferred f(d, 5.0)
            end
            @testset for f in [median, mean, var, entropy]
                @inferred f(d)
            end
            @inferred quantile(d, 0.3f0)
            @inferred quantile(d, 0.5)
            x = randn(Float32, 100)
            @inferred loglikelihood(d, x)
            x = randn(100)
            @inferred loglikelihood(d, x)
        end
    end
end

end # module