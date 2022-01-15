# Testing censored distributions

module TestCensored

using Distributions, Test
using Distributions: Censored

function _as_mixture(d::Censored{D,Discrete}) where {D}
    d0 = d.uncensored
    dtrunc = if d0 isa DiscreteUniform
        truncated(
            d0,
            d.lower === missing ? -Inf : floor(d.lower) + 1,
            d.upper === missing ? Inf : ceil(d.upper) - 1,
        )
    else
        error("truncation to open interval not implemented for $d0")
    end
    prob_lower = d.lower === missing ? 0 : cdf(d0, d.lower)
    prob_upper = d.upper === missing ? 0 : ccdf(d0, d.upper) + pdf(d0, d.upper)
    prob_interval = 1 - (prob_lower + prob_upper)
    components = Distribution[dtrunc]
    probs = [prob_interval]
    if prob_lower > 0
        push!(components, Dirac(d.lower))
        push!(probs, prob_lower)
    end
    if prob_upper > 0
        push!(components, Dirac(d.upper))
        push!(probs, prob_upper)
    end
    return MixtureModel(map(identity, components), probs)
end

@testset "censored" begin
    d0 = Normal(0, 1)
    @test_throws ErrorException censored(d0, 1, -1)

    d = censored(d0, -1, 1.0)
    @test d isa Censored
    @test d.lower === -1.0
    @test d.upper === 1.0

    d = censored(d0, missing, -1)
    @test d isa Censored
    @test ismissing(d.lower)
    @test d.upper == -1

    d = censored(d0, 1, missing)
    @test d isa Censored
    @test ismissing(d.upper)
    @test d.lower == 1

    d = censored(d0, missing, missing)
    @test d === d0
end

@testset "Censored" begin
    @testset "basic" begin
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
        @test sprint(show, "text/plain", d) == "Censored($(Normal(0.0, 1.0)), range=(-1, 2))"

        d = Censored(Cauchy(0, 1), missing, 2)
        @test d isa Censored
        @test eltype(d) === Base.promote_type(eltype(Cauchy(0, 1)), Int)
        @test params(d) == (params(Cauchy(0, 1))..., 2)
        @test partype(d) === Float64
        @test extrema(d) == (-Inf, 2.0)
        @test @inferred !islowerbounded(d)
        @test @inferred isupperbounded(d)
        @test @inferred insupport(d, 0.1)
        @test insupport(d, -3)
        @test insupport(d, 2)
        @test !insupport(d, 2.1)
        @test sprint(show, "text/plain", d) == "Censored($(Cauchy(0.0, 1.0)), range=(missing, 2))"

        d = Censored(Gamma(1, 2), 2, missing)
        @test d isa Censored
        @test eltype(d) === Base.promote_type(eltype(Gamma(1, 2)), Int)
        @test params(d) == (params(Gamma(1, 2))..., 2)
        @test partype(d) === Float64
        @test extrema(d) == (2.0, Inf)
        @test @inferred islowerbounded(d)
        @test @inferred !isupperbounded(d)
        @test @inferred insupport(d, 2.1)
        @test insupport(d, 2.0)
        @test !insupport(d, 1.9)
        @test sprint(show, "text/plain", d) == "Censored($(Gamma(1, 2)), range=(2, missing))"

        d = Censored(Binomial(10, 0.2), -1.5, 9.5)
        @test extrema(d) === (0.0, 9.5)
        @test @inferred islowerbounded(d)
        @test @inferred isupperbounded(d)
        @test @inferred !insupport(d, -1.5)
        @test insupport(d, 0)
        @test insupport(d, 9.5)
        @test !insupport(d, 10)

        @test censored(Censored(Normal(), 1, missing), missing, 2) == Censored(Normal(), 1, 2)
        @test censored(Censored(Normal(), missing, 1), -1, missing) == Censored(Normal(), -1, 1)
        @test censored(Censored(Normal(), 1, 2), 1.5, 2.5) == Censored(Normal(), 1.5, 2.0)
        @test censored(Censored(Normal(), 1, 3), 1.5, 2.5) == Censored(Normal(), 1.5, 2.5)
        @test censored(Censored(Normal(), 1, 2), 0.5, 2.5) == Censored(Normal(), 1.0, 2.0)
        @test censored(Censored(Normal(), 1, 2), 0.5, 1.5) == Censored(Normal(), 1.0, 1.5)

        @test censored(Censored(Normal(), missing, 1), missing, 1) == Censored(Normal(), missing, 1)
        @test censored(Censored(Normal(), missing, 1), missing, 2) == Censored(Normal(), missing, 1)
        @test censored(Censored(Normal(), missing, 1), missing, 1.5) == Censored(Normal(), missing, 1)
        @test censored(Censored(Normal(), missing, 1.5), missing, 1) == Censored(Normal(), missing, 1)

        @test censored(Censored(Normal(), 1, missing), 1, missing) == Censored(Normal(), 1, missing)
        @test censored(Censored(Normal(), 1, missing), 2, missing) == Censored(Normal(), 2, missing)
        @test censored(Censored(Normal(), 1, missing), 1.5, missing) == Censored(Normal(), 1.5, missing)
        @test censored(Censored(Normal(), 1.5, missing), 1, missing) == Censored(Normal(), 1.5, missing)
    end

    @testset "DiscreteUniform" begin
        d0 = DiscreteUniform(0, 10)
        bounds = [(missing, 8), (2, missing), (2, 8), (3.5, missing)]
        @testset "lower = $lower, upper = $upper" for (lower, upper) in bounds
            d = censored(d0, lower, upper)
            dmix = _as_mixture(d)
            @test extrema(d) == extrema(dmix)
            l, u = extrema(d)
            @testset for f in [pdf, logpdf, cdf, logcdf, ccdf, logccdf]
                @test f(d, l) ≈ f(dmix, l)
                @test f(d, l - 0.1) ≈ f(dmix, l - 0.1)
                @test f(d, u) ≈ f(dmix, u)
                @test f(d, u + 0.1) ≈ f(dmix, u + 0.1)
                @test f(d, 5) ≈ f(dmix, 5)
            end
            @testset for f in [mean, var]
                @test f(d) ≈ f(dmix)
            end
            @test median(d) ≈ clamp(median(d0), l, u)
            @test quantile(d, 0:0.01:1) ≈ clamp.(quantile(d0, 0:0.01:1), l, u)
            xs = rand(d, 100)
            @test loglikelihood(d, xs) ≈ loglikelihood(dmix, xs)
        end
    end
end

end # module