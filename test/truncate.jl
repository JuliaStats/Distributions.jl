# Testing discrete univariate distributions

module TestTruncate

using Distributions
using ForwardDiff: Dual, ForwardDiff
using Random: MersenneTwister
using StatsFuns
import JSON
using Test
using ..Main: fdm

function verify_and_test_drive(jsonfile, selected, n_tsamples::Int,lower::Int,upper::Int)
    R = JSON.parsefile(jsonfile)
    for dct in R
        ex = dct["expr"]
        dsym = Symbol(dct["dtype"])
        if dsym in [:Skellam, :NormalInverseGaussian]
            continue
        end

        dname = string(dsym)

        dsymt = Symbol("truncated($(dct["dtype"]),$lower,$upper")
        dnamet = string(dsym)

        # test whether it is included in the selected list
        # or all are selected (when selected is empty)
        if !isempty(selected) && !(dname in selected)
            continue
        end

        # perform testing
        d0 = eval(Meta.parse(ex))
        if minimum(d0) > lower || maximum(d0) < upper
            continue
        end

        println("    testing truncated(", ex, ", ", lower, ", ", upper, ")")
        d = truncated(d0,lower,upper)
        if d0 isa Uniform
            @test d isa Uniform
            @test minimum(d) == lower
            @test maximum(d) == upper
        elseif d0 isa DiscreteUniform
            @test d isa DiscreteUniform
            @test minimum(d) == lower
            @test maximum(d) == upper
        else
            @test d isa Truncated
            verify_and_test(d, dct, n_tsamples)
        end
    end
end


_parse_x(d::DiscreteUnivariateDistribution, x) = round(Int, x)
_parse_x(d::ContinuousUnivariateDistribution, x) = Float64(x)

_json_value(x::Number) = x

_json_value(x::AbstractString) =
    x == "inf" ? Inf :
    x == "-inf" ? -Inf :
    x == "nan" ? NaN :
    error("Invalid numerical value: $x")

function verify_and_test(d::UnivariateDistribution, dct::AbstractDict, n_tsamples::Int)
    # verify stats
    @test minimum(d) ≈ max(_json_value(dct["minimum"]),d.lower)
    @test maximum(d) ≈ min(_json_value(dct["maximum"]),d.upper)
    @test extrema(d) == (minimum(d), maximum(d))
    @test d == deepcopy(d)

    # verify logpdf and cdf at certain points
    pts = dct["points"]
    for pt in pts
        x = _parse_x(d, pt["x"])
        lp = d.lower <= x <= d.upper ? Float64(pt["logpdf"]) - d.logtp : -Inf
        cf = x < d.lower ? 0.0 : x >= d.upper ? 1.0 : (Float64(pt["cdf"]) - d.lcdf)/d.tp
        if !isa(d, Distributions.Truncated{Distributions.StudentizedRange{Float64},Distributions.Continuous})
            @test logpdf(d, x) ≈ lp atol=sqrt(eps())
        end
        @test cdf(d, x) ≈ cf atol=sqrt(eps())
        # NOTE: some distributions use pdf() in StatsFuns.jl which have no generic support yet
        if !any(T -> d isa T, [Distributions.Truncated{Distributions.NoncentralChisq{Float64},Distributions.Continuous, Float64},
                           Distributions.Truncated{Distributions.NoncentralF{Float64},Distributions.Continuous, Float64},
                           Distributions.Truncated{Distributions.NoncentralT{Float64},Distributions.Continuous, Float64},
                           Distributions.Truncated{Distributions.StudentizedRange{Float64},Distributions.Continuous, Float64},
                           Distributions.Truncated{Distributions.Rician{Float64},Distributions.Continuous, Float64},
                           Distributions.Truncated{Distributions.Hypergeometric,Distributions.Discrete, Float64}])
            @test isapprox(logpdf(d, Dual(float(x))), lp, atol=sqrt(eps()))
        end
        # NOTE: this test is disabled as StatsFuns.jl doesn't have generic support for cdf()
        # @test isapprox(cdf(d, Dual(x))   , cf, atol=sqrt(eps()))
    end

    # test if truncated quantile function can be evaluated consistently for different types at certain points
    @test isapprox(quantile(d, 0), quantile(d, Float32(0)))
    @test isapprox(quantile(d, 1), quantile(d, Float32(1.0)))
    @test isapprox(quantile(d, Float64(0.3)), quantile(d, Float32(0.3)))
    @test isapprox(quantile(d, Float64(0.7)), quantile(d, Float32(0.7)))

    try
        m = mgf(d,0.0)
        @test m == 1.0
        m = mgf(d,Dual(0.0))
        @test m == 1.0
    catch e
        isa(e, MethodError) || throw(e)
    end
    try
        c = cf(d,0.0)
        @test c == 1.0
        c = cf(d,Dual(0.0))
        @test c == 1.0
        # test some extra values: should all be well-defined
        for t in (0.1,-0.1,1.0,-1.0)
            @test !isnan(cf(d,t))
            @test !isnan(cf(d,Dual(t)))
        end
    catch e
        isa(e, MethodError) || throw(e)
    end

    # generic testing
    if isa(d, Cosine)
        n_tsamples = floor(Int, n_tsamples / 10)
    end
end

# default methods
for (μ, lower, upper) in [(0, -1, 1), (1, 2, 4)]
    d = truncated(Normal(μ, 1), lower, upper)
    @test d.untruncated === Normal(μ, 1)
    @test d.lower == lower
    @test d.upper == upper
    @test truncated(Normal(μ, 1); lower=lower, upper=upper) === d
end
for bound in (-2, 1)
    d = @test_deprecated Distributions.Truncated(Normal(), Float64(bound), Inf)
    @test truncated(Normal(); lower=bound, upper=Inf) == d

    d_nothing = truncated(Normal(); lower=bound)
    @test truncated(Normal(); lower=bound, upper=nothing) == d_nothing
    @test extrema(d_nothing) == promote(bound, Inf)

    d = @test_deprecated Distributions.Truncated(Normal(), -Inf, Float64(bound))
    @test truncated(Normal(); lower=-Inf, upper=bound) == d

    d_nothing = truncated(Normal(); upper=bound)
    @test truncated(Normal(); lower=nothing, upper=bound) == d_nothing
    @test extrema(d_nothing) == promote(-Inf, bound)
end
@test truncated(Normal()) === Normal()

## main

for c in ["discrete",
          "continuous"]

    title = string(uppercase(c[1]), c[2:end])
    println("    [$title]")
    println("    ------------")
    jsonfile = joinpath(@__DIR__, "ref", "$(c)_test.ref.json")
    verify_and_test_drive(jsonfile, ARGS, 10^6,3,5)
    println()
end

## automatic differentiation

f = x -> logpdf(truncated(Normal(x[1], x[2]), x[3], x[4]), mean(x))
at = [0.0, 1.0, 0.0, 1.0]
@test isapprox(ForwardDiff.gradient(f, at), fdm(f, at), atol=1e-6)

    @testset "errors" begin
        @test_throws ErrorException truncated(Normal(), 1, 0)
        @test_throws DomainError truncated(Uniform(), 1, 2)
        @test_throws ErrorException truncated(Exponential(), 3, 1)
    end

    @testset "#1328" begin
        dist = Poisson(2.0)
        dist_zeroinflated = MixtureModel([Dirac(0.0), dist], [0.4, 0.6])
        dist_zerotruncated = truncated(dist; lower=1)
        dist_zeromodified = MixtureModel([Dirac(0.0), dist_zerotruncated], [0.4, 0.6])

        @test logsumexp(logpdf(dist, x) for x in 0:1000) ≈ 0 atol=1e-15
        @test logsumexp(logpdf(dist_zeroinflated, x) for x in 0:1000) ≈ 0 atol=1e-15
        @test logsumexp(logpdf(dist_zerotruncated, x) for x in 0:1000) ≈ 0 atol=1e-15
        @test logsumexp(logpdf(dist_zeromodified, x) for x in 0:1000) ≈ 0 atol=1e-15
    end
end

@testset "show" begin
    @test sprint(show, "text/plain", truncated(Normal(); lower=2.0)) == "Truncated($(Normal()); lower=2.0)"
    @test sprint(show, "text/plain", truncated(Normal(); upper=3.0)) == "Truncated($(Normal()); upper=3.0)"
    @test sprint(show, "text/plain", truncated(Normal(), 2.0, 3.0)) == "Truncated($(Normal()); lower=2.0, upper=3.0)"
end

@testset "sampling with small mass (#1548)" begin
    d = truncated(Beta(10, 100); lower=0.5)
    x = rand(d, 10_000)
    @test mean(x) ≈ 0.5 atol=0.05
end

@testset "quantile examples (#1726)" begin
  d_narrow = truncated(Normal(0, 0.01), -1, 1.2)

  # prior to patch #1727 this was offering either ±Inf
  @test quantile(d_narrow, 1) ≤ d_narrow.upper && quantile(d_narrow, 1) ≈ d_narrow.upper
  @test quantile(d_narrow, 0) ≥ d_narrow.lower && quantile(d_narrow, 0) ≈ d_narrow.lower

  @test isa(quantile(d_narrow, ForwardDiff.Dual(0., 0.)), ForwardDiff.Dual)

  d = truncated(Normal(0, 2), 0, 1)

  # prior to patch #1727 this was offering 1.0000000000000002 > 1.
  @test quantile(d, 1) ≤ d.upper && quantile(d, 1) ≈ d.upper

  @test isa(quantile(d, ForwardDiff.Dual(1.,0.)), ForwardDiff.Dual)
end

@testset "cdf outside of [0, 1] (#1854)" begin
    dist = truncated(Normal(2.5, 0.2); lower=0.0)
    @test @inferred(cdf(dist, 3.741058503233821e-17)) === 0.0
    @test @inferred(ccdf(dist, 3.741058503233821e-17)) === 1.0
    @test @inferred(cdf(dist, 1.4354474178676617e-18)) === 0.0
    @test @inferred(ccdf(dist, 1.4354474178676617e-18)) === 1.0
    @test @inferred(cdf(dist, 8.834854780587132e-18)) === 0.0
    @test @inferred(ccdf(dist, 8.834854780587132e-18)) === 1.0

    μ, σ, lower, upper = 2.122039143928797, 0.07327367710864985, 1.9521656132878236, 2.8274429454898398
    dist = truncated(Normal(μ, σ); lower = lower, upper = upper)
    @test @inferred(cdf(dist, 2.82)) === 1.0
    # P(X > 2.82 | lower ≤ X ≤ upper) is about 4.6e-22, not zero: the upper tail is evaluated from
    # P(X > x), which does not round to zero the way 1 - P(X ≤ x) does
    ccdf_ref = setprecision(BigFloat, 256) do
        Q(x) = erfc((big(x) - big(μ)) / (big(σ) * sqrt(big(2)))) / 2
        (Q(2.82) - Q(upper)) / (Q(lower) - Q(upper))
    end
    @test @inferred(ccdf(dist, 2.82)) ≈ Float64(ccdf_ref) rtol=1e-12
end

@testset "distribution functions, quantiles and samples in deep tails" begin
    # Far in a tail, P(X ≤ bound) rounds to one or underflows, so cdf, ccdf, their logarithms,
    # quantile and rand have to work from log probabilities and, above the median, from the upper
    # tail P(X > x) of the untruncated distribution. The references are the closed forms of the
    # tails in BigFloat.
    setprecision(BigFloat, 2048) do
        Q(z) = erfc(z / sqrt(big(2))) / 2                    # P(Z > z) for a standard normal Z
        # P(X > x) for X ~ d0
        S(d0::Normal, x) = Q((big(x) - big(d0.μ)) / big(d0.σ))
        S(d0::LogNormal, x) = Q((log(big(x)) - big(d0.μ)) / big(d0.σ))
        S(d0::Gamma, x) = (1 + big(x)) * exp(-big(x))        # shape 2, scale 1
        S(d0::Poisson, k) = 1 - exp(-big(d0.λ)) * sum(big(d0.λ)^i / factorial(big(i)) for i in 0:Int(k))
        # P(X ≥ x) for X ~ d0
        Sincl(d0::DiscreteUnivariateDistribution, k) = S(d0, k - 1)
        Sincl(d0, x) = S(d0, x)
        # the mass of the untruncated distribution from `lower` on, and above `upper`
        mass_from(d0, ::Nothing) = big(1)
        mass_from(d0, lower) = Sincl(d0, lower)
        mass_above(d0, ::Nothing) = big(0)
        mass_above(d0, upper) = S(d0, upper)

        rng = MersenneTwister(2024)
        for (d0, lower, upper, xs, rtol) in (
            (Normal(), 8.0, 9.0, (8.1, 8.5, 8.9), 1e-13),                     # issue #2089
            (Normal(), 40.0, nothing, (40.01, 40.1, 40.5), 1e-11),
            (Normal(), 40.0, 41.0, (40.01, 40.5, 40.99), 1e-11),
            (Normal(), nothing, -40.0, (-40.5, -40.1, -40.01), 1e-11),
            (Normal(), -41.0, -40.0, (-40.99, -40.5, -40.01), 1e-11),
            (Normal(), -1.0, 10.0, (0.0, 5.0, 9.0), 1e-12),                   # only the upper bound in a tail
            (LogNormal(-40.0, 1.0), 1.0, nothing, (1.01, 1.1), 1e-11),
            (LogNormal(-3.0, 0.2), 1.0, nothing, (1.01, 1.05), 1e-12),
            (LogNormal(-8.0, 1.0), 1.0, exp(1.0), (1.1, 2.0), 1e-13),
            (Gamma(2.0, 1.0), 200.0, nothing, (200.5, 201.0, 205.0), 1e-11),
            (Poisson(2.0), 100, nothing, (100, 101, 103), 1e-10),
        )
            d = truncated(d0, lower, upper)
            tp = mass_from(d0, lower) - mass_above(d0, upper)
            @test d.logtp ≈ Float64(log(tp)) rtol=rtol
            for x in xs
                cdf_ref = (mass_from(d0, lower) - S(d0, x)) / tp
                ccdf_ref = (S(d0, x) - mass_above(d0, upper)) / tp
                @test cdf(d, x) ≈ Float64(cdf_ref) rtol=rtol
                @test ccdf(d, x) ≈ Float64(ccdf_ref) rtol=rtol
                @test cdf(d, x) + ccdf(d, x) ≈ 1 rtol=rtol
                @test logcdf(d, x) ≈ Float64(log(cdf_ref)) atol=rtol
                @test logccdf(d, x) ≈ Float64(log(ccdf_ref)) atol=rtol
                @test pdf(d, x) > 0
                @test pdf(d, x) ≈ exp(logpdf(d, x)) rtol=1e-14
                # the quantiles invert the distribution functions where the probability is not
                # within rounding of 0 or 1
                p = cdf(d, x)
                if d0 isa ContinuousUnivariateDistribution && 1e-10 < p < 1 - 1e-10
                    @test quantile(d, p) ≈ x rtol=1e-8 atol=1e-8
                    @test cquantile(d, ccdf(d, x)) ≈ x rtol=1e-8 atol=1e-8
                    @test invlogcdf(d, logcdf(d, x)) ≈ x rtol=1e-8 atol=1e-8
                    @test invlogccdf(d, logccdf(d, x)) ≈ x rtol=1e-8 atol=1e-8
                end
            end

            @test minimum(d) <= quantile(d, 0) ≈ minimum(d)
            if d0 isa ContinuousUnivariateDistribution || isfinite(maximum(d))
                # (an integer quantile cannot represent an infinite maximum)
                @test maximum(d) >= quantile(d, 1) ≈ maximum(d)
            end
            m = median(d)
            @test minimum(d) <= m <= maximum(d)
            if d0 isa ContinuousUnivariateDistribution
                @test cdf(d, m) ≈ 0.5 rtol=1e-8
                for p in (0.01, 0.3, 0.9, 0.999)
                    @test cdf(d, quantile(d, p)) ≈ p rtol=1e-8
                end
            else
                @test cdf(d, m) >= 0.5 && cdf(d, m - 1) < 0.5
            end

            x = rand(rng, d, 10_000)
            @test all(isfinite, x)
            @test all(xi -> insupport(d, xi), x)
            if d0 isa ContinuousUnivariateDistribution
                for p in (0.1, 0.5, 0.9)
                    @test mean(x .<= quantile(d, p)) ≈ p atol=0.03
                end
            else
                @test mean(x .== minimum(d)) ≈ pdf(d, minimum(d)) atol=0.03
            end
        end
    end
end

@testset "mass of a truncation in the upper tail" begin
    # above the median, 1 - P(X < lower) rounds to zero in the tail, so P(X ≥ lower) must be
    # computed directly
    setprecision(BigFloat, 256) do
        Q(z) = erfc(big(z) / sqrt(big(2))) / 2
        for (d0, l, u, x, logtp) in (
            (Normal(), 40.0, nothing, 40.5, log(Q(40))),
            (LogNormal(-40.0, 1.0), 1.0, nothing, 1.5, log(Q(40))),
            (Normal(), 40.0, 41.0, 40.5, log(Q(40) - Q(41))),
            (Poisson(2.0), 40, nothing, 41, log(sum(exp(-big(2)) * big(2)^k / factorial(big(k)) for k in 40:400))),
        )
            d = truncated(d0, l, u)
            @test d.logtp ≈ logtp rtol=1e-14
            @test logpdf(d, x) ≈ logpdf(d0, x) - logtp rtol=1e-14
        end
    end
end
