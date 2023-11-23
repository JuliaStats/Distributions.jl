# Testing discrete univariate distributions

module TestTruncate

using Distributions
using ForwardDiff: Dual, ForwardDiff
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
        dtype = eval(dsym)
        dtypet = Truncated
        d0 = eval(Meta.parse(ex))
        if minimum(d0) > lower || maximum(d0) < upper
            continue
        end

        println("    testing truncated($(ex),$lower,$upper)")
        d = truncated(eval(Meta.parse(ex)),lower,upper)
        if dtype != Uniform && dtype != DiscreteUniform && dtype != TruncatedNormal # Uniform is truncated to Uniform
            @assert isa(dtype, Type) && dtype <: UnivariateDistribution
            @test isa(d, dtypet)
            # verification and testing
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

function verify_and_test(d::UnivariateDistribution, dct::Dict, n_tsamples::Int)
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
                           Distributions.Truncated{Distributions.Rician{Float64},Distributions.Continuous, Float64}])
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
        @test_throws ArgumentError truncated(Uniform(), 1, 2)
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
