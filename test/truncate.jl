# Testing discrete univariate distributions

module TestTruncate

using Distributions
using ForwardDiff: Dual
import JSON
using Test


function verify_and_test_drive(jsonfile, selected, n_tsamples::Int,lower::Int,upper::Int)
    R = JSON.parsefile(jsonfile)
    for dct in R
        ex = dct["expr"]
        dsym = Symbol(dct["dtype"])
        if dsym in [:Skellam, :NormalInverseGaussian]
            continue
        end

        dname = string(dsym)

        dsymt = Symbol("Truncated($(dct["dtype"]),$lower,$upper")
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

        println("    testing Truncated($(ex),$lower,$upper)")
        d = Truncated(eval(Meta.parse(ex)),lower,upper)
        if dtype != Uniform # Uniform is truncated to Uniform
            if dtype != TruncatedNormal
                @assert isa(dtype, Type) && dtype <: UnivariateDistribution
                @test isa(d, dtypet)
            end
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

    # verify logpdf and cdf at certain points
    pts = dct["points"]
    for pt in pts
        x = _parse_x(d, pt["x"])
        lp = d.lower <= x <= d.upper ? Float64(pt["logpdf"]) - d.logtp : -Inf
        cf = x <= d.lower ? 0.0 : x >= d.upper ? 1.0 : (Float64(pt["cdf"]) - d.lcdf)/d.tp
        if !isa(d, Distributions.Truncated{Distributions.StudentizedRange{Float64},Distributions.Continuous})
            @test isapprox(logpdf(d, x), lp, atol=sqrt(eps()))
        end
        @test isapprox(cdf(d, x)   , cf, atol=sqrt(eps()))
        # NOTE: some distributions use pdf() in StatsFuns.jl which have no generic support yet
        if !(typeof(d) in [Distributions.Truncated{Distributions.NoncentralChisq{Float64},Distributions.Continuous},
                           Distributions.Truncated{Distributions.NoncentralF{Float64},Distributions.Continuous},
                           Distributions.Truncated{Distributions.NoncentralT{Float64},Distributions.Continuous},
                           Distributions.Truncated{Distributions.StudentizedRange{Float64},Distributions.Continuous}])
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


## main

for c in ["discrete",
          "continuous"]

    title = string(uppercase(c[1]), c[2:end])
    println("    [$title]")
    println("    ------------")
    jsonfile = joinpath(dirname(@__FILE__), "ref", "$(c)_test.ref.json")
    verify_and_test_drive(jsonfile, ARGS, 10^6,3,5)
    println()
end

end
