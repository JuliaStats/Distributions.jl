# Testing discrete univariate distributions

module TestTruncate

using Distributions
import JSON
using Base.Test
using Compat


function verify_and_test_drive(jsonfile, selected, n_tsamples::Int,lower::Int,upper::Int)
    R = JSON.parsefile(jsonfile)
    for (ex, dct) in R
        dsym = Symbol(dct["dtype"])
        dname = string(dsym)

        dsymt = Symbol("Truncated($(dct["dtype"]),$lower,$upper")
        dnamet = string(dsym)

        # test whether it is included in the selected list
        # or all are selected (when selected is empty)
        if !isempty(selected) && !(dname in selected)
            continue
        end

        # perform testing
        println("    testing Truncated($(ex),$lower,$upper)")
        dtype = eval(dsym)
        dtypet = Truncated
        if !(dsym in [:Skellam, ])
            d = Truncated(eval(parse(ex)),lower,upper)
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
    # verify parameters
    pdct = dct["params"]
    for (fname, val) in pdct
        f = eval(Symbol(fname))
        @assert isa(f, Function)
        Base.Test.test_approx_eq(f(d.untruncated), val, "$fname(d.untruncated)", "val")
    end

    # verify stats
    @test_approx_eq minimum(d) max(_json_value(dct["minimum"]),d.lower)
    @test_approx_eq maximum(d) min(_json_value(dct["maximum"]),d.upper)

    # verify logpdf and cdf at certain points
    pts = dct["points"]
    for pt in pts
        x = _parse_x(d, pt["x"])
        lp = d.lower <= x <= d.upper ? Float64(pt["logpdf"]) - d.logtp : -Inf
        cf = x <= d.lower ? 0.0 : x >= d.upper ? 1.0 : (Float64(pt["cdf"]) - d.lcdf)/d.tp
        @Base.Test.test_approx_eq_eps(logpdf(d, x), lp, sqrt(eps()))
        @Base.Test.test_approx_eq_eps(cdf(d, x), cf, sqrt(eps()))
    end

    try
        m = mgf(d,0.0)
        @test m == 1.0
    catch e
        isa(e, MethodError) || throw(e)
    end
    try
        c = cf(d,0.0)
        @test c == 1.0
        # test some extra values: should all be well-defined
        for t in (0.1,-0.1,1.0,-1.0)
            @test !isnan(cf(d,t))
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
    jsonfile = joinpath(dirname(@__FILE__), "$(c)_test.json")
    verify_and_test_drive(jsonfile, ARGS, 10^6,3,5)
    println()
end

end
