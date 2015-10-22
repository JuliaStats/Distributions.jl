# Testing discrete univariate distributions

using Distributions
import JSON
using Base.Test
using Compat


function verify_and_test_drive(jsonfile, selected, n_tsamples::Int, offset::Int)
    R = JSON.parsefile(jsonfile)

    for (ex, dct) in R
        dsym = symbol(dct["dtype"])
        dname = string(dsym)

        dsymt = symbol("Shifted($(dct["dtype"]),$offset")
        dnamet = string(dsym)

        # test whether it is included in the selected list
        # or all are selected (when selected is empty)
        if !isempty(selected) && !(dname in selected)
            continue
        end

        # perform testing
        println("    testing Shifted($(ex),$offset)")
        dtype = eval(dsym)
        dtypet = Shifted
        d = Shifted(eval(parse(ex)),offset)
        @test isa(d, dtypet)
        #@assert isa(dtype, Type) && dtype <: UnivariateDistribution # fails for TruncatedNormal for some reason

        # verification and testing
        verify_and_test(d, dct, n_tsamples)
    end
end


@compat _parse_x(d::DiscreteUnivariateDistribution, x) = round(Int, x)
@compat _parse_x(d::ContinuousUnivariateDistribution, x) = Float64(x)

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
        f = eval(symbol(fname))
        @assert isa(f, Function)
        Base.Test.test_approx_eq(f(d.unshifted), val, "$fname(d.unshifted)", "val")
    end

    # verify stats
    @test_approx_eq minimum(d) _json_value(dct["minimum"]) + d.offset
    @test_approx_eq maximum(d) _json_value(dct["maximum"]) + d.offset

    # verify logpdf and cdf at certain points
    pts = dct["points"]
    for pt in pts
        x = _parse_x(d, pt["x"])
        @Base.Test.test_approx_eq_eps(logpdf(d, x + d.offset), Float64(pt["logpdf"]), sqrt(eps()))
        @Base.Test.test_approx_eq_eps(cdf(d, x + d.offset), Float64(pt["cdf"]), sqrt(eps()))
    end

    rand(d) # makes sure this runs

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
    verify_and_test_drive(jsonfile, ARGS, 10^6, 3)
    println()
end
