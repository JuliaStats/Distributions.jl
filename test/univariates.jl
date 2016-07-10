# Testing discrete univariate distributions

using Distributions
import JSON
using Base.Test
using Compat


function verify_and_test_drive(jsonfile, selected, n_tsamples::Int)
    R = JSON.parsefile(jsonfile)
    for (ex, dct) in R
        dsym = Symbol(dct["dtype"])
        dname = string(dsym)

        # test whether it is included in the selected list
        # or all are selected (when selected is empty)
        if !isempty(selected) && !(dname in selected)
            continue
        end

        # perform testing
        println("    testing $(ex)")
        dtype = eval(dsym)
        d = eval(parse(ex))
        if dtype == TruncatedNormal
            @test isa(d, Truncated{Normal{Float64}})
        else
            @assert isa(dtype, Type) && dtype <: UnivariateDistribution
            @test isa(d, dtype)
        end

        # verification and testing
        verify_and_test(d, dct, n_tsamples)
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
        Base.Test.test_approx_eq(f(d), val, "$fname(d)", "val")
    end

    # verify stats
    @test_approx_eq minimum(d) _json_value(dct["minimum"])
    @test_approx_eq maximum(d) _json_value(dct["maximum"])
    @test_approx_eq_eps mean(d) _json_value(dct["mean"]) 1.0e-8
    if !isa(d, VonMises)
        @test_approx_eq_eps var(d) _json_value(dct["var"]) 1.0e-8
    end
    if !isa(d, Skellam)
        @test_approx_eq_eps median(d) _json_value(dct["median"]) 1.0
    end

    if applicable(entropy, d) && !isa(d, VonMises)  # SciPy VonMises entropy is wrong
        @test_approx_eq_eps entropy(d) dct["entropy"] 1.0e-7
    end

    # test conversions if distribution is parametric
    if !isempty(typeof(d).parameters) && !isa(d, Truncated)
        D = typeof(d).name.primary
        W = Float32
        @test typeof(convert(D{W}, d)) == D{W}
        @test typeof(convert(D{W}, params(d)...)) == D{W}
    end

    # TODO: test various constructors for promotion, all-Integer args, etc.

    # verify quantiles
    if !isa(d, Union{Skellam, VonMises})
        @test_approx_eq_eps quantile(d, 0.10) dct["q10"] 1.0e-8
        @test_approx_eq_eps quantile(d, 0.25) dct["q25"] 1.0e-8
        @test_approx_eq_eps quantile(d, 0.50) dct["q50"] 1.0e-8
        @test_approx_eq_eps quantile(d, 0.75) dct["q75"] 1.0e-8
        @test_approx_eq_eps quantile(d, 0.90) dct["q90"] 1.0e-8
    end

    # verify logpdf and cdf at certain points
    pts = dct["points"]
    for pt in pts
        x = _parse_x(d, pt["x"])
        lp = Float64(pt["logpdf"])
        cf = Float64(pt["cdf"])
        Base.Test.test_approx_eq(logpdf(d, x), lp, "logpdf(d, $x)", "lp")
        if !isa(d, Skellam)
            Base.Test.test_approx_eq(cdf(d, x), cf, "cdf(d, $x)", "cf")
        end
    end

    try
        m = mgf(d,0.0)
        Base.Test.test_approx_eq(m, 1.0, "mgf(d, 0.0)", "1.0")
    catch e
        isa(e, MethodError) || throw(e)
    end
    try
        c = cf(d,0.0)
        Base.Test.test_approx_eq(c, 1.0, "cf(d, 0.0)", "1.0")
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
    if !isa(d, Union{Skellam, VonMises})
        test_distr(d, n_tsamples)
    end
end


## main

for c in ["discrete",
          "continuous"]

    title = string(uppercase(c[1]), c[2:end])
    println("    [$title]")
    println("    ------------")
    jsonfile = joinpath(dirname(@__FILE__), "$(c)_test.json")
    verify_and_test_drive(jsonfile, ARGS, 10^6)
    println()
end
