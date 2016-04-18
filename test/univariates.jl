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
    @test_approx_eq_eps var(d) _json_value(dct["var"]) 1.0e-8
    @test_approx_eq_eps median(d) _json_value(dct["median"]) 1.0

    if applicable(entropy, d)
        @test_approx_eq_eps entropy(d) dct["entropy"] 1.0e-7
    end

    # verify quantiles
    @test_approx_eq_eps quantile(d, 0.10) dct["q10"] 1.0e-8
    @test_approx_eq_eps quantile(d, 0.25) dct["q25"] 1.0e-8
    @test_approx_eq_eps quantile(d, 0.50) dct["q50"] 1.0e-8
    @test_approx_eq_eps quantile(d, 0.75) dct["q75"] 1.0e-8
    @test_approx_eq_eps quantile(d, 0.90) dct["q90"] 1.0e-8

    # verify logpdf and cdf at certain points
    pts = dct["points"]
    for pt in pts
        x = _parse_x(d, pt["x"])
        lp = Float64(pt["logpdf"])
        cf = Float64(pt["cdf"])
        Base.Test.test_approx_eq(logpdf(d, x), lp, "logpdf(d, $x)", "lp")
        Base.Test.test_approx_eq(cdf(d, x), cf, "cdf(d, $x)", "cf")
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

    # test conversions between type parameters
    if isa(d, Normal)  # remove this check when all distributions parameterized
        @test isa(convert(Normal{Float64}, Normal(1, 2)), Normal{Float64})
        @test isa(convert(Normal{Float64}, 1, 2), Normal{Float64})
    end

    if haskey(dct, "conversions")
        for cv in dct["conversions"]
            pars = cv["from"]
            from = isa(pars, AbstractString) ? eval(parse(pars)) : pars
            to = eval(parse(cv["to"]))
            @test isa(convert(to, from...), to)
        end
    end

    # generic testing
    if isa(d, Cosine)
        n_tsamples = floor(Int, n_tsamples / 10)
    end
    test_distr(d, n_tsamples)
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
