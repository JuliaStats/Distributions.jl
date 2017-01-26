# Testing discrete univariate distributions

using Distributions
import JSON
using Base.Test
using Compat


function verify_and_test_drive(jsonfile, selected, n_tsamples::Int)
    R = JSON.parsefile(jsonfile)
    for dct in R
        ex = parse(dct["expr"])
        @assert ex.head == :call
        dsym = ex.args[1]
        dname = string(dsym)

        # test whether it is included in the selected list
        # or all are selected (when selected is empty)
        if !isempty(selected) && !(dname in selected)
            continue
        end

        # perform testing
        println("    testing $(ex)")
        dtype = eval(dsym)
        d = eval(ex)
        if dtype == TruncatedNormal
            @test isa(d, Truncated{Normal{Float64}})
        else
            @assert isa(dtype, Type) && dtype <: UnivariateDistribution
            @test isa(d, dtype)
        end

        # verification and testing
        verify_and_test(dtype, d, dct, n_tsamples)
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


function verify_and_test(D::Type, d::UnivariateDistribution, dct::Dict, n_tsamples::Int)
    # verify properties
    #
    # Note: properties include all applicable params and stats
    #
    assert(isa(d, D))
    
    pdct = dct["properties"]
    for (fname, val) in pdct
        expect_v = _json_value(val)
        f = eval(Symbol(fname))
        @assert isa(f, Function)
        Base.Test.test_approx_eq(f(d), expect_v, "$fname(d)", "expect_v")
    end

    # test various constructors for promotion, all-Integer args, etc.
    pars = params(d)

    # promotion constructor:
    float_pars = map(x -> isa(x, AbstractFloat), pars)
    if length(pars) > 1 && sum(float_pars) > 1
        mixed_pars = Any[pars...]
        first_float = findfirst(float_pars)
        mixed_pars[first_float] = Float32(mixed_pars[first_float])

        @test typeof(D(mixed_pars...)) == typeof(d)
    end

    # promote integer arguments to floats, where applicable
    if sum(float_pars) >= 1 && !any(map(isinf, pars)) && !isa(d, Geometric)
        int_pars = map(x -> ceil(Int, x), pars)
        @test typeof(D(int_pars...)) == typeof(d)
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
        else
            @test_throws MethodError cdf(d, x)
        end
    end

    # verify quantiles
    qts = dct["quans"]
    for qt in qts
        q = Float64(qt["q"])
        x = Float64(qt["x"])
        @test isapprox(quantile(d, q), x, atol=1.0e-8)
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
    jsonfile = joinpath(dirname(@__FILE__), "ref", "$(c)_test.ref.json")
    verify_and_test_drive(jsonfile, ARGS, 10^6)
    println()
end
