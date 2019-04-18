# Testing discrete univariate distributions

using Distributions
import JSON
using  Test


function verify_and_test_drive(jsonfile, selected, n_tsamples::Int)
    R = JSON.parsefile(jsonfile)
    for dct in R
        ex = Meta.parse(dct["expr"])
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


function verify_and_test(D::Union{Type,Function}, d::UnivariateDistribution, dct::Dict, n_tsamples::Int)
    # verify properties
    #
    # Note: properties include all applicable params and stats
    #

    # D can be a function, e.g. TruncatedNormal
    if isa(D, Type)
        @assert isa(d, D)
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

    # verify properties (params & stats)
    pdct = dct["properties"]
    for (fname, val) in pdct
        expect_v = _json_value(val)
        f = eval(Symbol(fname))
        @assert isa(f, Function)
        @test isapprox(f(d), expect_v; atol=1e-12, rtol=1e-8, nans=true)
    end
    @test extrema(d) == (minimum(d), maximum(d))

    # verify logpdf and cdf at certain points
    pts = dct["points"]
    for pt in pts
        x = _parse_x(d, pt["x"])
        p = _json_value(pt["pdf"])
        lp = _json_value(pt["logpdf"])
        cf = _json_value(pt["cdf"])

        # pdf method is not implemented for StudentizedRange
        if !isa(d, StudentizedRange)
            @test isapprox(pdf.(d, x),     p; atol=1e-16, rtol=1e-8)
            @test isapprox(logpdf.(d, x), lp; atol=isa(d, NoncentralHypergeometric) ? 1e-4 : 1e-12)
        end

        # cdf method is not implemented for NormalInverseGaussian
        if !isa(d, NormalInverseGaussian)
            @test isapprox(cdf(d, x), cf; atol=isa(d, NoncentralHypergeometric) ? 1e-8 : 1e-12)
        end
    end

    # verify quantiles
    if !isa(d, Union{Skellam, VonMises, NormalInverseGaussian})
        qts = dct["quans"]
        for qt in qts
            q = Float64(qt["q"])
            x = Float64(qt["x"])
            @test isapprox(quantile(d, q), x, atol=1.0e-8)
        end
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
    elseif isa(d, NoncentralBeta) ||
           isa(d, NoncentralChisq) ||
           isa(d, NoncentralF) ||
           isa(d, NoncentralT)
        n_tsamples = min(n_tsamples, 100)
    end

    if !isa(d, Union{Skellam,
                     VonMises,
                     NoncentralHypergeometric,
                     NormalInverseGaussian})
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
