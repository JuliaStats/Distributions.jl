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
        if dsym == :truncated
            @test isa(d, Truncated{Normal{Float64}})
        else
            @test dtype isa Type && dtype <: UnivariateDistribution
            @test d isa dtype
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
    if length(pars) > 1 && sum(float_pars) > 1 && !isa(D, typeof(truncated))
        mixed_pars = Any[pars...]
        first_float = findfirst(float_pars)
        mixed_pars[first_float] = Float32(mixed_pars[first_float])

        @test typeof(D(mixed_pars...)) == typeof(d)
    end

    # conversions
    if D isa Type && !isconcretetype(D)
        @test convert(D{partype(d)}, d) === d
        d32 = convert(D{Float32}, d)
        @test d32 isa D{Float32}
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
            @test Base.Fix1(pdf, d).(x) ≈ p atol=1e-16 rtol=1e-8
            @test Base.Fix1(logpdf, d).(x) ≈ lp atol=isa(d, NoncentralHypergeometric) ? 1e-4 : 1e-12
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
        @test m ≈ 1.0
    catch e
        isa(e, MethodError) || throw(e)
    end
    try
        c = cf(d,0.0)
        @test c ≈ 1.0
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
    jsonfile = joinpath(@__DIR__, "ref", "$(c)_test.ref.json")
    verify_and_test_drive(jsonfile, ARGS, 10^6)
    println()
end

# #1358
@testset "Poisson quantile" begin
    d = Poisson(1)
    @test quantile(d, 0.2) isa Int
    @test cquantile(d, 0.4) isa Int
    @test invlogcdf(d, log(0.2)) isa Int
    @test invlogccdf(d, log(0.6)) isa Int
end

# #1471
@testset "InverseGamma constructor (#1471)" begin
    @test_throws DomainError InverseGamma(-1, 2)
    InverseGamma(-1, 2; check_args=false) # no error
end

# #1479
@testset "Inner and outer constructors" begin
    @test_throws DomainError InverseGaussian(0.0, 0.0)
    @test InverseGaussian(0.0, 0.0; check_args=false) isa InverseGaussian{Float64}
    @test InverseGaussian{Float64}(0.0, 0.0) isa InverseGaussian{Float64}

    @test_throws DomainError Levy(0.0, 0.0)
    @test Levy(0.0, 0.0; check_args=false) isa Levy{Float64}
    @test Levy{Float64}(0.0, 0.0) isa Levy{Float64}
end
