# Testing discrete univariate distributions

using Distributions, Random
using Test
using Calculus: derivative

import JSON


@testset "Testing distributions with R counterparts" begin
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

        # promote integer arguments to floats, where applicable
        if sum(float_pars) >= 1 && !any(map(isinf, pars)) && !isa(d, Geometric) && !isa(D, typeof(truncated))
            int_pars = map(x -> ceil(Int, x), pars)
            @test typeof(D(int_pars...)) == typeof(d)
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
        jsonfile = joinpath(@__DIR__, "_ref", "$(c)_test.ref.json")
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

    @testset "Uniform type inference" begin
        for T in (Int, Float32)
            d = Uniform{T}(T(2), T(3))
            FT = float(T)
            XFT = promote_type(FT, Float64)

            @test @inferred(pdf(d, 1.5)) === zero(FT)
            @test @inferred(pdf(d, 2.5)) === one(FT)
            @test @inferred(pdf(d, 3.5)) === zero(FT)

            @test @inferred(logpdf(d, 1.5)) === FT(-Inf)
            @test @inferred(logpdf(d, 2.5)) === -zero(FT) # negative zero
            @test @inferred(logpdf(d, 3.5)) === FT(-Inf)

            @test @inferred(cdf(d, 1.5)) === zero(XFT)
            @test @inferred(cdf(d, 2.5)) === XFT(1//2)
            @test @inferred(cdf(d, 3.5)) === one(XFT)

            @test @inferred(ccdf(d, 1.5)) === one(XFT)
            @test @inferred(ccdf(d, 2.5)) === XFT(1//2)
            @test @inferred(ccdf(d, 3.5)) === zero(XFT)
        end
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
end

@testset "Testing additional distributions" begin
    n_tsamples = 100

    # additional distributions that have no direct counterparts in R references
    @testset "Testing $(distr)" for distr in [Biweight(),
                                            Biweight(1,3),
                                            Epanechnikov(),
                                            Epanechnikov(1,3),
                                            Triweight(),
                                            Triweight(2),
                                            Triweight(1, 3),
                                            Triweight(1)]

        test_distr(distr, n_tsamples; testquan=false)
    end

    # Test for non-Float64 input
    using ForwardDiff
    @test string(logpdf(Normal(0,1),big(1))) == "-1.418938533204672741780329736405617639861397473637783412817151540482765695927251"
    @test derivative(t -> logpdf(Normal(1.0, 0.15), t), 2.5) ≈ -66.66666666666667
    @test derivative(t -> pdf(Normal(t, 1.0), 0.0), 0.0) == 0.0


    @testset "Normal distribution with non-standard (ie not Float64) parameter types" begin
        n32 = Normal(1f0, 0.1f0)
        n64 = Normal(1., 0.1)
        nbig = Normal(big(pi), big(ℯ))

        @test eltype(typeof(n32)) === Float32
        @test eltype(rand(n32)) === Float32
        @test eltype(rand(n32, 4)) === Float32

        @test eltype(typeof(n64)) === Float64
        @test eltype(rand(n64)) === Float64
        @test eltype(rand(n64, 4)) === Float64
    end

    # Test for numerical problems
    @test pdf(Logistic(6,0.01),-2) == 0

    @testset "Normal with std=0" begin
        d = Normal(0.5,0.0)
        @test pdf(d, 0.49) == 0.0
        @test pdf(d, 0.5) == Inf
        @test pdf(d, 0.51) == 0.0

        @test cdf(d, 0.49) == 0.0
        @test cdf(d, 0.5) == 1.0
        @test cdf(d, 0.51) == 1.0

        @test ccdf(d, 0.49) == 1.0
        @test ccdf(d, 0.5) == 0.0
        @test ccdf(d, 0.51) == 0.0

        @test quantile(d, 0.0) == -Inf
        @test quantile(d, 0.49) == 0.5
        @test quantile(d, 0.5) == 0.5
        @test quantile(d, 0.51) == 0.5
        @test quantile(d, 1.0) == +Inf

        @test rand(d) == 0.5
        @test rand(MersenneTwister(123), d) == 0.5
    end

    # Test for parameters beyond those supported in R references
    @testset "VonMises with large kappa" begin
        d = VonMises(1.1, 1000)
        @test var(d) ≈ 0.0005001251251957198
        @test entropy(d) ≈ -2.034688918525470
        @test cf(d, 2.5) ≈ -0.921417 + 0.38047im atol=1e-6
        @test pdf(d, 0.5) ≈ 1.758235814051e-75
        @test logpdf(d, 0.5) ≈ -172.1295710466005
        @test cdf(d, 1.0) ≈ 0.000787319 atol=1e-9
    end

    @testset "NormalInverseGaussian random repeatable and basic metrics" begin
        rng = Random.MersenneTwister(42)
        rng2 = copy(rng)
        µ = 0.0
        α = 1.0
        β = 0.5
        δ = 3.0
        g = sqrt(α^2 - β^2)
        d = NormalInverseGaussian(μ, α, β, δ)
        v1 = rand(rng, d)
        v2 = rand(rng, d)
        v3 = rand(rng2, d)
        @test v1 ≈ v3
        @test v1 ≉ v2

        @test mean(d) ≈ µ + β * δ / g
        @test var(d) ≈ δ * α^2 / g^3
        @test skewness(d) ≈ 3β/(α*sqrt(δ*g))
    end

    @testset "edge cases" begin
        # issue #1371: cdf should not return -0.0
        @test cdf(Rayleigh(1), 0) === 0.0
        @test cdf(Rayleigh(1), -10) === 0.0
    end
end
