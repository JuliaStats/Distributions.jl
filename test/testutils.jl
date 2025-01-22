# Utilities to support the testing of distributions and samplers

using Distributions
using Random
using Printf: @printf
using Test: @test
import FiniteDifferences
import ForwardDiff

# to workaround issues of Base.linspace
function _linspace(a::Float64, b::Float64, n::Int)
    intv = (b - a) / (n - 1)
    r = Vector{Float64}(undef, n)
    @inbounds for i = 1:n
        r[i] = a + (i-1) * intv
    end
    r[n] = b
    return r
end


#################################################
#
#   Driver functions
#
#################################################

# testing the implementation of a discrete univariate distribution
#
function test_distr(distr::DiscreteUnivariateDistribution, n::Int;
                    testquan::Bool=true, rng::AbstractRNG = Random.default_rng())

    test_range(distr)
    vs = get_evalsamples(distr, 0.00001)

    test_support(distr, vs)
    test_evaluation(distr, vs, testquan)
    test_range_evaluation(distr)
    test_nonfinite(distr)

    test_stats(distr, vs)
    test_samples(distr, n)
    test_samples(distr, n; rng=rng)
    test_params(distr)
end

function test_cgf(dist, ts)
    κ₀ = cgf(dist, 0)
    @test κ₀ ≈ 0 atol=2*eps(one(float(κ₀)))
    d(f) = Base.Fix1(ForwardDiff.derivative, f)
    κ₁ = d(Base.Fix1(cgf, dist))(0)
    @test κ₁ ≈ mean(dist)
    if VERSION >= v"1.4"
        κ₂ = d(d(Base.Fix1(cgf, dist)))(0)
        @test κ₂ ≈ var(dist)
    end
    for t in ts
        val = @inferred cgf(dist, t)
        @test isfinite(val)
        if isfinite(mgf(dist, t))
            rtol = eps(float(one(t)))^(1/2)
            @test (exp∘cgf)(dist, t) ≈ mgf(dist, t) rtol=rtol
        end
    end
end

# testing the implementation of a continuous univariate distribution
#
function test_distr(distr::ContinuousUnivariateDistribution, n::Int;
                    testquan::Bool=true, rng::AbstractRNG=MersenneTwister(123))
    test_range(distr)
    vs = get_evalsamples(distr, 0.01, 2000)

    test_support(distr, vs)
    test_evaluation(distr, vs, testquan)
    test_nonfinite(distr)

    if isa(distr, StudentizedRange)
        n = 2000 # must use fewer values due to performance
    end
    xs = test_samples(distr, n)
    allow_test_stats(distr) && test_stats(distr, xs)
    xs = test_samples(distr, n, rng=rng)
    allow_test_stats(distr) && test_stats(distr, xs)
    test_params(distr)
end


#################################################
#
#   Core testing functions
#
#################################################

#### Testing sampleable objects (samplers)

# for discrete samplers
#
function test_samples(s::Sampleable{Univariate, Discrete},      # the sampleable instance
                      distr::DiscreteUnivariateDistribution,    # corresponding distribution
                      n::Int;                                   # number of samples to generate
                      q::Float64=1.0e-7,                        # confidence interval, 1 - q as confidence
                      verbose::Bool=false,                      # show intermediate info (for debugging)
                      rng::Union{AbstractRNG, Missing}=missing) # add an rng?

    # The basic idea
    # ------------------
    #   Generate n samples, and count the occurrences of each value within a reasonable range.
    #   For each distinct value, it computes an confidence interval of the counts
    #   and checks whether the count is within this interval.
    #
    #   If the distribution has a bounded range, it also checks whether
    #   the samples are all within this range.
    #
    #   By setting a small q, we ensure that failure of the tests rarely
    #   happen in practice.
    #

    verbose && println("test_samples on $(typeof(s))")

    n > 1 || error("The number of samples must be greater than 1.")
    0.0 < q < 0.1 || error("The value of q must be within the open interval (0.0, 0.1).")

    # determine the range of values to examine
    vmin = minimum(distr)
    vmax = maximum(distr)

    rmin = floor(Int,quantile(distr, 0.00001))::Int
    rmax = floor(Int,quantile(distr, 0.99999))::Int
    m = rmax - rmin + 1  # length of the range
    p0 = map(Base.Fix1(pdf, distr), rmin:rmax)  # reference probability masses
    @assert length(p0) == m

    # determine confidence intervals for counts:
    # with probability q, the count will be out of this interval.
    #
    clb = Vector{Int}(undef, m)
    cub = Vector{Int}(undef, m)
    for i = 1:m
        bp = Binomial(n, p0[i])
        clb[i] = floor(Int,quantile(bp, q/2))
        cub[i] = ceil(Int,cquantile(bp, q/2))
        @assert cub[i] >= clb[i]
    end

    # generate samples using RNG passed or default RNG
    # we also check reproducibility
    if rng === missing
        Random.seed!(1234)
        samples = rand(s, n)
        Random.seed!(1234)
        samples2 = rand(s, n)
        Random.seed!(1234)
        samples3 = [rand(s) for _ in 1:n]
        Random.seed!(1234)
        samples4 = [rand(s) for _ in 1:n]
    else
        # RNGs have to be copied with `copy`, not `deepcopy`
        # Ref https://github.com/JuliaLang/julia/issues/42899
        rng2 = copy(rng)
        rng3 = copy(rng)
        rng4 = copy(rng)
        samples = rand(rng, s, n)
        samples2 = rand(rng2, s, n)
        samples3 = [rand(rng3, s) for _ in 1:n]
        samples4 = [rand(rng4, s) for _ in 1:n]
    end
    @test length(samples) == n
    @test samples2 == samples
    @test samples3 == samples4

    # scan samples and get counts
    cnts = zeros(Int, m)
    cnts_sc = zeros(Int, m)
    for i = 1:n
        @inbounds si = samples[i]
        if rmin <= si <= rmax
            cnts[si - rmin + 1] += 1
        else
            vmin <= si <= vmax ||
                throw(DomainError(si, "sample generated by `rand(s, n)` is out of valid range [$vmin, $vmax]."))
        end

        @inbounds si_sc = samples3[i]
        if rmin <= si_sc <= rmax
            cnts_sc[si_sc - rmin + 1] += 1
        else
            vmin <= si_sc <= vmax ||
                throw(DomainError(si, "sample generated by `[rand(s) for _ in 1:n]` is out of valid range [$vmin, $vmax]."))
        end
    end

    # check the counts
    for i = 1:m
        verbose && println("v = $(rmin+i-1) ==> ($(clb[i]), $(cub[i])): $(cnts[i])")
        clb[i] <= cnts[i] <= cub[i] ||
            error("The counts of samples generated by `rand(s, n)` are out of the confidence interval.")

        verbose && println("v = $(rmin+i-1) ==> ($(clb[i]), $(cub[i])): $(cnts_sc[i])")
        clb[i] <= cnts_sc[i] <= cub[i] ||
            error("The counts of samples generated by `[rand(s) for _ in 1:n]` are out of the confidence interval.")
    end
    return samples
end

test_samples(distr::DiscreteUnivariateDistribution, n::Int;
             q::Float64=1.0e-6, verbose::Bool=false, rng=missing) =
    test_samples(distr, distr, n; q=q, verbose=verbose, rng=rng)

# for continuous samplers
#
function test_samples(s::Sampleable{Univariate, Continuous},    # the sampleable instance
                      distr::ContinuousUnivariateDistribution,  # corresponding distribution
                      n::Int;                                   # number of samples to generate
                      nbins::Int=50,                            # divide the main interval into nbins
                      q::Float64=1.0e-6,                        # confidence interval, 1 - q as confidence
                      verbose::Bool=false,                      # show intermediate info (for debugging)
                      rng::Union{AbstractRNG, Missing}=missing) # add an rng?

    # The basic idea
    # ------------------
    #   Generate n samples, divide the interval [q00, q99] into nbins bins, where
    #   q01 and q99 are 0.01 and 0.99 quantiles, and count the numbers of samples
    #   falling into each bin. For each bin, we will compute a confidence interval
    #   of the number, and see whether the actual number is in this interval.
    #
    #   If the distribution has a bounded range, it also checks whether
    #   the samples are all within this range.
    #
    #   By setting a small q, we ensure that failure of the tests rarely
    #   happen in practice.
    #

    verbose && println("test_samples on $(typeof(s))")

    n > 1 || error("The number of samples must be greater than 1.")
    nbins > 1 || error("The number of bins must be greater than 1.")
    0.0 < q < 0.1 || error("The value of q must be within the open interval (0.0, 0.1).")

    # determine the range of values to examine
    vmin = minimum(distr)
    vmax = maximum(distr)

    local rmin::Float64
    local rmax::Float64
    if applicable(quantile, distr, 0.5)
        rmin = quantile(distr, 0.01)
        rmax = quantile(distr, 0.99)
    elseif isfinite(vmin) && isfinite(vmax)
        rmin = vmin
        rmax = vmax
    end
    edges = _linspace(rmin, rmax, nbins + 1)

    # determine confidence intervals for counts:
    # with probability q, the count will be out of this interval.
    #
    clb = Vector{Int}(undef, nbins)
    cub = Vector{Int}(undef, nbins)
    cdfs = map(Base.Fix1(cdf, distr), edges)

    for i = 1:nbins
        pi = cdfs[i+1] - cdfs[i]
        bp = Binomial(n, pi)
        clb[i] = floor(Int,quantile(bp, q/2))
        cub[i] = ceil(Int,cquantile(bp, q/2))
        @assert cub[i] >= clb[i]
    end

    # generate samples using RNG passed or default RNG
    # we also check reproducibility
    if rng === missing
        Random.seed!(1234)
        samples = rand(s, n)
        Random.seed!(1234)
        samples2 = rand(s, n)
        Random.seed!(1234)
        samples3 = [rand(s) for _ in 1:n]
        Random.seed!(1234)
        samples4 = [rand(s) for _ in 1:n]
    else
        # RNGs have to be copied with `copy`, not `deepcopy`
        # Ref https://github.com/JuliaLang/julia/issues/42899
        rng2 = copy(rng)
        rng3 = copy(rng)
        rng4 = copy(rng)
        samples = rand(rng, s, n)
        samples2 = rand(rng2, s, n)
        samples3 = [rand(rng3, s) for _ in 1:n]
        samples4 = [rand(rng4, s) for _ in 1:n]
    end
    @test length(samples) == n
    @test samples2 == samples
    @test samples3 == samples4

    if isa(distr, StudentizedRange)
        samples[isnan.(samples)] .= 0.0 # Underlying implementation in Rmath can't handle very low values.
    end

    # check whether all samples are in the valid range
    for i = 1:n
        @inbounds si = samples[i]
        vmin <= si <= vmax ||
            throw(DomainError(si, "sample generated by `rand(s, n)` is out of valid range [$vmin, $vmax]."))
        @inbounds si_sc = samples3[i]
        vmin <= si_sc <= vmax ||
            throw(DomainError(si, "sample generated by `[rand(s) for _ in 1:n]` is out of valid range [$vmin, $vmax]."))
    end

    # get counts
    cnts = fit(Histogram, samples, edges; closed=:right).weights
    @assert length(cnts) == nbins

    cnts_sc = fit(Histogram, samples3, edges; closed=:right).weights
    @assert length(cnts_sc) == nbins

    # check the counts
    for i = 1:nbins
        if verbose
            @printf("[%.4f, %.4f) ==> (%d, %d): %d\n", edges[i], edges[i+1], clb[i], cub[i], cnts[i])
            @printf("[%.4f, %.4f) ==> (%d, %d): %d\n", edges[i], edges[i+1], clb[i], cub[i], cnts_sc[i])
        end
        clb[i] <= cnts[i] <= cub[i] ||
            error("The counts of samples generated by `rand(s, n)` are out of the confidence interval.")
        clb[i] <= cnts_sc[i] <= cub[i] ||
            error("The counts of samples generated by `[rand(s) for _ in 1:n]` are out of the confidence interval.")
    end
    return samples
end

test_samples(distr::ContinuousUnivariateDistribution, n::Int; nbins::Int=50, q::Float64=1.0e-6, verbose::Bool=false, rng=missing) =
    test_samples(distr, distr, n; nbins=nbins, q=q, verbose=verbose, rng=rng)


#### Testing range & support methods

function test_range(d::UnivariateDistribution)
    vmin = minimum(d)
    vmax = maximum(d)
    @test vmin <= vmax

    is_lb = islowerbounded(d)
    is_ub = isupperbounded(d)

    @test isfinite(vmin) == is_lb
    @test isfinite(vmax) == is_ub
    @test isbounded(d) == (is_lb && is_ub)
end

function get_evalsamples(d::DiscreteUnivariateDistribution, q::Float64)
    # samples for testing evaluation functions (even spacing)

    T = eltype(typeof(d))
    lv = (islowerbounded(d) ? minimum(d) : floor(T,quantile(d, q/2)))::T
    hv = (isupperbounded(d) ? maximum(d) : ceil(T,cquantile(d, q/2)))::T
    @assert lv <= hv
    return lv:hv
end

function get_evalsamples(d::ContinuousUnivariateDistribution, q::Float64, n::Int)
    # samples for testing evaluation functions (even spacing)

    lv = quantile(d, q/2)
    hv = cquantile(d, q/2)
    @assert isfinite(lv) && isfinite(hv) && lv <= hv
    return _linspace(lv, hv, n)
end

function test_support(d::UnivariateDistribution, vs::AbstractVector)
    for v in vs
        @test insupport(d, v)
    end
    @test all(insupport(d, vs))

    if islowerbounded(d)
        @test isfinite(minimum(d))
        @test insupport(d, minimum(d))
        @test !insupport(d, minimum(d)-1)
    end
    if isupperbounded(d)
        @test isfinite(maximum(d))
        @test insupport(d, maximum(d))
        @test !insupport(d, maximum(d)+1)
    end

    @test isbounded(d) == (isupperbounded(d) && islowerbounded(d))

    # Test the `Base.in` or `∈` operator
    # The `support` function is buggy for unbounded `DiscreteUnivariateDistribution`s
    if isbounded(d) || isa(d, ContinuousUnivariateDistribution)
        s = support(d)
        for v in vs
            @test v ∈ s
        end

        if islowerbounded(d)
            @test minimum(d) ∈ s
            @test (minimum(d) - 1) ∉ s
        end
        if isupperbounded(d)
            @test maximum(d) ∈ s
            @test (maximum(d) + 1) ∉ s
        end
    end

    if isbounded(d) && isa(d, DiscreteUnivariateDistribution)
        s = support(d)
        @test isa(s, AbstractUnitRange)
        @test first(s) == minimum(d)
        @test last(s) == maximum(d)
    end
end


#### Testing evaluation methods

function test_range_evaluation(d::DiscreteUnivariateDistribution)
    # check the consistency between range-based and ordinary pdf
    vmin = minimum(d)
    vmax = maximum(d)
    @test vmin <= vmax
    if islowerbounded(d)
        @test isa(vmin, Integer)
    end
    if isupperbounded(d)
        @test isa(vmax, Integer)
    end

    rmin = round(Int, islowerbounded(d) ? vmin : quantile(d, 0.001))::Int
    rmax = round(Int, isupperbounded(d) ? vmax : quantile(d, 0.999))::Int

    p0 = map(Base.Fix1(pdf, d), collect(rmin:rmax))
    @test map(Base.Fix1(pdf, d), rmin:rmax) ≈ p0
    if rmin + 2 <= rmax
        @test map(Base.Fix1(pdf, d), rmin+1:rmax-1) ≈ p0[2:end-1]
    end

    if isbounded(d)
        @test map(Base.Fix1(pdf, d), support(d)) ≈ p0
        @test map(Base.Fix1(pdf, d), rmin-2:rmax) ≈ vcat(0.0, 0.0, p0)
        @test map(Base.Fix1(pdf, d), rmin:rmax+3) ≈ vcat(p0, 0.0, 0.0, 0.0)
        @test map(Base.Fix1(pdf, d), rmin-2:rmax+3) ≈ vcat(0.0, 0.0, p0, 0.0, 0.0, 0.0)
    elseif islowerbounded(d)
        @test map(Base.Fix1(pdf, d), rmin-2:rmax) ≈ vcat(0.0, 0.0, p0)
    end
end


function test_evaluation(d::DiscreteUnivariateDistribution, vs::AbstractVector, testquan::Bool=true)
    nv  = length(vs)
    p   = Vector{Float64}(undef, nv)
    c   = Vector{Float64}(undef, nv)
    cc  = Vector{Float64}(undef, nv)
    lp  = Vector{Float64}(undef, nv)
    lc  = Vector{Float64}(undef, nv)
    lcc = Vector{Float64}(undef, nv)
    ci  = 0.

    for (i, v) in enumerate(vs)
        p[i] = pdf(d, v)
        c[i] = cdf(d, v)
        cc[i] = ccdf(d, v)
        lp[i] = logpdf(d, v)
        lc[i] = logcdf(d, v)
        lcc[i] = logccdf(d, v)

        @assert p[i] >= 0.0
        @assert (i == 1 || c[i] >= c[i-1])

        ci += p[i]
        @test ci ≈ c[i]
        @test isapprox(c[i] + cc[i], 1.0       , atol=1.0e-12)
        @test isapprox(lp[i]       , log(p[i]) , atol=1.0e-12)
        @test isapprox(lc[i]       , log(c[i]) , atol=1.0e-12)
        @test isapprox(lcc[i]      , log(cc[i]), atol=1.0e-12)

        if testquan
            ep = 1.0e-8
            if p[i] > 2 * ep   # ensure p[i] is large enough to guarantee a reliable result
                @test quantile(d, c[i] - ep) == v
                @test cquantile(d, cc[i] + ep) == v
                @test invlogcdf(d, lc[i] - ep) == v
                if 0.0 < c[i] < 1.0
                    @test invlogccdf(d, lcc[i] + ep) == v
                end
            end
        end
    end

    # consistency of scalar-based and vectorized evaluation
    @test Base.Fix1(pdf, d).(vs)  ≈ p
    @test Base.Fix1(cdf, d).(vs)  ≈ c
    @test Base.Fix1(ccdf, d).(vs) ≈ cc

    @test Base.Fix1(logpdf, d).(vs)  ≈ lp
    @test Base.Fix1(logcdf, d).(vs)  ≈ lc
    @test Base.Fix1(logccdf, d).(vs) ≈ lcc
end


function test_evaluation(d::ContinuousUnivariateDistribution, vs::AbstractVector, testquan::Bool=true)
    nv  = length(vs)
    p   = Vector{Float64}(undef, nv)
    c   = Vector{Float64}(undef, nv)
    cc  = Vector{Float64}(undef, nv)
    lp  = Vector{Float64}(undef, nv)
    lc  = Vector{Float64}(undef, nv)
    lcc = Vector{Float64}(undef, nv)

    for (i, v) in enumerate(vs)
        if !isa(d, StudentizedRange)
            p[i] = pdf(d, v)
            lp[i] = logpdf(d, v)
            @assert p[i] >= 0.0
        end

        c[i] = cdf(d, v)
        cc[i] = ccdf(d, v)
        lc[i] = logcdf(d, v)
        lcc[i] = logccdf(d, v)

        @assert (i == 1 || c[i] >= c[i-1])

        @test isapprox(c[i] + cc[i], 1.0       , atol=1.0e-12)
        if !isa(d, StudentizedRange)
            @test isapprox(lp[i]       , log(p[i]) , atol=1.0e-12)
        end
        @test isapprox(lc[i]       , log(c[i]) , atol=1.0e-12)
        @test isapprox(lcc[i]      , log(cc[i]), atol=1.0e-12)

        if testquan
            # TODO: remove this line when we have more accurate implementation
            # of quantile for InverseGaussian and StudentizedRange
            qtol = isa(d, InverseGaussian) ? 1.0e-4 : 1.0e-10
            qtol = isa(d, StudentizedRange) ? 1.0e-5 : qtol
            if p[i] > 1.0e-6
                @test isapprox(quantile(d, c[i])    , v, atol=qtol * (abs(v) + 1.0))
                @test isapprox(cquantile(d, cc[i])  , v, atol=qtol * (abs(v) + 1.0))
                @test isapprox(invlogcdf(d, lc[i])  , v, atol=qtol * (abs(v) + 1.0))
                @test isapprox(invlogccdf(d, lcc[i]), v, atol=qtol * (abs(v) + 1.0))
            end
        end
    end

    if !isa(d, StudentizedRange)
        # check: pdf should be the derivative of cdf
        for i = 2:(nv-1)
            if p[i] > 1.0e-6
                v = vs[i]
                ap = (cdf(d, v + 1.0e-6) - cdf(d, v - 1.0e-6)) / (2.0e-6)
                @test isapprox(p[i], ap, atol=p[i] * 1.0e-3)
            end
        end
    end

    # consistency of scalar-based and vectorized evaluation
    if !isa(d, StudentizedRange)
        @test Base.Fix1(pdf, d).(vs) ≈ p
        @test Base.Fix1(logpdf, d).(vs) ≈ lp
    end

    @test Base.Fix1(cdf, d).(vs)  ≈ c
    @test Base.Fix1(ccdf, d).(vs) ≈ cc

    @test Base.Fix1(logcdf, d).(vs)  ≈ lc
    @test Base.Fix1(logccdf, d).(vs) ≈ lcc
end

function test_nonfinite(distr::UnivariateDistribution)
    # non-finite bounds
    @test iszero(@inferred(cdf(distr, -Inf)))
    @test isone(@inferred(cdf(distr, Inf)))
    @test isone(@inferred(ccdf(distr, -Inf)))
    @test iszero(@inferred(ccdf(distr, Inf)))
    @test @inferred(logcdf(distr, -Inf)) == -Inf
    @test iszero(@inferred(logcdf(distr, Inf)))
    @test iszero(@inferred(logccdf(distr, -Inf)))
    @test @inferred(logccdf(distr, Inf)) == -Inf

    # NaN
    for f in (cdf, ccdf, logcdf, logccdf)
        if distr isa NoncentralT
            # broken in StatsFuns/R
            @test_broken isnan(f(distr, NaN))
        else
            @test isnan(f(distr, NaN))
        end
    end
end

#### Testing statistics methods

function test_stats(d::DiscreteUnivariateDistribution, vs::AbstractVector)
    # using definition (or an approximation)

    vf = Float64[v for v in vs]
    p = Base.Fix1(pdf, d).(vf)
    xmean = dot(p, vf)
    xvar = dot(p, abs2.(vf .- xmean))
    xstd = sqrt(xvar)
    xentropy = entropy(p)
    xskew = dot(p, (vf .- xmean).^3) / (xstd.^3)
    xkurt = dot(p, (vf .- xmean).^4) / (xvar.^2) - 3.0

    if isbounded(d)
        @test isapprox(mean(d), xmean, atol=1.0e-8)
        @test isapprox(var(d) , xvar , atol=1.0e-8)
        @test isapprox(std(d) , xstd , atol=1.0e-8)

        if applicable(skewness, d) && isfinite(skewness(d))
            @test isapprox(skewness(d), xskew   , atol=1.0e-8)
        end
        if applicable(kurtosis, d) && isfinite(kurtosis(d))
            @test isapprox(kurtosis(d), xkurt   , atol=1.0e-8)
        end
        if applicable(entropy, d)
            @test isapprox(entropy(d), xentropy, atol=1.0e-8)
        end
    else
        @test isapprox(mean(d), xmean, atol=1.0e-3 * (abs(xmean) + 1.0))
        @test isapprox(var(d) , xvar , atol=0.01 * xvar)
        @test isapprox(std(d) , xstd , atol=0.01 * xstd)
    end
end


allow_test_stats(d::UnivariateDistribution) = true
allow_test_stats(d::NoncentralBeta) = false
allow_test_stats(::StudentizedRange) = false
allow_test_stats(::LogitNormal) = false # `mean` is not defined since it has no analytical solution

function test_stats(d::ContinuousUnivariateDistribution, xs::AbstractVector{Float64})
    # using Monte Carlo methods

    if !(isfinite(mean(d)) && isfinite(var(d)))
        return
    end
    vd = var(d)

    n = length(xs)
    xmean = mean(xs)
    xvar = var(xs)
    xstd = sqrt(xvar)

    # we utilize central limit theorem, and consider xmean as (approximately) normally
    # distributed with std = std(d) / sqrt(n)
    #
    # For a normal distribution, it is extremely rare for a sample to deviate more
    # than 5 * std.dev, (chance < 6.0e-7)
    mean_tol = 5.0 * (sqrt(vd / n))
    @test isapprox(mean(d), xmean, atol=mean_tol)

    # test variance
    if applicable(kurtosis, d)
        kd = kurtosis(d)
        # when the excessive kurtosis is sufficiently large (i.e. > 2)
        # the sample variance has a finite variance, given by
        #
        #   (sigma^4 / n) * (k + 3 - (n-3)/(n-1))
        #
        # where k is the excessive kurtosis
        #
        if isfinite(kd) && kd > -2.0
            @test isapprox(var(d), xvar, atol=5.0 * vd * (kd + 2) / sqrt(n))
        end
    end
end

function test_params(d::Distribution)
    # simply test that params returns something sufficient to
    # reconstruct d
    D = typeof(d)
    pars = params(d)
    d_new = D(pars...)
    @test d_new == d
    @test d_new == deepcopy(d)
end

function test_params(d::Truncated)
    # simply test that params returns something sufficient to
    # reconstruct d
    d_unt = d.untruncated
    D = typeof(d_unt)
    pars = params(d_unt)
    d_new = truncated(D(pars...), d.lower, d.upper)
    @test d_new == d
    @test d == deepcopy(d)
end

# Finite difference differentiation
function fdm(f, at)
    map(1:length(at)) do i
        FiniteDifferences.central_fdm(5, 1)(x -> f([at[1:i-1]; x; at[i+1:end]]), at[i])
    end
end

# Equivalent to `ExactOneSampleKSTest` in HypothesisTests.jl
# We implement it here to avoid a circular dependency on HypothesisTests
# that causes test failures when preparing a breaking release of Distributions
function pvalue_kolmogorovsmirnoff(x::AbstractVector, d::UnivariateDistribution)
    # compute maximum absolute deviation from the empirical cdf
    n = length(x)
    cdfs = sort!(map(Base.Fix1(cdf, d), x))
    dmax = maximum(zip(cdfs, (0:(n-1))/n, (1:n)/n)) do (cdf, lower, upper)
        return max(cdf - lower, upper - cdf)
    end

    # compute asymptotic p-value (see `KSDist`)
    return ccdf(KSDist(n), dmax)
end

function test_affine_transformations(::Type{T}, params...) where {T<:UnivariateDistribution}
    @testset "affine transformations ($T)" begin
        # distribution
        d = T(params...)

        # random shift and scale
        c = randn()

        # addition
        for shift_d in (@inferred(d + c), @inferred(c + d))
            @test shift_d isa T
            @test location(shift_d) ≈ location(d) + c
            @test scale(shift_d) ≈ scale(d)
        end

        # multiplication (negative and positive values)
        for s in (-c, c)
            for scale_d in (@inferred(s * d), @inferred(d * s), @inferred(d / inv(s)))
                @test scale_d isa T
                if d isa Uniform
                    @test location(scale_d) ≈ (s > 0 ? s * minimum(d) : s * maximum(d))
                else
                    @test location(scale_d) ≈ s * location(d)
                end
                @test scale(scale_d) ≈ abs(s) * scale(d)
            end
        end
    end
end
