# Utilities to support the testing of distributions and samplers

import Base.Test: @test, @test_approx_eq, @test_approx_eq_eps

# auxiliary functions

# to workaround issues of Base.linspace
function _linspace(a::Float64, b::Float64, n::Int)
    intv = (b - a) / (n - 1)
    r = Array(Float64, n)
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
function test_distr(distr::DiscreteUnivariateDistribution, n::Int)

    test_range(distr)
    vs = get_evalsamples(distr, 0.00001)

    test_support(distr, vs)
    test_evaluation(distr, vs)
    test_range_evaluation(distr)

    test_stats(distr, vs)
    test_samples(distr, n)
    test_params(distr)
end


# testing the implementation of a continuous univariate distribution
#
function test_distr(distr::ContinuousUnivariateDistribution, n::Int)

    test_range(distr)
    vs = get_evalsamples(distr, 0.01, 2000)

    test_support(distr, vs)
    test_evaluation(distr, vs)

    xs = test_samples(distr, n)
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
                      verbose::Bool=false)                      # show intermediate info (for debugging)

    # The basic idea
    # ------------------
    #   Generate n samples, and count the occurences of each value within a reasonable range.
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
    p0 = pdf(distr, rmin:rmax)  # reference probability masses
    @assert length(p0) == m

    # determine confidence intervals for counts:
    # with probability q, the count will be out of this interval.
    #
    clb = Array(Int, m)
    cub = Array(Int, m)
    for i = 1:m
        bp = Binomial(n, p0[i])
        clb[i] = floor(Int,quantile(bp, q/2))
        cub[i] = ceil(Int,cquantile(bp, q/2))
        @assert cub[i] >= clb[i]
    end

    # generate samples
    samples = rand(s, n)
    @assert length(samples) == n

    # scan samples and get counts
    cnts = zeros(Int, m)
    for i = 1:n
        @inbounds si = samples[i]
        if rmin <= si <= rmax
            cnts[si - rmin + 1] += 1
        else
            vmin <= si <= vmax ||
                error("Sample value out of valid range.")
        end
    end

    # check the counts
    for i = 1:m
        verbose && println("v = $(rmin+i-1) ==> ($(clb[i]), $(cub[i])): $(cnts[i])")
        clb[i] <= cnts[i] <= cub[i] ||
            error("The counts are out of the confidence interval.")
    end
    return samples
end

test_samples(distr::DiscreteUnivariateDistribution, n::Int; q::Float64=1.0e-6, verbose::Bool=false) =
    test_samples(distr, distr, n; q=q, verbose=verbose)


# for continuous samplers
#
function test_samples(s::Sampleable{Univariate, Continuous},    # the sampleable instance
                      distr::ContinuousUnivariateDistribution,  # corresponding distribution
                      n::Int;                                   # number of samples to generate
                      nbins::Int=50,                            # divide the main interval into nbins
                      q::Float64=1.0e-6,                        # confidence interval, 1 - q as confidence
                      verbose::Bool=false)                      # show intermediate info (for debugging)

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
    clb = Array(Int, nbins)
    cub = Array(Int, nbins)
    cdfs = cdf(distr, edges)

    for i = 1:nbins
        pi = cdfs[i+1] - cdfs[i]
        bp = Binomial(n, pi)
        clb[i] = floor(Int,quantile(bp, q/2))
        cub[i] = ceil(Int,cquantile(bp, q/2))
        @assert cub[i] >= clb[i]
    end

    # generate samples
    samples = rand(s, n)
    @assert length(samples) == n

    # check whether all samples are in the valid range
    for i = 1:n
        @inbounds si = samples[i]
        vmin <= si <= vmax ||
            error("Sample value out of valid range.")
    end

    # get counts
    cnts = fit(Histogram, samples, edges).weights
    @assert length(cnts) == nbins

    # check the counts
    for i = 1:nbins
        if verbose
            @printf("[%.4f, %.4f) ==> (%d, %d): %d\n", edges[i], edges[i+1], clb[i], cub[i], cnts[i])
        end
        clb[i] <= cnts[i] <= cub[i] ||
            error("The counts are out of the confidence interval.")
    end
    return samples
end

test_samples(distr::ContinuousUnivariateDistribution, n::Int; nbins::Int=50, q::Float64=1.0e-6, verbose::Bool=false) =
    test_samples(distr, distr, n; nbins=nbins, q=q, verbose=verbose)


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

    lv = (islowerbounded(d) ? minimum(d) : floor(Int,quantile(d, q/2)))::Int
    hv = (isupperbounded(d) ? maximum(d) : ceil(Int,cquantile(d, q/2)))::Int
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

    if isbounded(d)
        if isa(d, DiscreteUnivariateDistribution)
            s = support(d)
            @test isa(s, UnitRange)
            @test first(s) == minimum(d)
            @test last(s) == maximum(d)
        end
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

    p0 = pdf(d, collect(rmin:rmax))
    @test_approx_eq pdf(d, rmin:rmax) p0
    if rmin + 2 <= rmax
        @test_approx_eq pdf(d, rmin+1:rmax-1) p0[2:end-1]
    end

    if isbounded(d)
        @test_approx_eq pdf(d) p0
        @test_approx_eq pdf(d, rmin-2:rmax) vcat(0.0, 0.0, p0)
        @test_approx_eq pdf(d, rmin:rmax+3) vcat(p0, 0.0, 0.0, 0.0)
        @test_approx_eq pdf(d, rmin-2:rmax+3) vcat(0.0, 0.0, p0, 0.0, 0.0, 0.0)
    elseif islowerbounded(d)
        @test_approx_eq pdf(d, rmin-2:rmax) vcat(0.0, 0.0, p0)
    end
end


function test_evaluation(d::DiscreteUnivariateDistribution, vs::AbstractVector)
    nv = length(vs)
    p = Array(Float64, nv)
    c = Array(Float64, nv)
    cc = Array(Float64, nv)
    lp = Array(Float64, nv)
    lc = Array(Float64, nv)
    lcc = Array(Float64, nv)
    ci = 0.

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
        @test_approx_eq ci c[i]
        @test_approx_eq_eps c[i] + cc[i] 1.0 1.0e-12
        @test_approx_eq_eps lp[i] log(p[i]) 1.0e-12
        @test_approx_eq_eps lc[i] log(c[i]) 1.0e-12
        @test_approx_eq_eps lcc[i] log(cc[i]) 1.0e-12

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

    # consistency of scalar-based and vectorized evaluation
    @test_approx_eq pdf(d, vs) p
    @test_approx_eq cdf(d, vs) c
    @test_approx_eq ccdf(d, vs) cc

    @test_approx_eq logpdf(d, vs) lp
    @test_approx_eq logcdf(d, vs) lc
    @test_approx_eq logccdf(d, vs) lcc
end


function test_evaluation(d::ContinuousUnivariateDistribution, vs::AbstractVector)
    nv = length(vs)
    p = Array(Float64, nv)
    c = Array(Float64, nv)
    cc = Array(Float64, nv)
    lp = Array(Float64, nv)
    lc = Array(Float64, nv)
    lcc = Array(Float64, nv)

    for (i, v) in enumerate(vs)
        p[i] = pdf(d, v)
        c[i] = cdf(d, v)
        cc[i] = ccdf(d, v)
        lp[i] = logpdf(d, v)
        lc[i] = logcdf(d, v)
        lcc[i] = logccdf(d, v)

        @assert p[i] >= 0.0
        @assert (i == 1 || c[i] >= c[i-1])

        @test_approx_eq_eps c[i] + cc[i] 1.0 1.0e-12
        @test_approx_eq_eps lp[i] log(p[i]) 1.0e-12
        @test_approx_eq_eps lc[i] log(c[i]) 1.0e-12
        @test_approx_eq_eps lcc[i] log(cc[i]) 1.0e-12

        # TODO: remove this line when we have more accurate implementation
        # of quantile for InverseGaussian
        qtol = isa(d, InverseGaussian) ? 1.0e-4 : 1.0e-10
        if p[i] > 1.0e-6
            @test_approx_eq_eps quantile(d, c[i]) v qtol * (abs(v) + 1.0)
            @test_approx_eq_eps cquantile(d, cc[i]) v qtol * (abs(v) + 1.0)
            @test_approx_eq_eps invlogcdf(d, lc[i]) v qtol * (abs(v) + 1.0)
            @test_approx_eq_eps invlogccdf(d, lcc[i]) v qtol * (abs(v) + 1.0)
        end
    end

    # check: pdf should be the derivative of cdf
    for i = 2:(nv-1)
        if p[i] > 1.0e-6
            v = vs[i]
            ap = (cdf(d, v + 1.0e-6) - cdf(d, v - 1.0e-6)) / (2.0e-6)
            @test_approx_eq_eps p[i] ap p[i] * 1.0e-3
        end
    end

    # consistency of scalar-based and vectorized evaluation
    @test_approx_eq pdf(d, vs) p
    @test_approx_eq cdf(d, vs) c
    @test_approx_eq ccdf(d, vs) cc

    @test_approx_eq logpdf(d, vs) lp
    @test_approx_eq logcdf(d, vs) lc
    @test_approx_eq logccdf(d, vs) lcc
end


#### Testing statistics methods

function test_stats(d::DiscreteUnivariateDistribution, vs::AbstractVector)
    # using definition (or an approximation)

    vf = Float64[v for v in vs]
    p = pdf(d, vf)
    xmean = dot(p, vf)
    xvar = dot(p, @compat(abs2.(vf .- xmean)))
    xstd = sqrt(xvar)
    xentropy = entropy(p)
    xskew = dot(p, (vf .- xmean).^3) / (xstd.^3)
    xkurt = dot(p, (vf .- xmean).^4) / (xvar.^2) - 3.0

    if isbounded(d)
        @test_approx_eq_eps mean(d)  xmean  1.0e-8
        @test_approx_eq_eps var(d)   xvar   1.0e-8
        @test_approx_eq_eps std(d)   xstd   1.0e-8

        if isfinite(skewness(d))
            @test_approx_eq_eps skewness(d) xskew 1.0e-8
        end
        if isfinite(kurtosis(d))
            @test_approx_eq_eps kurtosis(d) xkurt 1.0e-8
        end
        if applicable(entropy, d)
            @test_approx_eq_eps entropy(d) xentropy 1.0e-8
        end
    else
        @test_approx_eq_eps mean(d) xmean 1.0e-3 * (abs(xmean) + 1.0)
        @test_approx_eq_eps var(d)  xvar  0.01 * xvar
        @test_approx_eq_eps std(d)  xstd  0.01 * xstd
    end
end


allow_test_stats(d::UnivariateDistribution) = true
allow_test_stats(d::NoncentralBeta) = false

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
    @test_approx_eq_eps mean(d) xmean mean_tol

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
            @test_approx_eq_eps var(d) xvar 5.0 * vd * (kd + 2) / sqrt(n)
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
end

function test_params(d::Truncated)
    # simply test that params returns something sufficient to
    # reconstruct d
    d_unt = d.untruncated
    D = typeof(d_unt)
    pars = params(d_unt)
    d_new = Truncated(D(pars...), d.lower, d.upper)
    @test d_new == d
end
