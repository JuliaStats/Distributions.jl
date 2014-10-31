# Utilities to support the testing of distributions and samplers


#### Testing sampleable objects (samplers)

# for discrete samplers
#
function test_samples(s::Sampleable{Univariate, Discrete},      # the sampleable instance
                      distr::DiscreteUnivariateDistribution,    # corresponding distribution
                      n::Int;                                   # number of samples to generate
                      q::Float64=1.0e-6,                        # confidence interval, 1 - q as confidence
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

    rmin = ifloor(quantile(distr, 0.00001))::Int
    rmax = ifloor(quantile(distr, 0.99999))::Int
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
        clb[i] = ifloor(quantile(bp, q/2))
        cub[i] = iceil(cquantile(bp, q/2))
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
end


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

    rmin = ifloor(quantile(distr, 0.01))::Int
    rmax = ifloor(quantile(distr, 0.99))::Int
    edges = linspace(rmin, rmax, nbins+1)

    # determine confidence intervals for counts:
    # with probability q, the count will be out of this interval.
    #
    clb = Array(Int, nbins)
    cub = Array(Int, nbins)
    cdfs = cdf(distr, edges)

    for i = 1:nbins
        bp = Binomial(n, cdfs[i+1] - cdfs[i])
        clb[i] = ifloor(quantile(bp, q/2))
        cub[i] = iceil(cquantile(bp, q/2))
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
    cnts = hist(samples, edges)[2]
    @assert length(cnts) == nbins

    # check the counts
    for i = 1:nbins
        if verbose
            @printf("[%.4f, %.4f) ==> (%d, %d): %d\n", edges[i], edges[i+1], clb[i], cub[i], cnts[i])
        end
        clb[i] <= cnts[i] <= cub[i] ||
            error("The counts are out of the confidence interval.")
    end
end


