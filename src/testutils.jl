# Utilities to support the testing of distributions and samplers


#### Testing sampleable objects (samplers)

function test_samples(s::Sampleable{Univariate, Discrete},      # the sampleable instance
                      vrange::UnitRange,                        # the range of sample values to examine
                      p0::Vector{Float64},                      # the reference probabilities over vrange
                      n::Int;                                   # number of samples to generate
                      lbound::Bool=true,                        # whether the samples are lower bounded by vrange[1]
                      ubound::Bool=true,                        # whether the samples are upper bounded by vrange[end]
                      q::Float64=1.0e-6,                        # confidence interval, 1 - q as confidence
                      verbose::Bool=false)                      # show intermediate info (for debugging)
    
    # The basic idea
    # ------------------
    #   Generate n samples, and count the occurences of each value.
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
    m = length(vrange)
    length(p0) == m || error("length(p0) does not match length(vrange).")
    0.0 < q < 0.1 || error("The value of q must be within the open interval (0.0, 0.1).")

    # determine confidence intervals for counts
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

    # scan samples and get counts
    vmin, vmax = extrema(vrange)
    cnts = zeros(Int, m)
    for i = 1:n
        @inbounds si = samples[i]
        if vmin <= si <= vmax
            cnts[si - vmin + 1] += 1
        else
            ((lbound && si < vmin) || (ubound && si > vmax)) && 
                error("Sample value out of expected range.")
        end
    end

    # check the counts
    for i = 1:m
        verbose && println("v = $(vrange[i]) ==> ($(clb[i]), $(cub[i])): $(cnts[i])")
        clb[i] <= cnts[i] <= cub[i] ||
            error("The counts are out of the confidence interval.")
    end
end

