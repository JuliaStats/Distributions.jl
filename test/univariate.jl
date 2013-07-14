# This test suite exploits the Weak Law of Large Numbers
# to verify that all of the functions defined on a
# distribution produce values within a close range of
# their theoretically predicted values.
#
# This includes tests of means, variances, skewness and kurtosis;
# as well as tests of more complex quantities like quantiles,
# entropy, etc...
#
# These tests are quite slow, but are essential to verifying the
# accuracy of our distributions

using Distributions
using Base.Test

probpts(n::Int) = ((1:n) - 0.5)/n  
const pp  = float(probpts(100))  
const lpp = log(pp)

# Use a large, odd number of samples for testing all quantities
n_samples = 5_000_001

# Try out many parameterizations of any given distribution
for d in [Arcsine(),
          Bernoulli(0.1),
          Bernoulli(0.5),
          Bernoulli(0.9),
          Beta(2.0, 2.0),
          Beta(3.0, 4.0),
          Beta(17.0, 13.0),
          # BetaPrime(3.0, 3.0),
          # BetaPrime(3.0, 5.0),
          # BetaPrime(5.0, 3.0),
          Binomial(1, 0.5),
          Binomial(100, 0.1),
          Binomial(100, 0.9),
          Categorical([0.1, 0.9]),
          Categorical([0.5, 0.5]),
          Categorical([0.9, 0.1]),
          Cauchy(0.0, 1.0),
          Cauchy(10.0, 1.0),
          Cauchy(0.0, 10.0),
          # Chi(12),
          Chisq(8),
          Chisq(12.0),
          Chisq(20.0),
          # Cosine(),
          DiscreteUniform(0, 3),
          DiscreteUniform(2.0, 5.0),
          # Empirical(),
          Erlang(1),
          Erlang(17.0),
          Exponential(1.0),
          Exponential(5.1),
          FDist(9, 9),
          FDist(9, 21),
          FDist(21, 9),
          Gamma(3.0, 2.0),
          Gamma(2.0, 3.0),
          Gamma(3.0, 3.0),
          Geometric(0.1),
          Geometric(0.5),
          Geometric(0.9),
          Gumbel(3.0, 5.0),
          Gumbel(5, 3),
          # HyperGeometric(1.0, 1.0, 1.0),
          # HyperGeometric(2.0, 2.0, 2.0),
          # HyperGeometric(3.0, 2.0, 2.0),
          # HyperGeometric(2.0, 3.0, 2.0),
          # HyperGeometric(2.0, 2.0, 3.0),
          # InvertedGamma(),
          Laplace(0.0, 1.0),
          Laplace(10.0, 1.0),
          Laplace(0.0, 10.0),
          Levy(0.0, 1.0),
          Levy(2.0, 8.0),
          Levy(3.0, 3.0),
          Logistic(0.0, 1.0),
          Logistic(10.0, 1.0),
          Logistic(0.0, 10.0),
          LogNormal(0.0, 1.0),
          LogNormal(10.0, 1.0),
          LogNormal(0.0, 10.0),
          NegativeBinomial(),
          NegativeBinomial(5, 0.6),
          # NoncentralBeta(),
          # NoncentralChisq(),
          # NoncentralFDist(),
          # NoncentralTDist(),
          Normal(0.0, 1.0),
          Normal(-1.0, 10.0),
          Normal(1.0, 10.0),
          # Pareto(),
          Poisson(2.0),
          Poisson(10.0),
          Poisson(51.0),
          Rayleigh(1.0),
          Rayleigh(5.0),
          Rayleigh(10.0),
          # Skellam(10.0, 2.0), # Entropy wrong
          # TDist(1), # Entropy wrong
          # TDist(28), # Entropy wrong
          Triangular(3.0, 1.0),
          Triangular(3.0, 2.0),
          Triangular(10.0, 10.0),
          TruncatedNormal(Normal(0, 1), -3, 3),
          # TruncatedNormal(Normal(-100, 1), 0, 1),
          TruncatedNormal(Normal(27, 3), 0, Inf),
          Uniform(0.0, 1.0),
          Uniform(3.0, 17.0),
          Uniform(3.0, 3.1),
          Weibull(2.3),
          Weibull(23.0),
          Weibull(230.0)]

    # NB: Uncomment if test fails
    # Mention distribution being run
    # println(d)

    n = length(pp)
    is_continuous = isa(d, ContinuousDistribution)
    is_discrete = isa(d, DiscreteDistribution)

    @assert is_continuous == !is_discrete
    sample_ty = is_continuous ? Float64 : Int

    # avoid checking high order moments for LogNormal and Logistic
    avoid_highord = isa(d, LogNormal) || isa(d, Logistic) || isa(d, TruncatedNormal)

    #####
    #
    #  Part 1: Capability of random number generation
    #
    #####

    # check that we can generate a single random draw
    draw = rand(d)

    # check that draw satifies insupport()
    @test insupport(d, draw)

    # check that we can generate many random draws at once
    x = rand(d, n)

    # check that sequence of draws satifies insupport()
    @test insupport(d, x)

    # check that we can generate many random draws in-place
    rand!(d, x)

    ##### 
    #
    #  Part 2: Evaluation 
    #  ----------------------
    #
    #  This part tests the integrity/consistency of following functions:
    #
    #  - pdf
    #  - cdf
    #  - ccdf
    #
    #  - logpdf
    #  - logcdf
    #  - logccdf
    #
    #  - quantile
    #  - cquantile
    #  - invlogcdf
    #  - invlogccdf
    #
    #####

    # evaluate by scalar

    x = zeros(sample_ty, n)
    r_cquan = zeros(sample_ty, n)

    r_pdf = zeros(n)
    r_cdf = zeros(n)
    r_ccdf = zeros(n)

    r_logpdf = zeros(n)
    r_logcdf = zeros(n)
    r_logccdf = zeros(n)

    r_invlogcdf = zeros(n)
    r_invlogccdf = zeros(n)

    for i in 1:n
        x[i] = quantile(d, pp[i])
        r_cquan[i] = cquantile(d, pp[i])

        xi = x[i]
        r_pdf[i] = pdf(d, xi)     
        r_cdf[i] = cdf(d, xi)
        r_ccdf[i] = ccdf(d, xi)

        r_logpdf[i] = logpdf(d, xi)
        r_logcdf[i] = logcdf(d, xi)
        r_logccdf[i] = logccdf(d, xi)

        r_invlogcdf[i] = invlogcdf(d, lpp[i])
        r_invlogccdf[i] = invlogccdf(d, lpp[i])
    end

    # testing consistency between scalar evaluation and vectorized evaluation

    for i in 1:length(x)
        @test_approx_eq quantile(d, pp)  x
        @test_approx_eq cquantile(d, pp) r_cquan
        @test_approx_eq pdf(d, x)        r_pdf
        @test_approx_eq cdf(d, x)        r_cdf
        @test_approx_eq ccdf(d, x)       r_ccdf
        @test_approx_eq logpdf(d, x)     r_logpdf
        @test_approx_eq logcdf(d, x)     r_logcdf
        @test_approx_eq logccdf(d, x)    r_logccdf

        @test_approx_eq invlogcdf(d, lpp)  r_invlogcdf
        @test_approx_eq invlogccdf(d, lpp) r_invlogccdf
    end

    # # testing consistency between different functions

    @test_approx_eq logpdf(d, x) log(pdf(d, x))
    @test_approx_eq cquantile(d, 1 - pp) x

    if is_continuous
        @test_approx_eq cdf(d, x) pp
        @test_approx_eq ccdf(d, x) 1 - pp
        @test_approx_eq logcdf(d, x) lpp
        @test_approx_eq logccdf(d, x) lpp[end:-1:1]
        @test_approx_eq invlogcdf(d, lpp) x
        @test_approx_eq invlogccdf(d, lpp) x[end:-1:1]
    end

    # TODO: Test mgf, cf

    ##### 
    #
    #  Part 3: Other tests
    #
    #####

    # Test modes by looking at pdf(x +/- eps()) near a mode x

    if !isa(d, Uniform)
        ms = modes(d)
        if isa(d, ContinuousUnivariateDistribution)
            if insupport(d, ms[1] + 0.1)
                @test pdf(d, ms[1]) > pdf(d, ms[1] + 0.1)
            end
            if insupport(d, ms[1] - 0.1)
                @test pdf(d, ms[1]) > pdf(d, ms[1] - 0.1)
            end
        end
    end
end


