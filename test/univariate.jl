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
          # NegativeBinomial(),
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

    # Check that we can generate a single random draw
    draw = rand(d)

    # Check that draw satifies insupport()
    @test insupport(d, draw)

    # Check that we can generate many random draws at once
    x = rand(d, n_samples)

    # Check that sequence of draws satifies insupport()
    @test insupport(d, x)

    # Check that we can generate many random draws in-place
    rand!(d, x)

    mu, mu_hat = mean(d), mean(x)
    ent, ent_hat = entropy(d), -mean(logpdf(d, x))
    ent2, ent_hat2 = entropy(d), -mean(log(pdf(d, x)))
    m, m_hat = median(d), median(x)
    sigma, sigma_hat = var(d), var(x)

    # Check that KL between fitted distribution and true distribution
    #  is small
    # TODO: Restore the line below
    # d_hat = fit(typeof(d), x)
    # TODO: @test kl(d, d_hat) < 1e-2

    # Because of the Weak Law of Large Numbers,
    #  empirical mean should be close to theoretical value
    if isfinite(mu)
        if isfinite(sigma) && sigma > 0.0
            @test abs(mu - mu_hat) / sigma < 1e-0
        else
            @test abs(mu - mu_hat) < 1e-1
        end
    end

    # By the Asymptotic Equipartition Property,
    #  empirical mean negative log PDF should be close to theoretical value
    if isfinite(ent) && !isa(d, Arcsine)
        # @test norm(ent - ent_hat, Inf) < 1e-1
        if ent > 0.0
            @test abs(ent - ent_hat) / abs(ent) < 1e-0
        end
    end

    # TODO: Test logpdf!()

    # Test non-logged PDF
    if isfinite(ent2) && !isa(d, Arcsine)
        # @test norm(ent - ent_hat, Inf) < 1e-1
        if ent2 > 0.0
            @test abs(ent2 - ent_hat2) / abs(ent2) < 1e-0
        end
    end

    # TODO: Test pdf!()

    # TODO: Test independence of draws?

    # TODO: Test cdf, quantile
    if isa(d, ContinuousUnivariateDistribution)
        for p in 0.1:0.1:0.9
            @test abs(cdf(d, quantile(d, p)) - p) < 1e-8
        end
    end

    # TODO: Test mgf, cf

    # TODO: Test median
    if insupport(d, m_hat) && isa(d, ContinuousDistribution)
        if isa(d, Cauchy) || isa(d, Laplace)
            @test abs(m - m_hat) / d.scale < 1e-0
        elseif isa(d, FDist)
            println("Skipping median test for FDist")
        else
            # @test norm(m - m_hat, Inf) < 1e-1
            @test abs(m - m_hat) / sigma < 1e-0
        end
    end

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

    # Bail on higher moments for LogNormal distribution or
    # truncated distributions
    if isa(d, LogNormal) || isa(d, TruncatedUnivariateDistribution)
        continue
    end

    sk, sk_hat = skewness(d), skewness(x)
    k, k_hat = kurtosis(d), kurtosis(x)

    # Because of the Weak Law of Large Numbers,
    #  empirical covariance matrix should be close to theoretical value
    if isfinite(mu) && isfinite(sigma)
        if sigma > 0.0
            @test abs(sigma - sigma_hat) / sigma < 1e-0
        else
            @test abs(sigma - sigma_hat) < 1e-1
        end
    end

    # TODO: Test cov and cor for multivariate distributions
    # TODO: Decide how var(d::MultivariateDistribution) should be defined

    # Because of the Weak Law of Large Numbers,
    #  empirical skewness should be close to theoretical value
    if isfinite(mu) && isfinite(sk)
        if sk > 0.0
            @test abs(sk - sk_hat) / sigma < 1e-0
        else
            @test abs(sk - sk_hat) < 1e-1
        end
    end

    # Empirical kurtosis is very unstable for FDist
    if isa(d, FDist)
        continue
    end

    if isfinite(mu) && isfinite(k)
        if k > 0.0
            @test abs(k - k_hat) / abs(k) < 1e-0
        else
            @test abs(k - k_hat) < 1e-1
        end
    end
end
