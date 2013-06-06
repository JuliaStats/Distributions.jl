# Need to include probabilistic tests.
# Allowing slow tests is defensible to ensure correctness.

n_samples = 5_000_001

# Test default instances

for d in [Arcsine(),
          Bernoulli(0.5),
          Beta(2.0, 2.0),
          # BetaPrime(2.0, 2.0),
          Binomial(1, 0.5),
          Categorical([0.5, 0.5]),
          Cauchy(0.0, 1.0),
          # Chi(12),
          Chisq(12),
          # Cosine(),
          DiscreteUniform(2, 5),
          # Empirical(),
          Erlang(1),
          Exponential(1.0),
          Exponential(5.1),
          # FDist(2, 21), # Entropy wrong
          Gamma(3.0),
          Gamma(3.0, 3.0),
          Geometric(),
          Gumbel(),
          Gumbel(5, 3), 
          # Hypergeometric(),
          # InvertedGamma(),
          Laplace(0.0, 1.0),
          Levy(),
          Logistic(),
          logNormal(0.0, 1.0),
          # NegativeBinomial(),
          # NoncentralBeta(),
          # NoncentralChisq(),
          # NoncentralFDist(),
          # NoncentralTDist(),
          Normal(0.0, 1.0),
          # Pareto(),
          Poisson(10.0),
          Rayleigh(10.0),
          # Skellam(10.0, 2.0), # Entropy wrong
          # TDist(1), # Entropy wrong
          # TDist(28), # Entropy wrong
          Triangular(3.0, 2.0),
          TruncatedNormal(Normal(0, 1), -3, 3),
          # TruncatedNormal(Normal(-100, 1), 0, 1),
          TruncatedNormal(Normal(27, 3), 0, Inf),
          Uniform(0.0, 1.0),
          Weibull(2.3)]

    # NB: Uncomment if test fails
    # Mention distribution being run
    # println(d)

    # Check that we can generate a single random draw
    draw = rand(d)

    # Check that draw satifies insupport()
    @assert insupport(d, draw)

    # Check that we can generate many random draws at once
    X = rand(d, n_samples)

    # Check that sequence of draws satifies insupport()
    @assert insupport(d, X)

    # Check that we can generate many random draws in-place
    rand!(d, X)

    # Check that KL between fitted distribution and true distribution
    #  is small
    # TODO: Restore the line below
    # d_hat = fit(typeof(d), X)
    # TODO: @assert kl(d, d_hat) < 1e-2

    # Because of the Weak Law of Large Numbers,
    #  empirical mean should be close to theoretical value
    mu, mu_hat = mean(d), mean(X)
    if isfinite(mu)
        @assert norm(mu - mu_hat, Inf) < 1e-1
    end

    # Because of the Weak Law of Large Numbers,
    #  empirical covariance matrix should be close to theoretical value
    sigma, sigma_hat = var(d), var(X)
    if isfinite(mu)
        @assert norm(sigma - sigma_hat, Inf) < 1e-1
    end

    # TODO: Test cov and cor for multivariate distributions
    # TODO: Decide how var(d::MultivariateDistribution) should be defined

    # By the Asymptotic Equipartition Property,
    #  empirical mean negative log PDF should be close to theoretical value
    ent, ent_hat = entropy(d), -mean(logpdf(d, X))
    if isfinite(mu)
        @assert norm(ent - ent_hat, Inf) < 1e-1
    end

    # TODO: Test logpdf!()

    # Test non-logged PDF
    ent, ent_hat = entropy(d), -mean(log(pdf(d, X)))
    if isfinite(mu)
        @assert norm(ent - ent_hat, Inf) < 1e-1
    end

    # TODO: Test pdf!()

    # TODO: Test independence of draws?

    # TODO: Test cdf, quantile

    # TODO: Test mgf, cf

    # TODO: Test median
    m, m_hat = median(d), median(X)
    if insupport(d, m_hat) && isa(d, ContinuousDistribution)
        @assert norm(m - m_hat, Inf) < 1e-1
    end

    # Test modes by looking at pdf(x +/- eps()) near a mode x
    try
        ms = modes(d)
        if isa(d, ContinuousUnivariateDistribution)
            @assert pdf(d, ms[1]) > pdf(d, ms[1] + eps(ms[1]))
            @assert pdf(d, ms[1]) > pdf(d, ms[1] - eps(ms[1]))
        end
    catch
        @printf "Skipping %s\n" typeof(d)
    end

    # Bail on higher moments for LogNormal distribution or
    # truncated distributions
    if isa(d, logNormal) || isa(d, TruncatedUnivariateDistribution)
        continue
    end

    # Because of the Weak Law of Large Numbers,
    #  empirical skewness should be close to theoretical value
    sk, sk_hat = skewness(d), skewness(X)
    if isfinite(mu)
        @assert norm(sk - sk_hat, Inf) < 1e-1
    end

    # Because of the Weak Law of Large Numbers,
    #  empirical excess kurtosis should be close to theoretical value
    k, k_hat = kurtosis(d), kurtosis(X)
    if isfinite(mu)
        @assert norm(k - k_hat, Inf) < 1e-1
    end
end
