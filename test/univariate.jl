# Need to include probabilistic tests.
# Allowing slow tests is defensible to ensure correctness.

n_samples = 5_000_000

# Test default instances

for d in [Beta(2.0, 2.0),
          Cauchy(0.0, 1.0),
          Chisq(12),
          Exponential(1.0),
          Exponential(5.1),
          # FDist(2, 21), # Entropy wrong
          Gamma(3.0),
          Gamma(3.0, 3.0),
          Gumbel(),
          Gumbel(5, 3), 
          Logistic(),
          logNormal(0.0, 1.0),
          Normal(0.0, 1.0),
          #TDist(1), # Entropy wrong
          #TDist(28), # Entropy wrong
          TruncatedNormal(Normal(0, 1), -3, 3),
          # TruncatedNormal(Normal(-100, 1), 0, 1),
          TruncatedNormal(Normal(27, 3), 0, Inf),
          Uniform(0.0, 1.0),
          Weibull(2.3)]

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
    if !isnan(mu)
        @assert norm(mu - mu_hat, Inf) < 1e-1
    end

    # Because of the Weak Law of Large Numbers,
    #  empirical covariance matrix should be close to theoretical value
    sigma, sigma_hat = var(d), var(X)
    if !isnan(mu)
        @assert norm(sigma - sigma_hat, Inf) < 1e-1
    end

    # TODO: Test cov and cor for multivariate distributions
    # TODO: Decide how var(d::MultivariateDistribution) should be defined

    # By the Asymptotic Equipartition Property,
    #  empirical mean negative log PDF should be close to theoretical value
    ent, ent_hat = entropy(d), -mean(logpdf(d, X))
    if !isnan(mu)
        @assert norm(ent - ent_hat, Inf) < 1e-1
    end

    # TODO: Test logpdf!()

    # Test non-logged PDF
    ent, ent_hat = entropy(d), -mean(log(pdf(d, X)))
    if !isnan(mu)
        @assert norm(ent - ent_hat, Inf) < 1e-1
    end

    # TODO: Test pdf!()

    # TODO: Test independence of draws?

    # TODO: Test cdf, quantile

    # TODO: Test mgf, cf

    # TODO: Test median
    # TODO: Test modes

    # TODO: Test kurtosis
    # TODO: Test skewness
end
