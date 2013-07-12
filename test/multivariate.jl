# Need to include probabilistic tests.
# Allowing slow tests is defensible to ensure correctness.

n_samples = 1_000_000

for d in [Dirichlet([100.0, 17.0, 31.0, 45.0]),
          Multinomial(1, [1/3, 1/3, 1/3]),
          MultivariateNormal([0.0, 0.0], [1.0 0.9; 0.9 1.0])]
    # Check that we can generate a single random draw
    draw = rand(d)
    @assert length(draw) == dim(d)

    # Check that draw satifies insupport()
    @assert insupport(d, draw)

    # Check that we can generate many random draws at once
    X = rand(d, n_samples)
    @assert size(X, 1) == dim(d)

    # Check that sequence of draws satifies insupport()
    @assert insupport(d, X)

    # Check that we can generate many random draws in-place
    rand!(d, X)

    # Check that KL between fitted distribution and true distribution
    #  is small
    d_hat = fit(typeof(d), X)
    # TODO: @assert kl(d, d_hat) < 1e-2

    # Because of the Weak Law of Large Numbers,
    #  empirical mean should be close to theoretical value
    mu, mu_hat = mean(d), vec(mean(X, 2))
    @assert norm(mu - mu_hat, Inf) < 1e-2

    # Because of the Weak Law of Large Numbers,
    #  empirical covariance matrix should be close to theoretical value
    sigma, sigma_hat = cov(d), cov(X')
    @assert norm(sigma - sigma_hat, Inf) < 1e-2

    # TODO: Test cov and cor for multivariate distributions
    # TODO: Decide how var(d::MultivariateDistribution) should be defined

    # By the Asymptotic Equipartition Property,
    #  empirical mean negative log PDF should be close to theoretical value
    ent, ent_hat = entropy(d), -mean(logpdf(d, X))
    @assert norm(ent - ent_hat, Inf) < 1e-2

    # TODO: Test logpdf!()

    # Test non-logged PDF
    ent, ent_hat = entropy(d), -mean(log(pdf(d, X)))
    @assert norm(ent - ent_hat, Inf) < 1e-2

    # TODO: Test pdf!()

    # TODO: Test independence of draws?

    # TODO: Test cdf, quantile

    # TODO: Test mgf, cf

    # TODO: Test median
    # TODO: Test modes

    # TODO: Test kurtosis
    # TODO: Test skewness
end
