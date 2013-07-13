using Distributions
using Base.Test

# Need to include probabilistic tests.
# Allowing slow tests is defensible to ensure correctness.

n_samples = 1_000_000

for d in [Dirichlet([100.0, 17.0, 31.0, 45.0]),
          Multinomial(1, [1/3, 1/3, 1/3]),
          MultivariateNormal([0.0, 0.0], [1.0 0.9; 0.9 1.0])]
    # Check that we can generate a single random draw
    draw = rand(d)
    @test length(draw) == dim(d)

    # Check that draw satifies insupport()
    @test insupport(d, draw)

    # Check that we can generate many random draws at once
    X = rand(d, n_samples)
    @test size(X, 1) == dim(d)

    # Check that sequence of draws satifies insupport()
    @test insupport(d, X)

    # Check that we can generate many random draws in-place
    rand!(d, X)

    # Check that KL between fitted distribution and true distribution
    #  is small
    d_hat = fit(typeof(d), X)
    # TODO: @test kl(d, d_hat) < 1e-2

    # Because of the Weak Law of Large Numbers,
    #  empirical mean should be close to theoretical value
    mu, mu_hat = mean(d), vec(mean(X, 2))
    @test norm(mu - mu_hat, Inf) < 1e-2

    # Because of the Weak Law of Large Numbers,
    #  empirical covariance matrix should be close to theoretical value
    sigma, sigma_hat = cov(d), cov(X')
    @test norm(sigma - sigma_hat, Inf) < 1e-2

    # TODO: Test cov and cor for multivariate distributions
    # TODO: Decide how var(d::MultivariateDistribution) should be defined

    # By the Asymptotic Equipartition Property,
    #  empirical mean negative log PDF should be close to theoretical value
    ent, ent_hat = entropy(d), -mean(logpdf(d, X))
    @test norm(ent - ent_hat, Inf) < 1e-2

    # TODO: Test logpdf!()

    # Test non-logged PDF
    ent, ent_hat = entropy(d), -mean(log(pdf(d, X)))
    @test norm(ent - ent_hat, Inf) < 1e-2

    # TODO: Test pdf!()

    # TODO: Test independence of draws?

    # TODO: Test cdf, quantile

    # TODO: Test mgf, cf

    # TODO: Test median
    # TODO: Test modes

    # TODO: Test kurtosis
    # TODO: Test skewness
end


#####
#
# Specialized testings
#
#####

# Multinomial

d = Multinomial(1, [0.5, 0.4, 0.1])
d = Multinomial(1, 3)
d = Multinomial(2)
mean(d)
var(d)
@test insupport(d, [1, 0])
@test !insupport(d, [1, 1])
@test insupport(d, [0, 1])
pmf(d, [1, 0])
pmf(d, [1, 1])
pmf(d, [0, 1])
logpmf(d, [1, 0])
logpmf(d, [1, 1])
logpmf(d, [0, 1])
d = Multinomial(10)
rand(d)
A = Array(Int, 10, 2)
rand!(d, A)

# Dirichlet

d = Dirichlet([1.0, 2.0, 1.0])
d = Dirichlet(3)
mean(d)
var(d)
insupport(d, [0.1, 0.8, 0.1])
insupport(d, [0.1, 0.8, 0.2])
insupport(d, [0.1, 0.8])
pdf(d, [0.1, 0.8, 0.1])
rand(d)
A = Array(Float64, 3, 10)
rand!(d, A)

d = Dirichlet([1.5, 2.0, 2.5])
x = [0.2 0.5 0.3; 0.1 0.5 0.4; 0.8 0.1 0.1; 0.05 0.15 0.8]'

r0 = zeros(4)
for i = 1 : 4
    r0[i] = logpdf(d, x[:,i])
end
@test_approx_eq logpdf(d, x) r0

# MultivariateNormal

d = MultivariateNormal(zeros(2), eye(2))
@test abs(pdf(d, [0., 0.]) - 0.159155) < 1.0e-5
@test abs(pdf(d, [1., 0.]) - 0.0965324) < 1.0e-5
@test abs(pdf(d, [1., 1.]) - 0.0585498) < 1.0e-5

d = MultivariateNormal(zeros(3), [4. -2. -1.; -2. 5. -1.; -1. -1. 6.])
@test_approx_eq logpdf(d, [3., 4., 5.]) (-15.75539253001834)

x = [3. 4. 5.; 1. 2. 3.; -4. -3. -2.; -1. -3. -2.]'
r0 = zeros(4)
for i = 1 : 4
    r0[i] = logpdf(d, x[:,i])
end
@test_approx_eq logpdf(d, x) r0

