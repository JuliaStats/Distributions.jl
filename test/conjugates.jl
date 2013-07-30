using Base.Test
using Distributions

x = rand(Bernoulli(0.3), 10_000)
newprior = posterior(Beta(1, 1), Bernoulli, x)
@test_approx_eq_eps mean(newprior) 0.3 0.1

x = rand(Binomial(10, 0.3), 10_000)
newprior = posterior(Beta(1, 1), Binomial, 10, x)
@test_approx_eq_eps mean(newprior) 0.3 0.1

x = rand(Categorical([0.5, 0.25, 0.25]), 10_000)
newprior = posterior(Dirichlet([1., 1., 1.]), Categorical, x)
@test norm(mean(newprior) - [0.5, 0.25, 0.25]) < 0.1

x = rand(Multinomial(1, [0.5, 0.25, 0.25]))
newprior = posterior(Dirichlet([1., 1., 1.]), Multinomial, x)
@test sum(newprior.alpha) == 4.0

X = rand(Multinomial(1, [0.5, 0.25, 0.25]), 10_000)
newprior = posterior(Dirichlet([1., 1., 1.]), Multinomial, X)
@test norm(mean(newprior) - [0.5, 0.25, 0.25]) < 0.1

x = rand(Exponential(1 / 1.7), 10_000)
newprior = posterior(Gamma(1.0, 1.0), Exponential, x)
@test_approx_eq_eps mean(newprior) 1.7 0.1

x = rand(Normal(1.7, 3.0), 10_000)
newprior = posterior(Normal(0.0, 1.0), 3.0, Normal, x)
@test_approx_eq_eps mean(newprior) 1.7 0.1

x = rand(Normal(1.7, 3.0), 10_000)
newprior = posterior(1.7, InvertedGamma(1.0, 1.0), Normal, x)
@test_approx_eq_eps sqrt(mean(newprior)) 3.0 0.1
