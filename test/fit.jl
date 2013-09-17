using Distributions
using Base.Test

# Fit MLE

N = 10^5

d = fit(DiscreteUniform, rand(DiscreteUniform(10, 15), N))
@test isa(d, DiscreteUniform)
@test min(d) == 10
@test max(d) == 15

d = fit(Bernoulli, rand(Bernoulli(0.7), N))
@test isa(d, Bernoulli)
@test_approx_eq_eps mean(d) 0.7 0.01

d = fit(Beta, rand(Beta(1.3, 3.7), N))
@test isa(d, Beta)
@test_approx_eq_eps d.alpha 1.3 0.1
@test_approx_eq_eps d.beta  3.7 0.1

d = fit(Binomial, 100, rand(Binomial(100, 0.3), N))
@test isa(d, Binomial)
@test d.size == 100
@test_approx_eq_eps d.prob 0.3 0.01

p = [0.2, 0.5, 0.3]
x = rand(Categorical(p), N)
d = fit(Categorical, x)
@test isa(d, Categorical)
@test d.K == 3
@test_approx_eq_eps d.prob p 0.01

d = fit_mle(Categorical, (3, x))
@test isa(d, Categorical)
@test d.K == 3
@test_approx_eq_eps d.prob p 0.01

d = fit(Exponential, rand(Exponential(0.5), N))
@test isa(d, Exponential)
@test_approx_eq_eps mean(d) 0.5 0.01

x = rand(Normal(11.3, 5.7), N)
d = fit(Normal, x)
@test isa(d, Normal)
@test_approx_eq_eps mean(d) 11.3 0.1
@test_approx_eq_eps std(d) 5.7 0.1

d = fit_mle(Normal, suffstats(Normal, x))
@test isa(d, Normal)
@test_approx_eq_eps mean(d) 11.3 0.1
@test_approx_eq_eps std(d) 5.7 0.1

d = fit(Uniform, rand(Uniform(1.2, 10.3), N))
@test isa(d, Uniform)
@test 1.2 <= min(d) <= max(d) <= 10.3
@test_approx_eq_eps min(d) 1.2 0.02
@test_approx_eq_eps max(d) 10.3 0.02

d = fit(Gamma, rand(Gamma(3.9, 2.1), N))
@test isa(d, Gamma)
@test_approx_eq_eps d.shape 3.9 0.1
@test_approx_eq_eps d.scale 2.1 0.2

d = fit(Geometric, rand(Geometric(0.3), N))
@test isa(d, Geometric)
@test_approx_eq_eps d.prob 0.3 0.01

d = fit(Laplace, rand(Laplace(5.0, 3.0), N))
@test isa(d, Laplace)
@test_approx_eq_eps d.location 5.0 0.1
@test_approx_eq_eps d.scale 3.0 0.2

d = fit(Poisson, rand(Poisson(8.2), N))
@test isa(d, Poisson)
@test_approx_eq_eps mean(d) 8.2 0.2

