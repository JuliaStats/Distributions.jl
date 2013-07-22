using Base.Test

x = rand(Bernoulli(0.3), 10_000)
newprior = posterior(Bernoulli, Beta(1, 1), x)
@test_approx_eq_eps mean(newprior) 0.3 0.1

x = rand(Binomial(10, 0.3), 10_000)
X = hcat(x, [10 for i in 1:10_000])
newprior = posterior(Binomial, Beta(1, 1), X)
@test_approx_eq_eps mean(newprior) 0.3 0.1

x = rand(Categorical([0.5, 0.25, 0.25]), 10_000)
newprior = posterior(Categorical, Dirichlet([1., 1., 1.]), x)
@test norm(mean(newprior) - [0.5, 0.25, 0.25]) < 0.1

x = rand(Multinomial(1, [0.5, 0.25, 0.25]))
newprior = posterior(Multinomial, Dirichlet([1., 1., 1.]), x)
@test sum(newprior.alpha) == 4.0

X = rand(Multinomial(1, [0.5, 0.25, 0.25]), 10_000)
newprior = posterior(Multinomial, Dirichlet([1., 1., 1.]), X)
@test norm(mean(newprior) - [0.5, 0.25, 0.25]) < 0.1
