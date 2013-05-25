N = 100_000

fit(Bernoulli, rand(Bernoulli(0.7), N))

fit(Beta, rand(Beta(1.3, 3.7), N))

fit(Binomial, rand(Binomial(N, 0.3)), N)

fit(DiscreteUniform, rand(DiscreteUniform(300_000, 700_000), N))

fit(Exponential, rand(Exponential(0.1), N))

fit(Gamma, rand(Gamma(7.9, 3.1), N))

fit(Geometric, rand(Geometric(0.1), N))

fit(Laplace, rand(Laplace(10.0, 3.0), N))

fit(Normal, rand(Normal(11.3, 5.7), N))

fit(Poisson, rand(Poisson(19.0), N))

fit(Uniform, rand(Uniform(1.1, 98.3), N))
