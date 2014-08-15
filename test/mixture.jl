using Distributions
using Base.Test

m = MixtureModel([Poisson(2.0), Binomial(10, 0.3)],
                 [0.4, 0.6])
x = rand(m, (100,))
@test_approx_eq exp(logpdf(m,x)) pdf(m,x)
@test Distributions.variate_form(typeof(m))=== Univariate
@test Distributions.value_support(typeof(m))=== Discrete

