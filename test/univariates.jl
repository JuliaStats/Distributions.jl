# Testing univariate distributions

using Distributions

n_tsamples = 10^6

# This list includes both discrete and continuous distributions
#
distrlist = [
     # Cosine(),
     Exponential(1.0),
     Exponential(5.1),
     FDist(9, 9),
     FDist(9, 21),
     FDist(21, 9),
     Gumbel(3.0, 5.0),
     Gumbel(5, 3),
     InverseGaussian(1.0,1.0),
     InverseGaussian(2.0,7.0),
     InverseGamma(1.0, 1.0),
     InverseGamma(2.0, 3.0),
     # Kolmogorov(), # no quantile function
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
     Normal(),
     Normal(-1.0, 10.0),
     Normal(1.0, 10.0),
     NormalCanon(),
     NormalCanon(-1.0, 0.5),
     NormalCanon(2.0, 0.8),
     Pareto(),
     Pareto(5.0,2.0),
     Pareto(2.0,5.0),
     Rayleigh(1.0),
     Rayleigh(5.0),
     Rayleigh(10.0),
     # Skellam(10.0, 2.0), # no quantile function
     TDist(1),
     TDist(28),
     Uniform(0.0, 1.0),
     Uniform(3.0, 17.0),
     Uniform(3.0, 3.1),
     Weibull(0.23,0.1),
     Weibull(2.3,0.1),
     Weibull(23.0,0.1),
     Weibull(230.0,0.1),
     Weibull(0.23),
     Weibull(2.3),
     Weibull(23.0),
     Weibull(230.0),
     Weibull(0.23,10.0),
     Weibull(2.3,10.0),
     Weibull(23.0,10.0),
     Weibull(230.0,10.0)]


for distr in distrlist
     println("    testing $(distr)")
     test_distr(distr, n_tsamples)
end


