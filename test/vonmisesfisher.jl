# Tests for Von-Mises Fisher distribution

using Distributions
using Base.Test

D = 3
mu = randn(D)
mu = mu / norm(mu)
kappa = 100.0
d = VonMisesFisher(mu, kappa)

# Basics

@test length(d) == D
@test d.kappa == kappa
@test_approx_eq d.mu mean(d)
@test_approx_eq norm(d.mu) 1.0

# MLE

x = rand(d, 10_000)
dmle = fit_mle(VonMisesFisher, x')
@test all(abs(mean(d) - mean(dmle)) .< .01)
@test_approx_eq norm(dmle.mu) 1.0
#@test abs(scale(dmle) - scale(d)) < .01 * scale(d) # within 1%? not always...

# Density

# TODO: Check against R's movMF. (Currently I'm a bit suspicious about their code.)
# > set.seed(1)
# > mu=c(1,0,0)
# > kappa=1.
# > x = rmovMF(1, mu, kappa)
# > x
#           [,1]       [,2]     [,3]
# [1,] 0.1772372 -0.4566632 0.871806
# > dmovMF(x, mu, kappa)
# [1] 1.015923
# WEIRD:
# > dmovMF(x, mu, 100)
# [1] 1.015923
# > dmovMF(x, mu, 1)
# [1] 1.015923

# mu = [1.0, 0., 0.]
# kappa = 1.0
# x = [0.1772372, -0.4566632, 0.871806]
# d = VonMisesFisher(mu, kappa)
#@test abs(logpdf(d, x) - 1.015923) < .00001


