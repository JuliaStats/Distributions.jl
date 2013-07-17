# Tests on Multivariate Normal distributions

using NumericExtensions
using Distributions
using Base.Test

##### construction, basic properties, and evaluation

mu = rand(3)
va = [1.2, 3.4, 2.6]
C = [4. -2. -1.; -2. 5. -1.; -1. -1. 6.]

x1 = rand(3)
x = rand(3, 100)

# SGauss

g = MvNormal(mu, 2.0)
@test isa(g, MvNormal{ScalMat})
@test dim(g) == 3
@test mean(g) == mode(g) == mu
@test cov(g) == diagm(fill(2.0, 3))

# DGauss

g = MvNormal(mu, va)
@test isa(g, MvNormal{PDiagMat})
@test dim(g) == 3
@test mean(g) == mode(g) == mu
@test cov(g) == diagm(va)

# Gauss

g = MvNormal(mu, C)
@test isa(g, MvNormal{PDMat})
@test dim(g) == 3
@test mean(g) == mode(g) == mu
@test cov(g) == C


