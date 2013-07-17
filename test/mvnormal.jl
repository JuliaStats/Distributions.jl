# Tests on Multivariate Normal distributions

using NumericExtensions
using Distributions
using Base.Test

##### construction, basic properties, and evaluation

mu = [1., 2., 3.]
va = [1.2, 3.4, 2.6]
C = [4. -2. -1.; -2. 5. -1.; -1. -1. 6.]

x1 = [3.2, 1.8, 2.4]
x = rand(3, 100)

# SGauss

gs = MvNormal(mu, 2.0)
@test isa(gs, MvNormal{ScalMat})
@test dim(gs) == 3
@test mean(gs) == mode(gs) == mu
@test cov(gs) == diagm(fill(2.0, 3))
@test_approx_eq entropy(gs) 0.5 * logdet(2π * e * cov(gs))

# DGauss

gd = MvNormal(mu, va)
@test isa(gd, MvNormal{PDiagMat})
@test dim(gd) == 3
@test mean(gd) == mode(gd) == mu
@test cov(gd) == diagm(va)
@test_approx_eq entropy(gd) 0.5 * logdet(2π * e * cov(gd))

# Gauss

gf = MvNormal(mu, C)
@test isa(gf, MvNormal{PDMat})
@test dim(gf) == 3
@test mean(gf) == mode(gf) == mu
@test cov(gf) == C
@test_approx_eq entropy(gf) 0.5 * logdet(2π * e * cov(gf))


##### LogPDF/PDF evaluation

@test_approx_eq_eps logpdf(gs, x1) -5.106536370454 1.0e-8
@test_approx_eq_eps logpdf(gd, x1) -6.029399605174 1.0e-8
@test_approx_eq_eps logpdf(gf, x1) -5.680452770982 1.0e-8

n = size(x, 2)
r = zeros(n)

for i = 1:n; r[i] = logpdf(gs, x[:,i]); end
@test_approx_eq logpdf(gs, x) r
@test_approx_eq pdf(gs, x) exp(r)

for i = 1:n; r[i] = logpdf(gd, x[:,i]); end
@test_approx_eq logpdf(gd, x) r
@test_approx_eq pdf(gd, x) exp(r)

for i = 1:n; r[i] = logpdf(gf, x[:,i]); end
@test_approx_eq logpdf(gf, x) r
@test_approx_eq pdf(gf, x) exp(r)


##### Sampling 

x = rand(gs)
@test isa(x, Vector{Float64})
@test length(x) == dim(gs)

x = rand(gd)
@test isa(x, Vector{Float64})
@test length(x) == dim(gd)

x = rand(gf)
@test isa(x, Vector{Float64})
@test length(x) == dim(gf)

n = 10
x = rand(gs, n)
@test isa(x, Matrix{Float64})
@test size(x) == (dim(gs), n)

x = rand(gd, n)
@test isa(x, Matrix{Float64})
@test size(x) == (dim(gd), n)

x = rand(gf, n)
@test isa(x, Matrix{Float64})
@test size(x) == (dim(gf), n)

