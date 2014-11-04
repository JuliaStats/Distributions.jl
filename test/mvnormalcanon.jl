# Canonical form of multivariate normal

import PDMats: ScalMat, PDiagMat, PDMat

using Distributions
using Base.Test

##### construction, basic properties, and evaluation

h = [1., 2., 3.]
dv = [1.2, 3.4, 2.6]
J = [4. -2. -1.; -2. 5. -1.; -1. -1. 6.]

x1 = [3.2, 1.8, 2.4]
x = rand(3, 100)


gs = MvNormalCanon(h, 2.0)
@test isa(gs, IsoNormalCanon)
@test length(gs) == 3
@test mean(gs) == mode(gs) == h / 2.0
@test invcov(gs) == diagm(fill(2.0, 3))
@test cov(gs) == diagm(fill(0.5, 3))
@test var(gs) == diag(cov(gs))
@test_approx_eq entropy(gs) 0.5 * logdet(2π * e * cov(gs))

gsz = MvNormalCanon(3, 2.0)
@test isa(gsz, ZeroMeanIsoNormalCanon)
@test length(gsz) == 3
@test mean(gsz) == zeros(3)


gd = MvNormalCanon(h, dv)
@test isa(gd, DiagNormalCanon)
@test length(gd) == 3
@test_approx_eq mean(gd) h ./ dv
@test invcov(gd) == diagm(dv)
@test_approx_eq cov(gd) diagm(1.0 ./ dv)
@test_approx_eq var(gd) diag(cov(gd))
@test_approx_eq entropy(gd) 0.5 * logdet(2π * e * cov(gd))

gdz = MvNormalCanon(dv)
@test isa(gdz, ZeroMeanDiagNormalCanon)
@test length(gdz) == 3
@test mean(gdz) == zeros(3)

gf = MvNormalCanon(h, J)
@test isa(gf, FullNormalCanon)
@test length(gf) == 3
@test_approx_eq mean(gf) J \ h
@test invcov(gf) == J
@test_approx_eq cov(gf) inv(J)
@test_approx_eq var(gf) diag(cov(gf))
@test_approx_eq entropy(gf) 0.5 * logdet(2π * e * cov(gf))

gfz = MvNormalCanon(J)
@test isa(gfz, ZeroMeanFullNormalCanon)
@test length(gfz) == 3
@test mean(gfz) == zeros(3)



# conversion

us = meanform(gs)
gs2 = canonform(us)
@test isa(us, IsoNormal)
@test isa(gs2, IsoNormalCanon)
@test length(gs2) == length(gs)
@test_approx_eq gs2.h gs.h
@test_approx_eq gs2.J.value gs.J.value

ud = meanform(gd)
gd2 = canonform(ud)
@test isa(ud, DiagNormal)
@test isa(gd2, DiagNormalCanon)
@test length(gd2) == length(gd)
@test_approx_eq gd2.h gd.h
@test_approx_eq gd2.J.diag gd.J.diag

uf = meanform(gf)
gf2 = canonform(uf)
@test isa(uf, MvNormal)
@test isa(gf2, MvNormalCanon)
@test length(gf2) == length(gf)
@test_approx_eq gf2.h gf.h
@test_approx_eq gf2.J.mat gf.J.mat


# logpdf evaluation

@test_approx_eq logpdf(gs, x1) logpdf(us, x1)
@test_approx_eq logpdf(gs, x)  logpdf(us, x)
@test_approx_eq logpdf(gsz, x1) logpdf(meanform(gsz), x1)
@test_approx_eq logpdf(gsz, x)  logpdf(meanform(gsz), x)

@test_approx_eq logpdf(gd, x1) logpdf(ud, x1)
@test_approx_eq logpdf(gd, x)  logpdf(ud, x)
@test_approx_eq logpdf(gdz, x1) logpdf(meanform(gdz), x1)
@test_approx_eq logpdf(gdz, x)  logpdf(meanform(gdz), x)

@test_approx_eq logpdf(gf, x1) logpdf(uf, x1)
@test_approx_eq logpdf(gf, x)  logpdf(uf, x)
@test_approx_eq logpdf(gfz, x1) logpdf(meanform(gfz), x1)
@test_approx_eq logpdf(gfz, x)  logpdf(meanform(gfz), x)


# sampling

x = rand(gs)
@test isa(x, Vector{Float64})
@test length(x) == length(gs)

x = rand(gd)
@test isa(x, Vector{Float64})
@test length(x) == length(gd)

x = rand(gf)
@test isa(x, Vector{Float64})
@test length(x) == length(gf)

x = rand(gsz)
@test isa(x, Vector{Float64})
@test length(x) == length(gsz)

x = rand(gdz)
@test isa(x, Vector{Float64})
@test length(x) == length(gdz)

x = rand(gfz)
@test isa(x, Vector{Float64})
@test length(x) == length(gfz)

n = 10
x = rand(gs, n)
@test isa(x, Matrix{Float64})
@test size(x) == (length(gs), n)

x = rand(gd, n)
@test isa(x, Matrix{Float64})
@test size(x) == (length(gd), n)

x = rand(gf, n)
@test isa(x, Matrix{Float64})
@test size(x) == (length(gf), n)

x = rand(gsz, n)
@test isa(x, Matrix{Float64})
@test size(x) == (length(gsz), n)

x = rand(gdz, n)
@test isa(x, Matrix{Float64})
@test size(x) == (length(gdz), n)

x = rand(gfz, n)
@test isa(x, Matrix{Float64})
@test size(x) == (length(gfz), n)

