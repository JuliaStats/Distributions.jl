# Canonical form of multivariate normal

import NumericExtensions
import NumericExtensions.ScalMat
import NumericExtensions.PDiagMat
import NumericExtensions.PDMat

using Distributions
using Base.Test

##### construction, basic properties, and evaluation

h = [1., 2., 3.]
dv = [1.2, 3.4, 2.6]
J = [4. -2. -1.; -2. 5. -1.; -1. -1. 6.]

x1 = [3.2, 1.8, 2.4]
x = rand(3, 100)

# SGauss

gs = IsoNormalCanon(h, 2.0)
@test isa(gs, IsoNormalCanon)
@test dim(gs) == 3
@test mean(gs) == mode(gs) == h / 2.0
@test !gs.zeromean
@test invcov(gs) == diagm(fill(2.0, 3))
@test cov(gs) == diagm(fill(0.5, 3))
@test var(gs) == diag(cov(gs))
@test_approx_eq entropy(gs) 0.5 * logdet(2π * e * cov(gs))

gsz = IsoNormalCanon(3, 2.0)
@test isa(gsz, IsoNormalCanon)
@test dim(gsz) == 3
@test mean(gsz) == zeros(3)
@test gsz.zeromean

# DGauss

gd = DiagNormalCanon(h, dv)
@test isa(gd, DiagNormalCanon)
@test dim(gd) == 3
@test_approx_eq mean(gd) h ./ dv
@test !gd.zeromean
@test invcov(gd) == diagm(dv)
@test_approx_eq cov(gd) diagm(1.0 / dv)
@test_approx_eq var(gd) diag(cov(gd))
@test_approx_eq entropy(gd) 0.5 * logdet(2π * e * cov(gd))

gdz = DiagNormalCanon(dv)
@test isa(gdz, DiagNormalCanon)
@test dim(gdz) == 3
@test mean(gdz) == zeros(3)
@test gdz.zeromean

# Gauss

gf = MvNormalCanon(h, J)
@test isa(gf, MvNormalCanon)
@test dim(gf) == 3
@test_approx_eq mean(gf) J \ h
@test !gf.zeromean
@test invcov(gf) == J
@test_approx_eq cov(gf) inv(J)
@test_approx_eq var(gf) diag(cov(gf))
@test_approx_eq entropy(gf) 0.5 * logdet(2π * e * cov(gf))

gfz = MvNormalCanon(J)
@test isa(gfz, MvNormalCanon)
@test dim(gfz) == 3
@test mean(gfz) == zeros(3)
@test gfz.zeromean


# conversion

us = convert(IsoNormal, gs)
gs2 = convert(IsoNormalCanon, us)
@test isa(us, IsoNormal)
@test isa(gs2, IsoNormalCanon)
@test dim(gs2) == dim(gs)
@test_approx_eq gs2.h gs.h
@test_approx_eq gs2.J.value gs.J.value

ud = convert(DiagNormal, gd)
gd2 = convert(DiagNormalCanon, ud)
@test isa(ud, DiagNormal)
@test isa(gd2, DiagNormalCanon)
@test dim(gd2) == dim(gd)
@test_approx_eq gd2.h gd.h
@test_approx_eq gd2.J.diag gd.J.diag

uf = convert(MvNormal, gf)
gf2 = convert(MvNormalCanon, uf)
@test isa(uf, MvNormal)
@test isa(gf2, MvNormalCanon)
@test dim(gf2) == dim(gf)
@test_approx_eq gf2.h gf.h
@test_approx_eq gf2.J.mat gf.J.mat


# logpdf evaluation

@test_approx_eq logpdf(gs, x1) logpdf(us, x1)
@test_approx_eq logpdf(gs, x)  logpdf(us, x)
@test_approx_eq logpdf(gsz, x1) logpdf(convert(IsoNormal, gsz), x1)
@test_approx_eq logpdf(gsz, x)  logpdf(convert(IsoNormal, gsz), x)

@test_approx_eq logpdf(gd, x1) logpdf(ud, x1)
@test_approx_eq logpdf(gd, x)  logpdf(ud, x)
@test_approx_eq logpdf(gdz, x1) logpdf(convert(DiagNormal, gdz), x1)
@test_approx_eq logpdf(gdz, x)  logpdf(convert(DiagNormal, gdz), x)

@test_approx_eq logpdf(gf, x1) logpdf(uf, x1)
@test_approx_eq logpdf(gf, x)  logpdf(uf, x)
@test_approx_eq logpdf(gfz, x1) logpdf(convert(MvNormal, gfz), x1)
@test_approx_eq logpdf(gfz, x)  logpdf(convert(MvNormal, gfz), x)


# sampling

x = rand(gs)
@test isa(x, Vector{Float64})
@test length(x) == dim(gs)

x = rand(gd)
@test isa(x, Vector{Float64})
@test length(x) == dim(gd)

x = rand(gf)
@test isa(x, Vector{Float64})
@test length(x) == dim(gf)

x = rand(gsz)
@test isa(x, Vector{Float64})
@test length(x) == dim(gsz)

x = rand(gdz)
@test isa(x, Vector{Float64})
@test length(x) == dim(gdz)

x = rand(gfz)
@test isa(x, Vector{Float64})
@test length(x) == dim(gfz)

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

x = rand(gsz, n)
@test isa(x, Matrix{Float64})
@test size(x) == (dim(gsz), n)

x = rand(gdz, n)
@test isa(x, Matrix{Float64})
@test size(x) == (dim(gdz), n)

x = rand(gfz, n)
@test isa(x, Matrix{Float64})
@test size(x) == (dim(gfz), n)

