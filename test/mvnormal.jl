# Tests on Multivariate Normal distributions

import PDMats: ScalMat, PDiagMat, PDMat

using Distributions
using LinearAlgebra, Random, Test

import Distributions: distrname



####### Core testing procedure

function test_mvnormal(g::AbstractMvNormal, n_tsamples::Int=10^6)
    d = length(g)
    μ = mean(g)
    Σ = cov(g)
    @test partype(g) == Float64
    @test isa(μ, Vector{Float64})
    @test isa(Σ, Matrix{Float64})
    @test length(μ) == d
    @test size(Σ) == (d, d)
    @test var(g)     ≈ diag(Σ)
    @test entropy(g) ≈ 0.5 * logdet(2π * ℯ * Σ)
    ldcov = logdetcov(g)
    @test ldcov ≈ logdet(Σ)
    vs = diag(Σ)
    @test g == typeof(g)(params(g)...)

    # test sampling for AbstractMatrix (here, a SubArray):
    subX = view(rand(d, 2d), :, 1:d)
    @test isa(rand!(g, subX), SubArray)

    # sampling
    @test isa(rand(g), Vector{Float64})
    X = rand(g, n_tsamples)
    emp_mu = vec(mean(X, dims=2))
    Z = X .- emp_mu
    emp_cov = (Z * Z') * inv(n_tsamples)
    for i = 1:d
        @test isapprox(emp_mu[i]   , μ[i]  , atol=sqrt(vs[i] / n_tsamples) * 8.0)
    end
    for i = 1:d, j = 1:d
        @test isapprox(emp_cov[i,j], Σ[i,j], atol=sqrt(vs[i] * vs[j]) * 10.0 / sqrt(n_tsamples))
    end

    X = rand(MersenneTwister(14), g, n_tsamples)
    Y = rand(MersenneTwister(14), g, n_tsamples)
    @test X == Y
    emp_mu = vec(mean(X, dims=2))
    Z = X .- emp_mu
    emp_cov = (Z * Z') * inv(n_tsamples)
    for i = 1:d
        @test isapprox(emp_mu[i]   , μ[i]  , atol=sqrt(vs[i] / n_tsamples) * 8.0)
    end
    for i = 1:d, j = 1:d
        @test isapprox(emp_cov[i,j], Σ[i,j], atol=sqrt(vs[i] * vs[j]) * 10.0 / sqrt(n_tsamples))
    end


    # evaluation of sqmahal & logpdf
    U = X .- μ
    sqm = vec(sum(U .* (Σ \ U), dims=1))
    for i = 1:min(100, n_tsamples)
        @test sqmahal(g, X[:,i]) ≈ sqm[i]
    end
    @test sqmahal(g, X) ≈ sqm

    lp = -0.5 .* sqm .- 0.5 * (d * log(2.0 * pi) + ldcov)
    for i = 1:min(100, n_tsamples)
        @test logpdf(g, X[:,i]) ≈ lp[i]
    end
    @test logpdf(g, X) ≈ lp

    # log likelihood
    @test loglikelihood(g, X) ≈ sum([Distributions._logpdf(g, X[:,i]) for i in 1:size(X, 2)])
end

###### General Testing

mu = [1., 2., 3.]
va = [1.2, 3.4, 2.6]
C = [4. -2. -1.; -2. 5. -1.; -1. -1. 6.]

h = [1., 2., 3.]
dv = [1.2, 3.4, 2.6]
J = [4. -2. -1.; -2. 5. -1.; -1. -1. 6.]

for (T, g, μ, Σ) in [
    (IsoNormal, MvNormal(mu, sqrt(2.0)), mu, Matrix(2.0I, 3, 3)),
    (ZeroMeanIsoNormal, MvNormal(3, sqrt(2.0)), zeros(3), Matrix(2.0I, 3, 3)),
    (DiagNormal, MvNormal(mu, sqrt.(va)), mu, Matrix(Diagonal(va))),
    (ZeroMeanDiagNormal, MvNormal(sqrt.(va)), zeros(3), Matrix(Diagonal(va))),
    (FullNormal, MvNormal(mu, C), mu, C),
    (ZeroMeanFullNormal, MvNormal(C), zeros(3), C),
    (IsoNormalCanon, MvNormalCanon(h, 2.0), h / 2.0, Matrix(0.5I, 3, 3)),
    (ZeroMeanIsoNormalCanon, MvNormalCanon(3, 2.0), zeros(3), Matrix(0.5I, 3, 3)),
    (DiagNormalCanon, MvNormalCanon(h, dv), h ./ dv, Matrix(Diagonal(inv.(dv)))),
    (ZeroMeanDiagNormalCanon, MvNormalCanon(dv), zeros(3), Matrix(Diagonal(inv.(dv)))),
    (FullNormalCanon, MvNormalCanon(h, J), J \ h, inv(J)),
    (ZeroMeanFullNormalCanon, MvNormalCanon(J), zeros(3), inv(J)),
    (FullNormal, MvNormal(mu, Symmetric(C)), mu, Matrix(Symmetric(C))),
    (DiagNormal, MvNormal(mu, Diagonal(dv)), mu, Matrix(Diagonal(dv))) ]

    println("    testing $(distrname(g))")

    @test isa(g, T)
    @test mean(g)   ≈ μ
    @test cov(g)    ≈ Σ
    @test invcov(g) ≈ inv(Σ)
    test_mvnormal(g, 10^4)

    # conversion between mean form and canonical form
    if isa(g, MvNormal)
        gc = canonform(g)
        @test isa(gc, MvNormalCanon)
        @test length(gc) == length(g)
        @test mean(gc) ≈ mean(g)
        @test cov(gc)  ≈ cov(g)
    else
        @assert isa(g, MvNormalCanon)
        gc = meanform(g)
        @test isa(gc, MvNormal)
        @test length(gc) == length(g)
        @test mean(gc) ≈ mean(g)
        @test cov(gc)  ≈ cov(g)
    end
end

##### Constructors and conversions
mu = [1., 2., 3.]
C = [4. -2. -1.; -2. 5. -1.; -1. -1. 6.]
J = inv(C)
h = J \ mu
@test typeof(MvNormal(mu, PDMat(Array{Float32}(C)))) == typeof(MvNormal(mu, PDMat(C)))
@test typeof(MvNormal(mu, Array{Float32}(C))) == typeof(MvNormal(mu, PDMat(C)))
@test typeof(MvNormal(mu, 2.0f0)) == typeof(MvNormal(mu, 2.0))

@test typeof(MvNormalCanon(h, PDMat(Array{Float32}(J)))) == typeof(MvNormalCanon(h, PDMat(J)))
@test typeof(MvNormalCanon(h, Array{Float32}(J))) == typeof(MvNormalCanon(h, PDMat(J)))
@test typeof(MvNormalCanon(h, 2.0f0)) == typeof(MvNormalCanon(h, 2.0))

@test typeof(MvNormalCanon(mu, Array{Float16}(h), PDMat(Array{Float32}(J)))) == typeof(MvNormalCanon(mu, h, PDMat(J)))

d = MvNormal(Array{Float32}(mu), PDMat(Array{Float32}(C)))
@test typeof(convert(MvNormal{Float64}, d)) == typeof(MvNormal(mu, PDMat(C)))
@test typeof(convert(MvNormal{Float64}, d.μ, d.Σ)) == typeof(MvNormal(mu, PDMat(C)))

d = MvNormalCanon(Array{Float32}(mu), Array{Float32}(h), PDMat(Array{Float32}(J)))
@test typeof(convert(MvNormalCanon{Float64}, d)) == typeof(MvNormalCanon(mu, h, PDMat(J)))
@test typeof(convert(MvNormalCanon{Float64}, d.μ, d.h, d.J)) == typeof(MvNormalCanon(mu, h, PDMat(J)))

##### MLE

# a slow but safe way to implement MLE for verification

function _gauss_mle(x::Matrix{Float64})
    mu = vec(mean(x, dims=2))
    z = x .- mu
    C = (z * z') * (1/size(x,2))
    return mu, C
end

function _gauss_mle(x::Matrix{Float64}, w::Vector{Float64})
    sw = sum(w)
    mu = (x * w) * (1/sw)
    z = x .- mu
    C = (z * (Diagonal(w) * z')) * (1/sw)
    LinearAlgebra.copytri!(C, 'U')
    return mu, C
end

println("    testing fit_mle")

x = randn(3, 200) .+ randn(3) * 2.
w = rand(200)
u, C = _gauss_mle(x)
uw, Cw = _gauss_mle(x, w)

g = fit_mle(MvNormal, suffstats(MvNormal, x))
@test isa(g, FullNormal)
@test mean(g) ≈ u
@test cov(g)  ≈ C

g = fit_mle(MvNormal, x)
@test isa(g, FullNormal)
@test mean(g) ≈ u
@test cov(g)  ≈ C

g = fit_mle(MvNormal, x, w)
@test isa(g, FullNormal)
@test mean(g) ≈ uw
@test cov(g)  ≈ Cw

g = fit_mle(IsoNormal, x)
@test isa(g, IsoNormal)
@test g.μ       ≈ u
@test g.Σ.value ≈ mean(diag(C))

g = fit_mle(IsoNormal, x, w)
@test isa(g, IsoNormal)
@test g.μ       ≈ uw
@test g.Σ.value ≈ mean(diag(Cw))

g = fit_mle(DiagNormal, x)
@test isa(g, DiagNormal)
@test g.μ      ≈ u
@test g.Σ.diag ≈ diag(C)

g = fit_mle(DiagNormal, x, w)
@test isa(g, DiagNormal)
@test g.μ      ≈ uw
@test g.Σ.diag ≈ diag(Cw)
