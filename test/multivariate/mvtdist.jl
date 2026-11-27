using Distributions, Random, StaticArrays, LinearAlgebra
using Test

import Distributions: GenericMvTDist, mvtdist, DiagTDist
import PDMats: PDMat

@testset "mvtdist" begin
# Set location vector mu and scale matrix Sigma as in
# Hofert M. On Sampling from the Multivariate t Distribution. The R Journal
mu = [1., 2]
Sigma = [4. 2; 2 3]

# LogPDF evaluation for varying degrees of freedom df
# Julia's output is compared to R's corresponding values obtained via R's mvtnorm package
# R code exemplifying how the R values (rvalues) were obtained:
# options(digits=20)
# library("mvtnorm")
# mu <- 1:2
# Sigma <- matrix(c(4, 2, 2, 3), ncol=2)
# dmvt(c(-2., 3.), delta=mu, sigma=Sigma, df=1)
rvalues = [-5.6561739738159975133,
           -5.4874952805811396672,
           -5.4441948098568158088,
           -5.432461875138580254,
           -5.4585441614404803801]
df = [1., 2, 3, 5, 10]
for i = 1:length(df)
    d = MvTDist(df[i], mu, Sigma)
    @test isapprox(logpdf(d, [-2., 3]), rvalues[i], atol=1.0e-8)
    dd = typeof(d)(params(d)...)
    @test d.df == dd.df
    @test Vector(d.μ) == Vector(dd.μ)
    @test Matrix(d.Σ) == Matrix(dd.Σ)
end

# test constructors for mixed inputs:
@test typeof(GenericMvTDist(1, Vector{Float32}(mu), PDMat(Sigma))) == typeof(GenericMvTDist(1., mu, PDMat(Sigma)))

@test typeof(GenericMvTDist(1, mu, PDMat(Array{Float32}(Sigma)))) == typeof(GenericMvTDist(1., mu, PDMat(Sigma)))

d = GenericMvTDist(1, Array{Float32}(mu), PDMat(Array{Float32}(Sigma)))
@test convert(GenericMvTDist{Float32}, d) === d
@test typeof(convert(GenericMvTDist{Float64}, d)) == typeof(GenericMvTDist(1, mu, PDMat(Sigma)))
@test typeof(convert(GenericMvTDist{Float64}, d.df, d.dim, d.μ, d.Σ)) == typeof(GenericMvTDist(1, mu, PDMat(Sigma)))
@test partype(d) == Float32
@test d == deepcopy(d)

@test size(rand(MvTDist(1., mu, Sigma))) == (2,)
@test size(rand(MvTDist(1., mu, Sigma), 10)) == (2,10)
@test size(rand(MersenneTwister(123), MvTDist(1., mu, Sigma))) == (2,)
@test size(rand(MersenneTwister(123), MvTDist(1., mu, Sigma), 10)) == (2,10)

# static array for mean/variance
mu_static = @SVector [1., 2]
# depends on PDMats#101 (merged but not released)
# Sigma_static = @SMatrix [4. 2; 2 3]

for i in 1:length(df)
    d = GenericMvTDist(df[i], mu_static, PDMat(Sigma))
    d32 = convert(GenericMvTDist{Float32}, d)
    @test d.μ isa SVector
    @test isapprox(logpdf(d, [-2., 3]), rvalues[i], atol=1.0e-8)
    @test isa(logpdf(d32, [-2f0, 3f0]), Float32)
    @test isapprox(logpdf(d32, [-2f0, 3f0]), convert(Float32, rvalues[i]), atol=1.0e-4)
    dd = typeof(d)(params(d)...)
    @test d.df == dd.df
    @test d.μ == dd.μ
    @test Matrix(d.Σ) == Matrix(dd.Σ)
end

@testset "zero-mean" begin

    X_implicit = GenericMvTDist(2.0, PDMat(Sigma))
    X_expicit = GenericMvTDist(2.0, zeros(2), PDMat(Sigma))

    # Check that the means equal the same thing.
    @test mean(X_expicit) == mean(X_implicit)

    # Check that generated random numbers are the same.
    @test isapprox(
        rand(MersenneTwister(123456), X_expicit),
        rand(MersenneTwister(123456), X_implicit),
    )

    # Check that the logpdf computed is the same.
    x = rand(X_implicit)
    @test logpdf(X_implicit, x) ≈ logpdf(X_expicit, x)
end

@testset "mvtdist constructors" begin
    df = 3.0
    μ = [1.0, 2, 3]
    k = length(μ)
    σ = 2.0
    σv = [1.0, 2, 3]
    Σ = [4.0 2 1; 2 3 0.5; 1 0.5 2]
    x = [0.5, -1.0, 2.0]
    z = zeros(k)

    # (description, distribution, expected mean parameter, expected scale matrix)
    cases = [
        ("(df, μ, PDMat)", mvtdist(df, μ, PDMat(Σ)), μ, Σ),
        ("(df, μ, PDiagMat)", mvtdist(df, μ, PDiagMat(σv .^ 2)), μ, Matrix(Diagonal(σv .^ 2))),
        ("(df, μ, ScalMat)", mvtdist(df, μ, ScalMat(k, σ^2)), μ, σ^2 * Matrix(I, k, k)),
        ("(df, PDMat)", mvtdist(df, PDMat(Σ)), z, Σ),
        ("(df, PDiagMat)", mvtdist(df, PDiagMat(σv .^ 2)), z, Matrix(Diagonal(σv .^ 2))),
        ("(df, μ, σ::Real)", mvtdist(df, μ, σ), μ, σ^2 * Matrix(I, k, k)),
        ("(df, d::Int, σ)", mvtdist(df, k, σ), z, σ^2 * Matrix(I, k, k)),
        ("(df, μ, σ::Vector)", mvtdist(df, μ, σv), μ, Matrix(Diagonal(σv .^ 2))),
        ("(df, μ, Σ::Matrix)", mvtdist(df, μ, Σ), μ, Σ),
        ("(df, Σ::Matrix)", mvtdist(df, Σ), z, Σ),
        # mixed / integer inputs
        ("Int df", mvtdist(3, μ, Σ), μ, Σ),
        ("Int μ", mvtdist(df, [1, 2, 3], Σ), μ, Σ),
        ("Int df, Int σ", mvtdist(3, k, 2), z, 4 * Matrix(I, k, k)),
    ]

    @testset "$name" for (name, d, μ_exp, Σ_exp) in cases
        @test d isa GenericMvTDist
        @test length(d) == k
        @test d.df == df
        @test Vector(d.μ) ≈ μ_exp
        @test Matrix(d.Σ) ≈ Σ_exp

        # agrees with building the distribution directly
        ref = GenericMvTDist(df, μ_exp, PDMat(Σ_exp))
        @test logpdf(d, x) ≈ logpdf(ref, x)
        @test cov(d) ≈ cov(ref)
        @test mean(d) ≈ mean(ref)

        # sampling works with the right shapes
        @test size(rand(d)) == (k,)
        @test size(rand(d, 5)) == (k, 5)
    end

    # the specializations for plain Float64 inputs return the concrete aliases
    @test mvtdist(df, μ, Σ) isa MvTDist
    @test mvtdist(df, μ, σv) isa DiagTDist
    @test Distributions.MvTDist(df, μ, Σ) isa MvTDist

    # invalid arguments are rejected
    @test_throws ErrorException mvtdist(-1.0, μ, Σ)       # df must be positive
    @test_throws DimensionMismatch mvtdist(df, [1.0, 2], PDMat(Σ))  # μ / Σ size mismatch
end

end
