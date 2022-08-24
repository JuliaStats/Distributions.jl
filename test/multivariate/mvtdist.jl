using Distributions, Random, StaticArrays, LinearAlgebra
using Test

import Distributions: GenericMvTDist
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

@testset "MvNormal affine tranformations" begin
    @testset "moment identities" begin
        for n in 1:5                       # dimension
            # distribution
            μ = randn(n)
            ν = 4
            for Σ in (randn(n, n) |> A -> A*A',  # dense
                      Diagonal(abs2.(randn(n)))) # diagonal
                d = GenericMvTDist(ν, μ, PDMat(Σ))
                
                # random arrays for transformations
                c = randn(n)
                m = rand(1:n)
                B = randn(m, n)
                b = randn(n)

                d_c = d + c
                c_d = c + d
                @test mean(d_c) == mean(c_d) == μ + c
                @test cov(c_d) == cov(d_c) == cov(d)

                d_c = d - c
                @test mean(d_c) == μ - c
                @test scale(d_c) == scale(d)

                B_d = B * d
                @test B_d isa GenericMvTDist
                @test length(B_d) == m
                @test mean(B_d) == B * μ
                @test scale(B_d) ≈ B * Σ * B'

                d_trans = B * (d + c)
                d_trans == GenericMvTDist(ν, B * (μ + c), PDMat(X_A_Xt(d.Σ, B)))
            end
        end
    end

    @testset "dimension mismatch errors" begin
        d4 = GenericMvTDist(4.5, zeros(4), PDMat(Diagonal(ones(4))))
        o3 = ones(3)
        @test_throws DimensionMismatch d4 + o3
        @test_throws DimensionMismatch o3 + d4
        @test_throws DimensionMismatch ones(3, 3) * d4
    end
end
end
