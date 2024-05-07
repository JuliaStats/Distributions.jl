module DistributionsTestExt

using Distributions
using Distributions.LinearAlgebra
using Distributions.Random
using Test

__rand(::Nothing, args...) = rand(args...)
__rand(rng::AbstractRNG, args...) = rand(rng, args...)

__rand!(::Nothing, args...) = rand!(args...)
__rand!(rng::AbstractRNG, args...) = rand!(rng, args...)

"""
    test_mvnormal(
        g::AbstractMvNormal,
        n_tsamples::Int=10^6,
        rng::Union{Random.AbstractRNG, Nothing}=nothing,
    )

Test that `AbstractMvNormal` implements the expected API.

!!! Note
    On Julia >= 1.9, you have to load the `Test` standard library to be able to use
    this function.
"""
function Distributions.TestUtils.test_mvnormal(
    g::AbstractMvNormal, n_tsamples::Int=10^6, rng::Union{AbstractRNG, Nothing}=nothing
)
    d = length(g)
    μ = mean(g)
    Σ = cov(g)
    @test length(μ) == d
    @test size(Σ) == (d, d)
    @test var(g) ≈ diag(Σ)
    @test entropy(g) ≈ 0.5 * logdet(2π * ℯ * Σ)
    ldcov = logdetcov(g)
    @test ldcov ≈ logdet(Σ)
    vs = diag(Σ)
    @test g == typeof(g)(params(g)...)
    @test g == deepcopy(g)
    @test minimum(g) == fill(-Inf, d)
    @test maximum(g) == fill(Inf, d)
    @test extrema(g) == (minimum(g), maximum(g))
    @test isless(extrema(g)...)

    # test sampling for AbstractMatrix (here, a SubArray):
    subX = view(__rand(rng, d, 2d), :, 1:d)
    @test isa(__rand!(rng, g, subX), SubArray)

    # sampling
    @test isa(__rand(rng, g), Vector{Float64})
    X = __rand(rng, g, n_tsamples)
    emp_mu = vec(mean(X, dims=2))
    Z = X .- emp_mu
    emp_cov = (Z * Z') * inv(n_tsamples)

    mean_atols = 8 .* sqrt.(vs ./ n_tsamples)
    cov_atols = 10 .* sqrt.(vs .* vs') ./ sqrt.(n_tsamples)
    for i = 1:d
        @test isapprox(emp_mu[i], μ[i], atol=mean_atols[i])
    end
    for i = 1:d, j = 1:d
        @test isapprox(emp_cov[i,j], Σ[i,j], atol=cov_atols[i,j])
    end

    X = rand(MersenneTwister(14), g, n_tsamples)
    Y = rand(MersenneTwister(14), g, n_tsamples)
    @test X == Y
    emp_mu = vec(mean(X, dims=2))
    Z = X .- emp_mu
    emp_cov = (Z * Z') * inv(n_tsamples)
    for i = 1:d
        @test isapprox(emp_mu[i]   , μ[i]  , atol=mean_atols[i])
    end
    for i = 1:d, j = 1:d
        @test isapprox(emp_cov[i,j], Σ[i,j], atol=cov_atols[i,j])
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
    @test loglikelihood(g, X) ≈ sum(i -> Distributions._logpdf(g, X[:,i]), 1:n_tsamples)
    @test loglikelihood(g, X[:, 1]) ≈ logpdf(g, X[:, 1])
    @test loglikelihood(g, [X[:, i] for i in axes(X, 2)]) ≈ loglikelihood(g, X)
end

end # module