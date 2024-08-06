# Test the extension interface described in https://juliastats.org/Distributions.jl/stable/extends/#Create-New-Samplers-and-Distributions

module Extensions

using Distributions
using Random

### Samplers

## Univariate Sampler
struct Dirac1Sampler{T} <: Sampleable{Univariate, Continuous}
    x::T
end

Distributions.rand(::AbstractRNG, s::Dirac1Sampler) = s.x

## Multivariate Sampler
struct DiracNSampler{T} <: Sampleable{Multivariate, Continuous}
    x::Vector{T}
end

Base.length(s::DiracNSampler) = length(s.x)
Distributions._rand!(::AbstractRNG, s::DiracNSampler, x::AbstractVector) = x .= s.x

## Matrix-variate sampler
struct DiracMVSampler{T} <: Sampleable{Matrixvariate, Continuous}
    x::Matrix{T}
end

Base.size(s::DiracMVSampler) = size(s.x)
Distributions._rand!(::AbstractRNG, s::DiracMVSampler, x::AbstractMatrix) = x .= s.x



### Distributions

## Univariate distribution
struct Dirac1{T} <: ContinuousUnivariateDistribution
    x::T
end

# required methods
Distributions.rand(::AbstractRNG, d::Dirac1) = d.x
Distributions.logpdf(d::Dirac1, x::Real) = x == d.x ? Inf : 0.0
Distributions.cdf(d::Dirac1, x::Real) = x < d.x ? false : true
function Distributions.quantile(d::Dirac1, p::Real)
    (p < zero(p) || p > oneunit(p)) && throw(DomainError())
    return iszero(p) ? typemin(d.x) : d.x
end
Distributions.minimum(d::Dirac1) = typemin(d.x)
Distributions.maximum(d::Dirac1) = typemax(d.x)
Distributions.insupport(d::Dirac1, x::Real) = minimum(d) < x < maximum(d)

# recommended methods
Distributions.mean(d::Dirac1) = d.x
Distributions.var(d::Dirac1) = zero(d.x)
Distributions.mode(d::Dirac1) = d.x
# Distributions.modes(d::Dirac1) = [mode(d)]   # test the fallback
Distributions.skewness(d::Dirac1) = zero(d.x)
Distributions.kurtosis(d::Dirac1, ::Bool) = zero(d.x)   # conceived as the limit of a Gaussian for σ → 0
Distributions.entropy(d::Dirac1) = zero(d.x)
Distributions.mgf(d::Dirac1, t::Real) = exp(t * d.x)
Distributions.cf(d::Dirac1, t::Real) = exp(t * d.x * im)


## Multivariate distribution
struct DiracN{T} <: ContinuousMultivariateDistribution
    x::Vector{T}
end

# required methods
Base.length(d::DiracN) = length(d.x)
Base.eltype(::DiracN{T}) where T = T
Distributions._rand!(::AbstractRNG, d::DiracN, x::AbstractVector) = x .= d.x
Distributions._rand!(::AbstractRNG, d::DiracN, x::AbstractMatrix) = x .= d.x
Distributions._logpdf(d::DiracN, x::AbstractVector{<:Real}) = x == d.x ? Inf : 0.0
Distributions._logpdf(d::DiracN, x::AbstractMatrix{<:Real}) = map(y -> y == d.x ? Inf : 0.0, eachcol(x))

# recommended methods
Distributions.mean(d::DiracN) = d.x
Distributions.var(d::DiracN) = zero(d.x)
Distributions.entropy(::DiracN{T}) where T = zero(T)
Distributions.cov(d::DiracN) = zero(d.x) * zero(d.x)'


## Matrix-variate distribution
struct DiracMV{T} <: ContinuousMatrixDistribution
    x::Matrix{T}
end

# required methods
Base.size(d::DiracMV) = size(d.x)
Distributions._rand!(::AbstractRNG, d::DiracMV, x::AbstractMatrix) = x .= d.x
Distributions._logpdf(d::DiracMV, x::AbstractMatrix{<:Real}) = x == d.x ? Inf : 0.0


end # module Extensions

using Distributions
using Random
using Test

@testset "Extensions" begin
    ## Samplers
    # Univariate
    s = Extensions.Dirac1Sampler(1.0)
    @test rand(s) == 1.0
    @test rand(s, 5) == ones(5)
    @test rand!(s, zeros(5)) == ones(5)
    # Multivariate
    s = Extensions.DiracNSampler([1.0, 2.0, 3.0])
    @test rand(s) == [1.0, 2.0, 3.0]
    @test rand(s, 5) == rand!(s, zeros(3, 5)) == repeat([1.0, 2.0, 3.0], 1, 5)
    # Matrix-variate
    s = Extensions.DiracMVSampler([1.0 2.0 3.0; 4.0 5.0 6.0])
    @test rand(s) == [1.0 2.0 3.0; 4.0 5.0 6.0]
    @test rand(s, 5) == rand!(s, [zeros(2, 3) for i=1:5]) == [[1.0 2.0 3.0; 4.0 5.0 6.0] for i = 1:5]

    ## Distributions
    # Univariate
    d = Extensions.Dirac1(1.0)
    @test rand(d) == 1.0
    @test rand(d, 5) == ones(5)
    @test rand!(d, zeros(5)) == ones(5)
    @test logpdf(d, 1.0) == Inf
    @test logpdf(d, 2.0) == 0.0
    @test cdf(d, 0.0) == false
    @test cdf(d, 1.0) == true
    @test cdf(d, 2.0) == true
    @test quantile(d, 0.0) == -Inf
    @test quantile(d, 0.5) == 1.0
    @test quantile(d, 1.0) == 1.0
    @test minimum(d) == -Inf
    @test maximum(d) == Inf
    @test insupport(d, 0.0) == true
    @test insupport(d, 1.0) == true
    @test insupport(d, -Inf) == false
    @test mean(d) == 1.0
    @test var(d) == 0.0
    @test mode(d) == 1.0
    @test skewness(d) == 0.0
    @test_broken kurtosis(d) == 0.0
    @test entropy(d) == 0.0
    @test mgf(d, 0.0) == 1.0
    @test mgf(d, 1.0) == exp(1.0)
    @test cf(d, 0.0) == 1.0
    @test cf(d, 1.0) == exp(im)
    # MixtureModel of Univariate
    d = MixtureModel([Extensions.Dirac1(1.0), Extensions.Dirac1(2.0), Extensions.Dirac1(3.0)])
    @test rand(d) ∈ (1.0, 2.0, 3.0)
    @test all(∈((1.0, 2.0, 3.0)), rand(d, 5))
    @test all(∈((1.0, 2.0, 3.0)), rand!(d, zeros(5)))
    @test logpdf(d, 1.5) == 0.0
    @test logpdf(d, 2) == Inf
    @test logpdf(d, [0.5, 2.0, 2.5]) == [0.0, Inf, 0.0]
    @test mean(d) == 2

    # Multivariate
    d = Extensions.DiracN([1.0, 2.0, 3.0])
    @test length(d) == 3
    @test eltype(d) == Float64
    @test rand(d) == [1.0, 2.0, 3.0]
    @test rand(d, 5) == rand!(d, zeros(3, 5)) == repeat([1.0, 2.0, 3.0], 1, 5)
    @test logpdf(d, [1.0, 2, 3]) == Inf
    @test logpdf(d, [1.0, 2, 4]) == 0.0
    @test logpdf(d, [1.0 1; 2 2; 3 4]) == [Inf, 0.0]
    @test mean(d) == [1.0, 2.0, 3.0]
    @test var(d) == [0.0, 0.0, 0.0]
    @test entropy(d) == 0.0
    @test cov(d) == zeros(3, 3)
    # Mixture model of multivariate
    d = MixtureModel([Extensions.DiracN([1.0, 2.0, 3.0]), Extensions.DiracN([4.0, 5.0, 6.0])])
    @test rand(d) ∈ ([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
    @test all(∈(([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])), eachcol(rand(d, 5)))
    @test all(∈(([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])), eachcol(rand!(d, zeros(3, 5))))
    @test logpdf(d, [1.0, 2, 3]) == Inf
    @test logpdf(d, [4.0, 5, 6]) == Inf
    @test logpdf(d, [1.0, 2, 4]) == 0.0

    # Matrix-variate
    d = Extensions.DiracMV([1.0 2.0 3.0; 4.0 5.0 6.0])
    @test size(d) == (2, 3)
    @test rand(d) == [1.0 2.0 3.0; 4.0 5.0 6.0]
    @test rand(d, 5) == rand!(d, [zeros(2, 3) for i=1:5]) == [[1.0 2.0 3.0; 4.0 5.0 6.0] for i = 1:5]
    @test logpdf(d, [1.0 2.0 3.0; 4.0 5.0 6.0]) == Inf
    @test logpdf(d, [1.0 2.0 3.0; 4.0 5.0 7.0]) == 0.0
    @test logpdf(d, [[1.0 2.0 3.0; 4.0 5.0 7.0], [1.0 2.0 3.0; 4.0 5.0 6.0]]) == [0.0, Inf]
    # Mixtures of matrix-variate
    d = MixtureModel([Extensions.DiracMV([1.0 2.0 3.0; 4.0 5.0 6.0]), Extensions.DiracMV([7.0 8.0 9.0; 10.0 11.0 12.0])])
    @test_broken rand(d) ∈ ([1.0 2.0 3.0; 4.0 5.0 6.0], [7.0 8.0 9.0; 10.0 11.0 12.0])
    @test_broken all(∈(([1.0 2.0 3.0; 4.0 5.0 6.0], [7.0 8.0 9.0; 10.0 11.0 12.0])), eachslice(rand(d, 5), dims=3))
end
