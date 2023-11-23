# Multivariate LogNormal distribution

###########################################################
#
#   Abstract base class for multivariate lognormal
#
#   Each subtype should provide the following methods:
#
#   - length(d)         vector dimension
#   - params(d)         Get the parameters from the underlying Normal distribution
#   - location(d)       Location parameter
#   - scale(d)          Scale parameter
#   - _rand!(d, x)      Sample random vector(s)
#   - _logpdf(d,x)      Evaluate logarithm of pdf
#   - _pdf(d,x)         Evaluate the pdf
#   - mean(d)           Mean of the distribution
#   - median(d)         Median of the distribution
#   - mode(d)           Mode of the distribution
#   - var(d)            Vector of element-wise variance
#   - cov(d)            Covariance matrix
#   - entropy(d)        Compute the entropy
#
#
###########################################################

abstract type AbstractMvLogNormal <: ContinuousMultivariateDistribution end

function insupport(::Type{D},x::AbstractVector{T}) where {T<:Real,D<:AbstractMvLogNormal}
    for i=1:length(x)
      @inbounds 0.0<x[i]<Inf ? continue : (return false)
    end
    true
end
insupport(l::AbstractMvLogNormal,x::AbstractVector{T}) where {T<:Real} = insupport(typeof(l),x)
assertinsupport(::Type{D},m::AbstractVector) where {D<:AbstractMvLogNormal} = @assert insupport(D,m) "Mean of LogNormal distribution should be strictly positive"

###Internal functions to calculate scale and location for a desired average and covariance
function _location!(::Type{D},::Type{Val{:meancov}},mn::AbstractVector,S::AbstractMatrix,μ::AbstractVector) where D<:AbstractMvLogNormal
    @simd for i=1:length(mn)
      @inbounds μ[i] = log(mn[i]/sqrt(1+S[i,i]/mn[i]/mn[i]))
    end
    μ
end

function _scale!(::Type{D},::Type{Val{:meancov}},mn::AbstractVector,S::AbstractMatrix,Σ::AbstractMatrix) where D<:AbstractMvLogNormal
    for j=1:length(mn)
      @simd for i=j:length(mn)
        @inbounds Σ[i,j] = Σ[j,i] = log(1 + S[j,i]/mn[i]/mn[j])
      end
    end
    Σ
end

function _location!(::Type{D},::Type{Val{:mean}},mn::AbstractVector,S::AbstractMatrix,μ::AbstractVector) where D<:AbstractMvLogNormal
    @simd for i=1:length(mn)
      @inbounds μ[i] = log(mn[i]) - S[i,i]/2
    end
    μ
end

function _location!(::Type{D},::Type{Val{:median}},md::AbstractVector,S::AbstractMatrix,μ::AbstractVector) where D<:AbstractMvLogNormal
    @simd for i=1:length(md)
      @inbounds μ[i] = log(md[i])
    end
    μ
end

function _location!(::Type{D},::Type{Val{:mode}},mo::AbstractVector,S::AbstractMatrix,μ::AbstractVector) where D<:AbstractMvLogNormal
    @simd for i=1:length(mo)
      @inbounds μ[i] = log(mo[i]) + S[i,i]
    end
    μ
end

###Functions to calculate location and scale for a distribution with desired :mean, :median or :mode and covariance
"""
    location!{D<:AbstractMvLogNormal}(::Type{D},s::Symbol,m::AbstractVector,S::AbstractMatrix,μ::AbstractVector)

Calculate the location vector (as above) and store the result in ``μ``
"""
function location!(::Type{D},s::Symbol,m::AbstractVector,S::AbstractMatrix,μ::AbstractVector) where D<:AbstractMvLogNormal
    @assert size(S) == (length(m),length(m)) && length(m) == length(μ)
    assertinsupport(D,m)
    _location!(D,Val{s},m,S,μ)
end

"""
    location{D<:AbstractMvLogNormal}(::Type{D},s::Symbol,m::AbstractVector,S::AbstractMatrix)

Calculate the location vector (the mean of the underlying normal distribution).

- If `s == :meancov`, then m is taken as the mean, and S the covariance matrix of a
  lognormal distribution.
- If `s == :mean | :median | :mode`, then m is taken as the mean, median or
  mode of the lognormal respectively, and S is interpreted as the scale matrix
  (the covariance of the underlying normal distribution).

It is not possible to analytically calculate the location vector from e.g., median + covariance,
or from mode + covariance.
"""
function location(::Type{D},s::Symbol,m::AbstractVector,S::AbstractMatrix) where D<:AbstractMvLogNormal
    @assert size(S) == (length(m),length(m))
    assertinsupport(D,m)
    _location!(D,Val{s},m,S,similar(m))
end

"""
    scale!{D<:AbstractMvLogNormal}(::Type{D},s::Symbol,m::AbstractVector,S::AbstractMatrix,Σ::AbstractMatrix)

Calculate the scale parameter, as defined for the location parameter above and store the result in `Σ`.
"""
function scale!(::Type{D},s::Symbol,m::AbstractVector,S::AbstractMatrix,Σ::AbstractMatrix) where D<:AbstractMvLogNormal
    @assert size(S) == size(Σ) == (length(m),length(m))
    assertinsupport(D,m)
    _scale!(D,Val{s},m,S,Σ)
end

"""
    scale{D<:AbstractMvLogNormal}(::Type{D},s::Symbol,m::AbstractVector,S::AbstractMatrix)

Calculate the scale parameter, as defined for the location parameter above.
"""
function scale(::Type{D},s::Symbol,m::AbstractVector,S::AbstractMatrix) where D<:AbstractMvLogNormal
    @assert size(S) == (length(m),length(m))
    assertinsupport(D,m)
    _scale!(D,Val{s},m,S,similar(S))
end

"""
    params!{D<:AbstractMvLogNormal}(::Type{D},m::AbstractVector,S::AbstractMatrix,μ::AbstractVector,Σ::AbstractMatrix)

Calculate (scale,location) for a given mean and covariance, and store the results in `μ` and `Σ`
"""
params!(::Type{D},m::AbstractVector,S::AbstractMatrix,μ::AbstractVector,Σ::AbstractMatrix) where {D<:AbstractMvLogNormal} = location!(D,:meancov,m,S,μ),scale!(D,:meancov,m,S,Σ)

"""
    params{D<:AbstractMvLogNormal}(::Type{D},m::AbstractVector,S::AbstractMatrix)

Return (scale,location) for a given mean and covariance
"""
params(::Type{D},m::AbstractVector,S::AbstractMatrix) where {D<:AbstractMvLogNormal} = params!(D,m,S,similar(m),similar(S))

#########################################################
#
#   MvLogNormal
#
#   Multivariate lognormal distribution based on MvNormal
#
#########################################################
"""
    MvLogNormal(d::MvNormal)

The [Multivariate lognormal distribution](http://en.wikipedia.org/wiki/Log-normal_distribution)
is a multidimensional generalization of the *lognormal distribution*.

If ``\\boldsymbol X \\sim \\mathcal{N}(\\boldsymbol\\mu,\\,\\boldsymbol\\Sigma)`` has a multivariate
normal distribution then ``\\boldsymbol Y=\\exp(\\boldsymbol X)`` has a multivariate lognormal
distribution.

Mean vector ``\\boldsymbol{\\mu}`` and covariance matrix ``\\boldsymbol{\\Sigma}`` of the
underlying normal distribution are known as the *location* and *scale*
parameters of the corresponding lognormal distribution.
"""
struct MvLogNormal{T<:Real,Cov<:AbstractPDMat,Mean<:AbstractVector} <: AbstractMvLogNormal
    normal::MvNormal{T,Cov,Mean}
end

# Constructors mirror the ones for MvNormmal
MvLogNormal(μ::AbstractVector{<:Real}, Σ::AbstractMatrix{<:Real}) = MvLogNormal(MvNormal(μ, Σ))
MvLogNormal(Σ::AbstractMatrix{<:Real}) = MvLogNormal(MvNormal(Σ))
MvLogNormal(μ::AbstractVector{<:Real}, Σ::UniformScaling{<:Real}) = MvLogNormal(MvNormal(μ, Σ))

# Deprecated constructors
MvLogNormal(μ::AbstractVector,σ::Vector) = MvLogNormal(MvNormal(μ,σ))
MvLogNormal(μ::AbstractVector,s::Real) = MvLogNormal(MvNormal(μ,s))
MvLogNormal(σ::AbstractVector) = MvLogNormal(MvNormal(σ))
MvLogNormal(d::Int,s::Real) = MvLogNormal(MvNormal(d,s))

Base.eltype(::Type{<:MvLogNormal{T}}) where {T} = T

### Conversion
function convert(::Type{MvLogNormal{T}}, d::MvLogNormal) where T<:Real
    MvLogNormal(convert(MvNormal{T}, d.normal))
end
Base.convert(::Type{MvLogNormal{T}}, d::MvLogNormal{T}) where {T<:Real} = d

function convert(::Type{MvLogNormal{T}}, pars...) where T<:Real
    MvLogNormal(convert(MvNormal{T}, MvNormal(pars...)))
end

length(d::MvLogNormal) = length(d.normal)
params(d::MvLogNormal) = params(d.normal)
@inline partype(d::MvLogNormal{T}) where {T<:Real} = T

"""
    location(d::MvLogNormal)

Return the location vector of the distribution (the mean of the underlying normal distribution).
"""
location(d::MvLogNormal) = mean(d.normal)

"""
    scale(d::MvLogNormal)

Return the scale matrix of the distribution (the covariance matrix of the underlying normal distribution).
"""
scale(d::MvLogNormal) = cov(d.normal)

#See https://en.wikipedia.org/wiki/Log-normal_distribution
mean(d::MvLogNormal) = exp.(mean(d.normal) .+ var(d.normal)/2)

"""
    median(d::MvLogNormal)

Return the median vector of the lognormal distribution. which is strictly smaller than the mean.
"""
median(d::MvLogNormal) = exp.(mean(d.normal))

"""
    mode(d::MvLogNormal)

Return the mode vector of the lognormal distribution, which is strictly smaller than the mean and median.
"""
mode(d::MvLogNormal) = exp.(mean(d.normal) .- var(d.normal))
function cov(d::MvLogNormal)
    m = mean(d)
    return m*m'.*(exp.(cov(d.normal)) .- 1)
end
var(d::MvLogNormal) = diag(cov(d))

#see Zografos & Nadarajah (2005) Stat. Prob. Let 71(1) pp71-84 DOI: 10.1016/j.spl.2004.10.023
entropy(d::MvLogNormal) = length(d)*(1+log2π)/2 + logdetcov(d.normal)/2 + sum(mean(d.normal))

#See https://en.wikipedia.org/wiki/Log-normal_distribution
function _rand!(rng::AbstractRNG, d::MvLogNormal, x::AbstractVecOrMat{<:Real})
    _rand!(rng, d.normal, x)
    map!(exp, x, x)
    return x
end

_logpdf(d::MvLogNormal, x::AbstractVecOrMat{T}) where {T<:Real} = insupport(d, x) ? (_logpdf(d.normal, log.(x)) - sum(log.(x))) : -Inf
_pdf(d::MvLogNormal, x::AbstractVecOrMat{T}) where {T<:Real} = insupport(d,x) ? _pdf(d.normal, log.(x))/prod(x) : 0.0

Base.show(io::IO,d::MvLogNormal) = show_multline(io, d, [(:dim, length(d)), (:μ, mean(d.normal)), (:Σ, cov(d.normal))])
