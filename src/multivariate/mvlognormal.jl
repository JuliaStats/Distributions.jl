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

abstract AbstractMvLogNormal <: ContinuousMultivariateDistribution

function insupport{T<:Real,D<:AbstractMvLogNormal}(::Type{D},x::AbstractVector{T})
    for i=1:length(x)
      @inbounds 0.0<x[i]<Inf?continue:(return false)
    end
    true
end
insupport{T<:Real}(l::AbstractMvLogNormal,x::AbstractVector{T}) = insupport(typeof(l),x)
assertinsupport{D<:AbstractMvLogNormal}(::Type{D},m::AbstractVector) = @assert insupport(D,m) "Mean of LogNormal distribution should be strictly positive"

###Internal functions to calculate scale and location for a desired average and covariance
function _location!{D<:AbstractMvLogNormal}(::Type{D},::Type{Val{:meancov}},mn::AbstractVector,S::AbstractMatrix,μ::AbstractVector)
    @simd for i=1:length(mn)
      @inbounds μ[i] = log(mn[i]/sqrt(1+S[i,i]/mn[i]/mn[i]))
    end
    μ
end

function _scale!{D<:AbstractMvLogNormal}(::Type{D},::Type{Val{:meancov}},mn::AbstractVector,S::AbstractMatrix,Σ::AbstractMatrix)
    for j=1:length(mn)
      @simd for i=j:length(mn)
        @inbounds Σ[i,j] = Σ[j,i] = log(1 + S[j,i]/mn[i]/mn[j])
      end
    end
    Σ
end

function _location!{D<:AbstractMvLogNormal}(::Type{D},::Type{Val{:mean}},mn::AbstractVector,S::AbstractMatrix,μ::AbstractVector)
    @simd for i=1:length(mn)
      @inbounds μ[i] = log(mn[i]) - S[i,i]/2
    end
    μ
end

function _location!{D<:AbstractMvLogNormal}(::Type{D},::Type{Val{:median}},md::AbstractVector,S::AbstractMatrix,μ::AbstractVector)
    @simd for i=1:length(md)
      @inbounds μ[i] = log(md[i])
    end
    μ
end

function _location!{D<:AbstractMvLogNormal}(::Type{D},::Type{Val{:mode}},mo::AbstractVector,S::AbstractMatrix,μ::AbstractVector)
    @simd for i=1:length(mo)
      @inbounds μ[i] = log(mo[i]) + S[i,i]
    end
    μ
end

###Functions to calculate location and scale for a distribution with desired :mean, :median or :mode and covariance
function location!{D<:AbstractMvLogNormal}(::Type{D},s::Symbol,m::AbstractVector,S::AbstractMatrix,μ::AbstractVector)
  @assert size(S) == (length(m),length(m)) && length(m) == length(μ)
  assertinsupport(D,m)
  _location!(D,Val{s},m,S,μ)
end

function location{D<:AbstractMvLogNormal}(::Type{D},s::Symbol,m::AbstractVector,S::AbstractMatrix)
    @assert size(S) == (length(m),length(m))
    assertinsupport(D,m)
    _location!(D,Val{s},m,S,similar(m))
end

function scale!{D<:AbstractMvLogNormal}(::Type{D},s::Symbol,m::AbstractVector,S::AbstractMatrix,Σ::AbstractMatrix)
    @assert size(S) == size(Σ) == (length(m),length(m))
    assertinsupport(D,m)
    _scale!(D,Val{s},m,S,Σ)
end

function scale{D<:AbstractMvLogNormal}(::Type{D},s::Symbol,m::AbstractVector,S::AbstractMatrix)
    @assert size(S) == (length(m),length(m))
    assertinsupport(D,m)
    _scale!(D,Val{s},m,S,similar(S))
end

params!{D<:AbstractMvLogNormal}(::Type{D},m::AbstractVector,S::AbstractMatrix,μ::AbstractVector,Σ::AbstractMatrix) = location!(D,:meancov,m,S,μ),scale!(D,:meancov,m,S,Σ)
params{D<:AbstractMvLogNormal}(::Type{D},m::AbstractVector,S::AbstractMatrix) = params!(D,m,S,similar(m),similar(S))

#########################################################
#
#   MvLogNormal
#
#   Multivariate lognormal distribution based on MvNormal
#
#########################################################
immutable MvLogNormal{T<:Real,Cov<:AbstractPDMat,Mean<:Union{Vector, ZeroVector}} <: AbstractMvLogNormal
    normal::MvNormal{T,Cov,Mean}
end

#Constructors mirror the ones for MvNormmal
MvLogNormal(μ::Union{Vector,ZeroVector},Σ::AbstractPDMat) = MvLogNormal(MvNormal(μ,Σ))
MvLogNormal(Σ::AbstractPDMat) = MvLogNormal(MvNormal(ZeroVector(eltype(Σ),dim(Σ)),Σ))
MvLogNormal(μ::Vector,Σ::Matrix) = MvLogNormal(MvNormal(μ,Σ))
MvLogNormal(μ::Vector,σ::Vector) = MvLogNormal(MvNormal(μ,σ))
MvLogNormal(μ::Vector,s::Real) = MvLogNormal(MvNormal(μ,s))
MvLogNormal(Σ::Matrix) = MvLogNormal(MvNormal(Σ))
MvLogNormal(σ::Vector) = MvLogNormal(MvNormal(σ))
MvLogNormal(d::Int,s::Real) = MvLogNormal(MvNormal(d,s))

### Conversion
function convert{T<:Real}(::Type{MvLogNormal{T}}, d::MvLogNormal)
    MvLogNormal(convert(MvNormal{T}, d.normal))
end
function convert{T<:Real}(::Type{MvLogNormal{T}}, pars...)
    MvLogNormal(convert(MvNormal{T}, MvNormal(pars...)))
end

length(d::MvLogNormal) = length(d.normal)
params(d::MvLogNormal) = params(d.normal)
@inline partype{T<:Real}(d::MvLogNormal{T}) = T
location(d::MvLogNormal) = mean(d.normal)
scale(d::MvLogNormal) = cov(d.normal)

#See https://en.wikipedia.org/wiki/Log-normal_distribution
mean(d::MvLogNormal) = @compat(exp.(mean(d.normal) + var(d.normal)/2))
median(d::MvLogNormal) = @compat(exp.(mean(d.normal)))
mode(d::MvLogNormal) = @compat(exp.(mean(d.normal) - var(d.normal)))
function cov(d::MvLogNormal)
    m = mean(d)
    return m*m'.*(@compat(exp.(cov(d.normal))) - 1)
  end
var(d::MvLogNormal) = diag(cov(d))

#see Zografos & Nadarajah (2005) Stat. Prob. Let 71(1) pp71-84 DOI: 10.1016/j.spl.2004.10.023
entropy(d::MvLogNormal) = length(d)*(1+log2π)/2 + logdetcov(d.normal)/2 + sum(mean(d.normal))

#See https://en.wikipedia.org/wiki/Log-normal_distribution
_rand!{T<:Real}(d::MvLogNormal, x::AbstractVecOrMat{T}) = @compat(exp!(_rand!(d.normal, x)))
@compat _logpdf{T<:Real}(d::MvLogNormal, x::AbstractVecOrMat{T}) = insupport(d, x) ? (_logpdf(d.normal, log.(x)) - sum(log.(x))) : -Inf
_pdf{T<:Real}(d::MvLogNormal, x::AbstractVecOrMat{T}) = insupport(d,x) ? _pdf(d.normal, @compat(log.(x)))/prod(x) : 0.0

Base.show(io::IO,d::MvLogNormal) = show_multline(io, d, [(:dim, length(d)), (:μ, mean(d.normal)), (:Σ, cov(d.normal))])
