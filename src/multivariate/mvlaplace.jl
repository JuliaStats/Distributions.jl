### Implementation of the symmetric multivariate Laplace distribution using the formatlisation from:
###     T. Eltoft, Taesu Kim and Te-Won Lee, 
###     "On the multivariate Laplace distribution," 
###     in IEEE Signal Processing Letters, vol. 13, no. 5, pp. 300-303, May 2006, doi: 10.1109/LSP.2006.870353


struct SymmetricMvLaplace{T<:Real,Cov<:AbstractPDMat,iCov<:AbstractMatrix,Mean<:AbstractVector} <: ContinuousMultivariateDistribution
    μ::Mean
    Γ::Cov
    iΓ::Cov
    λ::T
end

### generic methods for SymmetricMvLaplace
length(d::SymmetricMvLaplace) = length(d.μ)
params(d::SymmetricMvLaplace) = (d.μ, d.Σ)

insupport(d::SymmetricMvLaplace, x::AbstractVector) =
    length(d) == length(x) && all(isfinite, x)

minimum(d::SymmetricMvLaplace) = fill(eltype(d)(-Inf), length(d))
maximum(d::SymmetricMvLaplace) = fill(eltype(d)(Inf), length(d))
mode(d::SymmetricMvLaplace) = mean(d)
modes(d::SymmetricMvLaplace) = [mean(d)]
mean(d::SymmetricMvLaplace) = d.μ
var(d::SymmetricMvLaplace) = diag(d.λ * d.Γ)
cov(d::SymmetricMvLaplace) = d.λ * d.Γ
invcov(d::SymmetricMvLaplace) = inv(d.λ * d.Γ)

### Construction when only an overall covariance is specified
function SymmetricMvLaplace(μ::AbstractVector{T}, Σ::AbstractPDMat{T}) where {T<:Real}
    size(Σ, 1) == length(μ) || throw(DimensionMismatch("The dimensions of mu and Sigma are inconsistent."))
    λ = det(Σ)^(1/size(Σ,1))
    Γ = 1/λ * Σ
    iΓ = inv(Γ)
    SymmetricMvLaplace{T,typeof(Γ), typeof(iΓ), typeof(μ)}(μ, Γ, iΓ, λ)
end

function SymmetricMvLaplace(μ::AbstractVector{<:Real}, Σ::AbstractPDMat{<:Real})
    R = Base.promote_eltype(μ, Σ)
    SymmetricMvLaplace(convert(AbstractArray{R}, μ), convert(AbstractArray{R}, Σ))
end

# constructor with general covariance matrix
SymmetricMvLaplace(μ::AbstractVector{<:Real}, Σ::AbstractMatrix{<:Real}) = SymmetricMvLaplace(μ, PDMat(Σ))
SymmetricMvLaplace(μ::AbstractVector{<:Real}, Σ::Diagonal{<:Real}) = SymmetricMvLaplace(μ, PDiagMat(Σ.diag))
SymmetricMvLaplace(μ::AbstractVector{<:Real}, Σ::Union{Symmetric{<:Real,<:Diagonal{<:Real}},Hermitian{<:Real,<:Diagonal{<:Real}}}) = SymmetricMvLaplace(μ, PDiagMat(Σ.data.diag))
SymmetricMvLaplace(μ::AbstractVector{<:Real}, Σ::UniformScaling{<:Real}) =
    SymmetricMvLaplace(μ, ScalMat(length(μ), Σ.λ))
function SymmetricMvLaplace(
    μ::AbstractVector{<:Real}, Σ::Diagonal{<:Real,<:FillArrays.AbstractFill{<:Real,1}}
)
    return SymmetricMvLaplace(μ, ScalMat(size(Σ, 1), FillArrays.getindex_value(Σ.diag)))
end

### generic methods for SymmetricMvLaplace
Base.eltype(::Type{<:SymmetricMvLaplace{T}}) where {T} = T
@inline partype(d::SymmetricMvLaplace{T}) where {T<:Real} = T

function _rand!(rng::AbstractRNG, d::SymmetricMvLaplace, x::VecOrMat)
    unwhiten!(d.Γ, randn!(rng, x))
    x .*= sqrt.(rand(Exponential(d.λ), 1, size(x,2)))
    x .+=  d.μ
    return x
end

function _logpdf(d::SymmetricMvLaplace, x::AbstractArray)
    dim_half = length(d) / 2
    _2byλ = 2/d.λ
    xdif = x - d.μ
    _qs = sqrt(_2byλ * dot(xdif, d.iΓ, xdif))
    return log(2π^(-dim_half)) + log(_2byλ) + bessely(dim_half-1, _qs) - log(_qs^(dim_half - 1))
end