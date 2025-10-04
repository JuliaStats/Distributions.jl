"""

The [symmetric multivariate Laplace distribution](https://en.wikipedia.org/wiki/Multivariate_Laplace_distribution#Symmetric_multivariate_Laplace_distribution)
is a multidimensional generalization of the *Laplace distribution*, as described in 
T. Eltoft, Taesu Kim and Te-Won Lee, 
"On the multivariate Laplace distribution," 
in IEEE Signal Processing Letters, vol. 13, no. 5, pp. 300-303, May 2006, doi: 10.1109/LSP.2006.870353.
The symmetric multivariate Laplace distribution is defined by three parameters:
- ``\\lambda``, which is a positive real number used to define an exponential distribution `Exp(λ)`
- ``\\boldsymbol{\\Gamma}``, which is a k-by-k positive definite matrix with `det(Γ)=1` (as per the assumptions in the source paper)
- ``\\boldsymbol{\\mu}``, which is a k-dimensional real-valued vector
The expected valued of the symmetric multivariate Laplace distribution is simply ``\\boldsymbol{\\mu}``,
whereas the covariance ``\\boldsymbol{\\Sigma} = \\lambda \\boldsymbol{\\Gamma}``.

The symmetric multivariate Laplace distribution can be created by specifying either a ``\\boldsymbol{\\mu}`` and a ``\\boldsymbol{\\Sigma}`` 
(analogously to a `MvNormal`) and the ``\\lambda`` and ``\\boldsymbol{\\Gamma}`` are calculated internally,
or by specifying all three parameters, ``\\boldsymbol{\\mu}``, ``\\lambda`` and ``\\boldsymbol{\\Gamma}``.

The probability density function of
a k-dimensional symmetric multivariate Laplace distribution with parameters ``\\boldsymbol{\\mu}``,  ``\\lambda`` and ``\\boldsymbol{\\Gamma}`` is:
```math
f(\\mathbf{x}; \\boldsymbol{\\mu}, \\lambda, \\boldsymbol{\\Gamma}) = \\frac{1}{(2 \\pi)^{d/2}} \\frac{2}{\\lambda}
\frac{\\mathrm{K}_{(d/2)-1}\\left(\\sqrt{\\frac{2}{\\lambda}q(\\mathbf{x})}\\right)}{\\left(\\sqrt{\\frac{2}{\\lambda}q(\\mathbf{x})}\\right)^{(d/2)-1}}
```
where ``\\mathrm{K}_m (y)`` is the Bessel function of the second kind of order ``m`` evaluated at ``y`` and
```math
q(\\mathbf{x} =  (\\mathbf{x} - \\boldsymbol{\\mu})^T \\Gamma^{-1} (\\mathbf{x} - \\boldsymbol{\\mu})
```
"""
struct SymmetricMvLaplace{T<:Real,Cov<:AbstractPDMat,iCov<:AbstractMatrix,Mean<:AbstractVector} <: ContinuousMultivariateDistribution
    μ::Mean
    Γ::Cov
    iΓ::Cov
    λ::T
end

### generic methods for SymmetricMvLaplace
length(d::SymmetricMvLaplace) = length(d.μ)
params(d::SymmetricMvLaplace) = (d.μ, d.λ, d.Γ)

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

### Construction when all three parameters are specified
function SymmetricMvLaplace(μ::AbstractVector{<:Real}, λ::Real, Γ::AbstractPDMat{<:Real})
    size(Γ, 1) == length(μ) || throw(DimensionMismatch("The dimensions of mu and Gamma are inconsistent."))
    isapprox(det(Γ), 1) || throw(ArgumentError("det(Gamma) is not approximately equal to 1."))
    λ > 0 || throw(ArgumentError("λ is not a positive real number."))

    R = Base.promote_eltype(μ, λ, Γ)
    _μ, _λ, _Γ = convert(AbstractArray{R}, μ), convert(R, λ), convert(AbstractArray{R}, Γ)
    _iΓ = inv(_Γ) 
    SymmetricMvLaplace{R,typeof(_Γ), typeof(_iΓ), typeof(_μ)}(_μ, _Γ, _iΓ, _λ)
end

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

# constructor with general gamma matrix and lambda
SymmetricMvLaplace(μ::AbstractVector{<:Real}, λ::Real, Γ::AbstractMatrix{<:Real}) = SymmetricMvLaplace(μ, λ, PDMat(Γ))
SymmetricMvLaplace(μ::AbstractVector{<:Real}, λ::Real, Γ::Diagonal{<:Real}) = SymmetricMvLaplace(μ, λ, PDiagMat(Γ.diag))
SymmetricMvLaplace(μ::AbstractVector{<:Real}, λ::Real, Γ::Union{Symmetric{<:Real,<:Diagonal{<:Real}},Hermitian{<:Real,<:Diagonal{<:Real}}}) = SymmetricMvLaplace(μ, λ, PDiagMat(Γ.data.diag))
SymmetricMvLaplace(μ::AbstractVector{<:Real}, λ::Real, Γ::UniformScaling{<:Real}) =
    SymmetricMvLaplace(μ, λ, ScalMat(length(μ), Γ.λ))
function SymmetricMvLaplace(
    μ::AbstractVector{<:Real}, λ::Real, Γ::Diagonal{<:Real,<:FillArrays.AbstractFill{<:Real,1}}
)
    return SymmetricMvLaplace(μ, λ, ScalMat(size(Γ, 1), FillArrays.getindex_value(Γ.diag)))
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
    x .*= sqrt.(rand(rng, Exponential(d.λ), 1, size(x,2)))
    x .+=  d.μ
    return x
end

function _logpdf(d::SymmetricMvLaplace, x::AbstractArray)
    _d = length(d) / 2
    xdif = x - d.μ
    q = dot(xdif, d.iΓ, xdif)
    return _d * log(2π) + log(2/d.λ) + log(besselk(_d-1, sqrt(2/d.λ * q))) + 0.5*(_d-1)*log(0.5*d.λ*q)
end