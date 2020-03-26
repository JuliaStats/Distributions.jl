# Multivariate Normal distribution


###########################################################
#
#   Abstract base class for multivariate normal
#
#   Each subtype should provide the following methods:
#
#   - length(d):        vector dimension
#   - mean(d):          the mean vector (in full form)
#   - cov(d):           the covariance matrix (in full form)
#   - invcov(d):        inverse of covariance
#   - logdetcov(d):     log-determinant of covariance
#   - sqmahal(d, x):        Squared Mahalanobis distance to center
#   - sqmahal!(r, d, x):    Squared Mahalanobis distances
#   - gradlogpdf(d, x):     Gradient of logpdf w.r.t. x
#   - _rand!(d, x):         Sample random vector(s)
#
#   Other generic functions will be implemented on top
#   of these core functions.
#
###########################################################

"""

The [Multivariate normal distribution](http://en.wikipedia.org/wiki/Multivariate_normal_distribution)
is a multidimensional generalization of the *normal distribution*. The probability density function of
a d-dimensional multivariate normal distribution with mean vector ``\\boldsymbol{\\mu}`` and
covariance matrix ``\\boldsymbol{\\Sigma}`` is:

```math
f(\\mathbf{x}; \\boldsymbol{\\mu}, \\boldsymbol{\\Sigma}) = \\frac{1}{(2 \\pi)^{d/2} |\\boldsymbol{\\Sigma}|^{1/2}}
\\exp \\left( - \\frac{1}{2} (\\mathbf{x} - \\boldsymbol{\\mu})^T \\Sigma^{-1} (\\mathbf{x} - \\boldsymbol{\\mu}) \\right)
```

We realize that the mean vector and the covariance often have special forms in practice,
which can be exploited to simplify the computation. For example, the mean vector is sometimes
just a zero vector, while the covariance matrix can be a diagonal matrix or even in the form
of ``\\sigma^2 \\mathbf{I}``. To take advantage of such special cases, we introduce a parametric
type `MvNormal`, defined as below, which allows users to specify the special structure of
the mean and covariance.

```julia
struct MvNormal{Cov<:AbstractPDMat,Mean<:AbstractVector} <: AbstractMvNormal
    μ::Mean
    Σ::Cov
end
```

Here, the mean vector can be an instance of any `AbstractVector`. The covariance can be
of any subtype of `AbstractPDMat`. Particularly, one can use `PDMat` for full covariance,
`PDiagMat` for diagonal covariance, and `ScalMat` for the isotropic covariance -- those
in the form of ``\\sigma \\mathbf{I}``. (See the Julia package
[PDMats](https://github.com/lindahua/PDMats.jl) for details).

We also define a set of alias for the types using different combinations of mean vectors and covariance:

```julia
const IsoNormal  = MvNormal{ScalMat,  Vector{Float64}}
const DiagNormal = MvNormal{PDiagMat, Vector{Float64}}
const FullNormal = MvNormal{PDMat,    Vector{Float64}}

const ZeroMeanIsoNormal{Axes}  = MvNormal{ScalMat,  Zeros{Float64,1,Axes}}
const ZeroMeanDiagNormal{Axes} = MvNormal{PDiagMat, Zeros{Float64,1,Axes}}
const ZeroMeanFullNormal{Axes} = MvNormal{PDMat,    Zeros{Float64,1,Axes}}
```

Multivariate normal distributions support affine transformations:
```julia
d = MvNormal(μ, Σ)
c + B * d    # == MvNormal(B * μ + c, B * Σ * B')
dot(b, d)    # == Normal(dot(b, μ), b' * Σ * b)
```
"""
abstract type AbstractMvNormal <: ContinuousMultivariateDistribution end

### Generic methods (for all AbstractMvNormal subtypes)

insupport(d::AbstractMvNormal, x::AbstractVector) =
    length(d) == length(x) && all(isfinite, x)

mode(d::AbstractMvNormal) = mean(d)
modes(d::AbstractMvNormal) = [mean(d)]

"""
    rand(::AbstractRNG, ::Distributions.AbstractMvNormal)

Sample a random vector from the provided multi-variate normal distribution.
"""
rand(::AbstractRNG, ::Distributions.AbstractMvNormal)

function entropy(d::AbstractMvNormal)
    ldcd = logdetcov(d)
    T = typeof(ldcd)
    (length(d) * (T(log2π) + one(T)) + ldcd)/2
end

mvnormal_c0(g::AbstractMvNormal) = -(length(g) * Float64(log2π) + logdetcov(g))/2

"""
    invcov(d::AbstractMvNormal)

Return the inversed covariance matrix of d.
"""
invcov(d::AbstractMvNormal)

"""
    logdetcov(d::AbstractMvNormal)

Return the log-determinant value of the covariance matrix.
"""
logdetcov(d::AbstractMvNormal)

"""
    sqmahal(d, x)

Return the squared Mahalanobis distance from x to the center of d, w.r.t. the covariance.
When x is a vector, it returns a scalar value. When x is a matrix, it returns a vector of length size(x,2).

`sqmahal!(r, d, x)` with write the results to a pre-allocated array `r`.
"""
sqmahal(d::AbstractMvNormal, x::AbstractArray)

sqmahal(d::AbstractMvNormal, x::AbstractMatrix) = sqmahal!(Vector{promote_type(partype(d), eltype(x))}(undef, size(x, 2)), d, x)

_logpdf(d::AbstractMvNormal, x::AbstractVector) = mvnormal_c0(d) - sqmahal(d, x)/2

function _logpdf!(r::AbstractArray, d::AbstractMvNormal, x::AbstractMatrix)
    sqmahal!(r, d, x)
    c0 = mvnormal_c0(d)
    for i = 1:size(x, 2)
        @inbounds r[i] = c0 - r[i]/2
    end
    r
end

_pdf!(r::AbstractArray, d::AbstractMvNormal, x::AbstractMatrix) = exp!(_logpdf!(r, d, x))

###########################################################
#
#   MvNormal
#
#   Multivariate normal distribution with mean parameters
#
###########################################################
"""
    MvNormal

Generally, users don't have to worry about these internal details.
We provide a common constructor `MvNormal`, which will construct a distribution of
appropriate type depending on the input arguments.

    MvNormal(sig)

Construct a multivariate normal distribution with zero mean and covariance represented by `sig`.

    MvNormal(mu, sig)

Construct a multivariate normal distribution with mean `mu` and covariance represented by `sig`.

    MvNormal(d, sig)

Construct a multivariate normal distribution of dimension `d`, with zero mean, and an
isotropic covariance matrix corresponding `abs2(sig)*I`.

# Arguments
- `mu::Vector{T<:Real}`: The mean vector.
- `d::Real`: dimension of distribution.
- `sig`: The covariance, which can in of either of the following forms (with `T<:Real`):
    1. subtype of `AbstractPDMat`,
    2. symmetric matrix of type `Matrix{T}`,
    3. vector of type `Vector{T}`: indicating a diagonal covariance as `diagm(abs2(sig))`,
    4. real-valued number: indicating an isotropic covariance matrix corresponding `abs2(sig) * I`.

**Note:** The constructor will choose an appropriate covariance form internally, so that
special structure of the covariance can be exploited.
"""
struct MvNormal{T<:Real,Cov<:AbstractPDMat,Mean<:AbstractVector} <: AbstractMvNormal
    μ::Mean
    Σ::Cov
end

const MultivariateNormal = MvNormal  # for the purpose of backward compatibility

const IsoNormal  = MvNormal{Float64,ScalMat{Float64},Vector{Float64}}
const DiagNormal = MvNormal{Float64,PDiagMat{Float64,Vector{Float64}},Vector{Float64}}
const FullNormal = MvNormal{Float64,PDMat{Float64,Matrix{Float64}},Vector{Float64}}

const ZeroMeanIsoNormal{Axes}  = MvNormal{Float64,ScalMat{Float64},Zeros{Float64,1,Axes}}
const ZeroMeanDiagNormal{Axes} = MvNormal{Float64,PDiagMat{Float64,Vector{Float64}},Zeros{Float64,1,Axes}}
const ZeroMeanFullNormal{Axes} = MvNormal{Float64,PDMat{Float64,Matrix{Float64}},Zeros{Float64,1,Axes}}

### Construction
function MvNormal(μ::AbstractVector{T}, Σ::AbstractPDMat{T}) where {T<:Real}
    dim(Σ) == length(μ) || throw(DimensionMismatch("The dimensions of mu and Sigma are inconsistent."))
    MvNormal{T,typeof(Σ), typeof(μ)}(μ, Σ)
end

function MvNormal(μ::AbstractVector, Σ::AbstractPDMat)
    R = Base.promote_eltype(μ, Σ)
    MvNormal(convert(AbstractArray{R}, μ), convert(AbstractArray{R}, Σ))
end

# constructor with general covariance matrix
MvNormal(μ::AbstractVector{<:Real}, Σ::AbstractMatrix{<:Real}) = MvNormal(μ, PDMat(Σ, cholesky(Σ)))
MvNormal(μ::AbstractVector{<:Real}, Σ::Diagonal{<:Real}) = MvNormal(μ, PDiagMat(diag(Σ)))
MvNormal(μ::AbstractVector{<:Real}, Σ::UniformScaling{<:Real}) =
    MvNormal(μ, ScalMat(length(μ), Σ.λ))

# constructor with vector of standard deviations
MvNormal(μ::AbstractVector{<:Real}, σ::AbstractVector{<:Real}) = MvNormal(μ, PDiagMat(abs2.(σ)))

# constructor with scalar standard deviation
MvNormal(μ::AbstractVector{<:Real}, σ::Real) = MvNormal(μ, ScalMat(length(μ), abs2(σ)))

# constructor without mean vector
MvNormal(Σ::AbstractVecOrMat{<:Real}) = MvNormal(Zeros{eltype(Σ)}(size(Σ, 1)), Σ)

# special constructor
MvNormal(d::Int, σ::Real) = MvNormal(Zeros{typeof(σ)}(d), σ)

Base.eltype(::Type{<:MvNormal{T}}) where {T} = T

### Conversion
function convert(::Type{MvNormal{T}}, d::MvNormal) where T<:Real
    MvNormal(convert(AbstractArray{T}, d.μ), convert(AbstractArray{T}, d.Σ))
end
function convert(::Type{MvNormal{T}}, μ::AbstractVector, Σ::AbstractPDMat) where T<:Real
    MvNormal(convert(AbstractArray{T}, μ), convert(AbstractArray{T}, Σ))
end

### Show

distrname(d::IsoNormal)  = "IsoNormal"    # Note: IsoNormal, etc are just alias names.
distrname(d::DiagNormal) = "DiagNormal"
distrname(d::FullNormal) = "FullNormal"

distrname(d::ZeroMeanIsoNormal) = "ZeroMeanIsoNormal"
distrname(d::ZeroMeanDiagNormal) = "ZeroMeanDiagNormal"
distrname(d::ZeroMeanFullNormal) = "ZeroMeanFullNormal"

Base.show(io::IO, d::MvNormal) =
    show_multline(io, d, [(:dim, length(d)), (:μ, mean(d)), (:Σ, cov(d))])

### Basic statistics

length(d::MvNormal) = length(d.μ)
mean(d::MvNormal) = d.μ
params(d::MvNormal) = (d.μ, d.Σ)
@inline partype(d::MvNormal{T}) where {T<:Real} = T

var(d::MvNormal) = diag(d.Σ)
cov(d::MvNormal) = Matrix(d.Σ)

invcov(d::MvNormal) = Matrix(inv(d.Σ))
logdetcov(d::MvNormal) = logdet(d.Σ)

### Evaluation

sqmahal(d::MvNormal, x::AbstractVector) = invquad(d.Σ, x .- d.μ)

sqmahal!(r::AbstractVector, d::MvNormal, x::AbstractMatrix) =
    invquad!(r, d.Σ, x .- d.μ)

gradlogpdf(d::MvNormal, x::AbstractVector{<:Real}) = -(d.Σ \ (x .- d.μ))

# Sampling (for GenericMvNormal)

_rand!(rng::AbstractRNG, d::MvNormal, x::VecOrMat) =
    add!(unwhiten!(d.Σ, randn!(rng, x)), d.μ)

# Workaround: randn! only works for Array, but not generally for AbstractArray
function _rand!(rng::AbstractRNG, d::MvNormal, x::AbstractVector)
    for i in eachindex(x)
        @inbounds x[i] = randn(rng,eltype(d))
    end
    add!(unwhiten!(d.Σ, x), d.μ)
end

### Affine transformations

+(d::MvNormal, c::AbstractVector) = MvNormal(d.μ .+ c, d.Σ)

+(c::AbstractVector, d::MvNormal) = d + c

*(B::AbstractMatrix, d::MvNormal) = MvNormal(B * d.μ, X_A_Xt(d.Σ, B))

dot(b::AbstractVector, d::MvNormal) = Normal(dot(d.μ, b), √quad(d.Σ, b))

dot(d::MvNormal, b::AbstractVector) = dot(b, d)

###########################################################
#
#   Estimation of MvNormal
#
###########################################################

### Estimation with known covariance

struct MvNormalKnownCov{Cov<:AbstractPDMat}
    Σ::Cov
end

MvNormalKnownCov(d::Int, σ::Real) = MvNormalKnownCov(ScalMat(d, abs2(Float64(σ))))
MvNormalKnownCov(σ::Vector{Float64}) = MvNormalKnownCov(PDiagMat(abs2.(σ)))
MvNormalKnownCov(Σ::Matrix{Float64}) = MvNormalKnownCov(PDMat(Σ))

length(g::MvNormalKnownCov) = dim(g.Σ)

struct MvNormalKnownCovStats{Cov<:AbstractPDMat}
    invΣ::Cov              # inverse covariance
    sx::Vector{Float64}    # (weighted) sum of vectors
    tw::Float64            # sum of weights
end

function suffstats(g::MvNormalKnownCov{Cov}, x::AbstractMatrix{Float64}) where Cov<:AbstractPDMat
    size(x,1) == length(g) || throw(DimensionMismatch("Invalid argument dimensions."))
    invΣ = inv(g.Σ)
    sx = vec(sum(x, dims=2))
    tw = Float64(size(x, 2))
    MvNormalKnownCovStats{Cov}(invΣ, sx, tw)
end

function suffstats(g::MvNormalKnownCov{Cov}, x::AbstractMatrix{Float64}, w::AbstractVector) where Cov<:AbstractPDMat
    (size(x,1) == length(g) && size(x,2) == length(w)) ||
        throw(DimensionMismatch("Inconsistent argument dimensions."))
    invΣ = inv(g.Σ)
    sx = x * vec(w)
    tw = sum(w)
    MvNormalKnownCovStats{Cov}(invΣ, sx, tw)
end

## MLE estimation with covariance known

fit_mle(g::MvNormalKnownCov{C}, ss::MvNormalKnownCovStats{C}) where {C<:AbstractPDMat} =
    MvNormal(ss.sx * inv(ss.tw), g.Σ)

function fit_mle(g::MvNormalKnownCov, x::AbstractMatrix{Float64})
    d = length(g)
    size(x,1) == d || throw(DimensionMismatch("Invalid argument dimensions."))
    μ = multiply!(vec(sum(x,dims=2)), inv(size(x,2)))
    MvNormal(μ, g.Σ)
end

function fit_mle(g::MvNormalKnownCov, x::AbstractMatrix{Float64}, w::AbstractVector)
    d = length(g)
    (size(x,1) == d && size(x,2) == length(w)) ||
        throw(DimensionMismatch("Inconsistent argument dimensions."))
    μ = BLAS.gemv('N', inv(sum(w)), x, vec(w))
    MvNormal(μ, g.Σ)
end


### Estimation (both mean and cov unknown)

struct MvNormalStats <: SufficientStats
    s::Vector{Float64}  # (weighted) sum of x
    m::Vector{Float64}  # (weighted) mean of x
    s2::Matrix{Float64} # (weighted) sum of (x-μ) * (x-μ)'
    tw::Float64         # total sample weight
end

function suffstats(D::Type{MvNormal}, x::AbstractMatrix{Float64})
    d = size(x, 1)
    n = size(x, 2)
    s = vec(sum(x, dims=2))
    m = s * inv(n)
    z = x .- m
    s2 = z * z'
    MvNormalStats(s, m, s2, Float64(n))
end

function suffstats(D::Type{MvNormal}, x::AbstractMatrix{Float64}, w::AbstractVector)
    d = size(x, 1)
    n = size(x, 2)
    length(w) == n || throw(DimensionMismatch("Inconsistent argument dimensions."))

    tw = sum(w)
    s = x * vec(w)
    m = s * inv(tw)
    z = similar(x)
    for j = 1:n
        xj = view(x,:,j)
        zj = view(z,:,j)
        swj = sqrt(w[j])
        for i = 1:d
            @inbounds zj[i] = swj * (xj[i] - m[i])
        end
    end
    s2 = z * z'
    MvNormalStats(s, m, s2, tw)
end


# Maximum Likelihood Estimation
#
# Specialized algorithms are respectively implemented for
# each kind of covariance
#

fit_mle(D::Type{MvNormal}, ss::MvNormalStats) = fit_mle(FullNormal, ss)
fit_mle(D::Type{MvNormal}, x::AbstractMatrix{Float64}) = fit_mle(FullNormal, x)
fit_mle(D::Type{MvNormal}, x::AbstractMatrix{Float64}, w::AbstractArray{Float64}) = fit_mle(FullNormal, x, w)

fit_mle(D::Type{FullNormal}, ss::MvNormalStats) = MvNormal(ss.m, ss.s2 * inv(ss.tw))

function fit_mle(D::Type{FullNormal}, x::AbstractMatrix{Float64})
    n = size(x, 2)
    mu = vec(mean(x, dims=2))
    z = x .- mu
    C = BLAS.syrk('U', 'N', inv(n), z)
    LinearAlgebra.copytri!(C, 'U')
    MvNormal(mu, PDMat(C))
end

function fit_mle(D::Type{FullNormal}, x::AbstractMatrix{Float64}, w::AbstractVector)
    m = size(x, 1)
    n = size(x, 2)
    length(w) == n || throw(DimensionMismatch("Inconsistent argument dimensions"))

    inv_sw = inv(sum(w))
    mu = BLAS.gemv('N', inv_sw, x, w)

    z = Matrix{Float64}(undef, m, n)
    for j = 1:n
        cj = sqrt(w[j])
        for i = 1:m
            @inbounds z[i,j] = (x[i,j] - mu[i]) * cj
        end
    end
    C = BLAS.syrk('U', 'N', inv_sw, z)
    LinearAlgebra.copytri!(C, 'U')
    MvNormal(mu, PDMat(C))
end

function fit_mle(D::Type{DiagNormal}, x::AbstractMatrix{Float64})
    m = size(x, 1)
    n = size(x, 2)

    mu = vec(mean(x, dims=2))
    va = zeros(Float64, m)
    for j = 1:n
        for i = 1:m
            @inbounds va[i] += abs2(x[i,j] - mu[i])
        end
    end
    multiply!(va, inv(n))
    MvNormal(mu, PDiagMat(va))
end

function fit_mle(D::Type{DiagNormal}, x::AbstractMatrix{Float64}, w::AbstractVector)
    m = size(x, 1)
    n = size(x, 2)
    length(w) == n || throw(DimensionMismatch("Inconsistent argument dimensions"))

    inv_sw = inv(sum(w))
    mu = BLAS.gemv('N', inv_sw, x, w)

    va = zeros(Float64, m)
    for j = 1:n
        @inbounds wj = w[j]
        for i = 1:m
            @inbounds va[i] += abs2(x[i,j] - mu[i]) * wj
        end
    end
    multiply!(va, inv_sw)
    MvNormal(mu, PDiagMat(va))
end

function fit_mle(D::Type{IsoNormal}, x::AbstractMatrix{Float64})
    m = size(x, 1)
    n = size(x, 2)

    mu = vec(mean(x, dims=2))
    va = 0.
    for j = 1:n
        va_j = 0.
        for i = 1:m
            @inbounds va_j += abs2(x[i,j] - mu[i])
        end
        va += va_j
    end
    MvNormal(mu, ScalMat(m, va / (m * n)))
end

function fit_mle(D::Type{IsoNormal}, x::AbstractMatrix{Float64}, w::AbstractVector)
    m = size(x, 1)
    n = size(x, 2)
    length(w) == n || throw(DimensionMismatch("Inconsistent argument dimensions"))

    sw = sum(w)
    inv_sw = 1.0 / sw
    mu = BLAS.gemv('N', inv_sw, x, w)

    va = 0.
    for j = 1:n
        @inbounds wj = w[j]
        va_j = 0.
        for i = 1:m
            @inbounds va_j += abs2(x[i,j] - mu[i]) * wj
        end
        va += va_j
    end
    MvNormal(mu, ScalMat(m, va / (m * sw)))
end
