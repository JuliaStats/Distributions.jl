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

abstract AbstractMvNormal <: ContinuousMultivariateDistribution

### Generic methods (for all AbstractMvNormal subtypes)

insupport{T<:Real}(d::AbstractMvNormal, x::AbstractVector{T}) =
    length(d) == length(x) && allfinite(x)

mode(d::AbstractMvNormal) = mean(d)
modes(d::AbstractMvNormal) = [mean(d)]

@compat entropy(d::AbstractMvNormal) = 0.5 * (length(d) * (Float64(log2π) + 1.0) + logdetcov(d))

@compat mvnormal_c0(g::AbstractMvNormal) = -0.5 * (length(g) * Float64(log2π) + logdetcov(g))

sqmahal{T<:Real}(d::AbstractMvNormal, x::DenseMatrix{T}) = sqmahal!(Array(Float64, size(x, 2)), d, x)

_logpdf{T<:Real}(d::AbstractMvNormal, x::DenseVector{T}) = mvnormal_c0(d) - 0.5 * sqmahal(d, x)

function _logpdf!{T<:Real}(r::DenseArray, d::AbstractMvNormal, x::AbstractMatrix{T})
    sqmahal!(r, d, x)
    c0::Float64 = mvnormal_c0(d)
    for i = 1:size(x, 2)
        @inbounds r[i] = c0 - 0.5 * r[i]
    end
    r
end

_pdf!{T<:Real}(r::DenseArray, d::AbstractMvNormal, x::AbstractMatrix{T}) = exp!(_logpdf!(r, d, x))


###########################################################
#
#   MvNormal
#
#   Multivariate normal distribution with mean parameters
#
###########################################################

@compat immutable MvNormal{Cov<:AbstractPDMat,Mean<:Union{Vector{Float64},ZeroVector{Float64}}} <: AbstractMvNormal
    μ::Mean
    Σ::Cov
end

const MultivariateNormal = MvNormal  # for the purpose of backward compatibility

typealias IsoNormal  MvNormal{ScalMat,Vector{Float64}}
typealias DiagNormal MvNormal{PDiagMat,Vector{Float64}}
typealias FullNormal MvNormal{PDMat,Vector{Float64}}

typealias ZeroMeanIsoNormal  MvNormal{ScalMat,ZeroVector{Float64}}
typealias ZeroMeanDiagNormal MvNormal{PDiagMat,ZeroVector{Float64}}
typealias ZeroMeanFullNormal MvNormal{PDMat,ZeroVector{Float64}}

### Construction

function MvNormal{Cov<:AbstractPDMat}(μ::Vector{Float64}, Σ::Cov)
    dim(Σ) == length(μ) || throw(DimensionMismatch("The dimensions of mu and Sigma are inconsistent."))
    MvNormal{Cov,Vector{Float64}}(μ, Σ)
end

MvNormal{Cov<:AbstractPDMat}(Σ::Cov) = MvNormal{Cov,ZeroVector{Float64}}(ZeroVector(Float64, dim(Σ)), Σ)

MvNormal(μ::Vector{Float64}, Σ::Matrix{Float64}) = MvNormal(μ, PDMat(Σ))
MvNormal(μ::Vector{Float64}, σ::Vector{Float64}) = MvNormal(μ, PDiagMat(abs2(σ)))
@compat MvNormal(μ::Vector{Float64}, σ::Real) = MvNormal(μ, ScalMat(length(μ), abs2(Float64(σ))))

MvNormal(Σ::Matrix{Float64}) = MvNormal(PDMat(Σ))
MvNormal(σ::Vector{Float64}) = MvNormal(PDiagMat(abs2(σ)))
@compat MvNormal(d::Int, σ::Real) = MvNormal(ScalMat(d, abs2(Float64(σ))))

### Show

distrname(d::IsoNormal) = "IsoNormal"    # Note: IsoNormal, etc are just alias names
distrname(d::DiagNormal) = "DiagNormal"
distrname(d::FullNormal) = "FullNormal"

distrname(d::ZeroMeanIsoNormal) = "ZeroMeanIsoNormal"
distrname(d::ZeroMeanDiagNormal) = "ZeroMeanDiagNormal"
distrname(d::ZeroMeanFullNormal) = "ZeroMeanFullNormal"

Base.show(io::IO, d::MvNormal) =
    show_multline(io, d, [(:dim, length(d)), (:μ, mean(d)), (:Σ, cov(d))])

### Basic statistics

length(d::MvNormal) = length(d.μ)
mean(d::MvNormal) = convert(Vector{Float64}, d.μ)

var(d::MvNormal) = diag(d.Σ)
cov(d::MvNormal) = full(d.Σ)
invcov(d::MvNormal) = full(inv(d.Σ))
logdetcov(d::MvNormal) = logdet(d.Σ)

### Evaluation

sqmahal{T<:Real}(d::MvNormal, x::DenseVector{T}) = invquad(d.Σ, x - d.μ)

sqmahal!{T<:Real}(r::DenseVector, d::MvNormal, x::DenseMatrix{T}) =
    invquad!(r, d.Σ, x .- d.μ)

gradlogpdf(d::MvNormal, x::Vector{Float64}) = -(d.Σ \ (x - d.μ))

# Sampling (for GenericMvNormal)

_rand!(d::MvNormal, x::VecOrMat{Float64}) = add!(unwhiten!(d.Σ, randn!(x)), d.μ)

# Workaround: randn! only works for Array, but not generally for DenseArray
function _rand!(d::MvNormal, x::DenseVecOrMat{Float64})
    for i = 1:length(x)
        @inbounds x[i] = randn()
    end
    add!(unwhiten!(d.Σ, x), d.μ)
end

###########################################################
#
#   Estimation of MvNormal
#
###########################################################

### Estimation with known covariance

immutable MvNormalKnownCov{Cov<:AbstractPDMat}
    Σ::Cov
end

MvNormalKnownCov{Cov<:AbstractPDMat}(C::Cov) = MvNormalKnownCov{Cov}(C)
@compat MvNormalKnownCov(d::Int, σ::Real) = MvNormalKnownCov(ScalMat(d, abs2(Float64(σ))))
MvNormalKnownCov(σ::Vector{Float64}) = MvNormalKnownCov(PDiagMat(abs2(σ)))
MvNormalKnownCov(Σ::Matrix{Float64}) = MvNormalKnownCov(PDMat(Σ))

length(g::MvNormalKnownCov) = dim(g.Σ)

immutable MvNormalKnownCovStats{Cov<:AbstractPDMat}
    invΣ::Cov              # inverse covariance
    sx::Vector{Float64}    # (weighted) sum of vectors
    tw::Float64            # sum of weights
end

function suffstats{Cov<:AbstractPDMat}(g::MvNormalKnownCov{Cov}, x::AbstractMatrix{Float64})
    size(x,1) == length(g) || throw(DimensionMismatch("Invalid argument dimensions."))
    invΣ = inv(g.Σ)
    sx = vec(sum(x, 2))
    @compat tw = Float64(size(x, 2))
    MvNormalKnownCovStats{Cov}(invΣ, sx, tw)
end

function suffstats{Cov<:AbstractPDMat}(g::MvNormalKnownCov{Cov}, x::AbstractMatrix{Float64}, w::AbstractArray{Float64})
    (size(x,1) == length(g) && size(x,2) == length(w)) ||
        throw(DimensionMismatch("Inconsistent argument dimensions."))
    invΣ = inv(g.Σ)
    sx = x * vec(w)
    tw = sum(w)
    MvNormalKnownCovStats{Cov}(invΣ, sx, tw)
end

## MLE estimation with covariance known

fit_mle{C<:AbstractPDMat}(g::MvNormalKnownCov{C}, ss::MvNormalKnownCovStats{C}) =
    MvNormal(ss.sx * inv(ss.tw), g.Σ)

function fit_mle(g::MvNormalKnownCov, x::AbstractMatrix{Float64})
    d = length(g)
    size(x,1) == d || throw(DimensionMismatch("Invalid argument dimensions."))
    μ = multiply!(vec(sum(x,2)), 1.0 / size(x,2))
    MvNormal(μ, g.Σ)
end

function fit_mle(g::MvNormalKnownCov, x::AbstractMatrix{Float64}, w::AbstractArray{Float64})
    d = length(g)
    (size(x,1) == d && size(x,2) == length(w)) ||
        throw(DimensionMismatch("Inconsistent argument dimensions."))
    μ = Base.LinAlg.BLAS.gemv('N', inv(sum(w)), x, vec(w))
    MvNormal(μ, g.Σ)
end


### Estimation (both mean and cov unknown)

immutable MvNormalStats <: SufficientStats
    s::Vector{Float64}  # (weighted) sum of x
    m::Vector{Float64}  # (weighted) mean of x
    s2::Matrix{Float64} # (weighted) sum of (x-μ) * (x-μ)'
    tw::Float64         # total sample weight
end

function suffstats(D::Type{MvNormal}, x::AbstractMatrix{Float64})
    d = size(x, 1)
    n = size(x, 2)
    s = vec(sum(x,2))
    m = s * inv(n)
    z = x .- m
    s2 = A_mul_Bt(z, z)
    @compat MvNormalStats(s, m, s2, Float64(n))
end

function suffstats(D::Type{MvNormal}, x::AbstractMatrix{Float64}, w::Array{Float64})
    d = size(x, 1)
    n = size(x, 2)
    length(w) == n || throw(ArgumentError("Inconsistent argument dimensions."))

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
    s2 = A_mul_Bt(z, z)
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
    mu = vec(mean(x, 2))
    z = x .- mu
    C = Base.LinAlg.BLAS.gemm('N', 'T', 1.0/n, z, z)
    MvNormal(mu, PDMat(C))
end

function fit_mle(D::Type{FullNormal}, x::AbstractMatrix{Float64}, w::AbstractVector{Float64})
    m = size(x, 1)
    n = size(x, 2)
    length(w) == n || throw(DimensionMismatch("Inconsistent argument dimensions"))

    inv_sw = 1.0 / sum(w)
    mu = Base.LinAlg.BLAS.gemv('N', inv_sw, x, w)

    z = Array(Float64, m, n)
    for j = 1:n
        cj = sqrt(w[j])
        for i = 1:m
            @inbounds z[i,j] = (x[i,j] - mu[i]) * cj
        end
    end
    C = Base.LinAlg.BLAS.gemm('N', 'T', inv_sw, z, z)
    MvNormal(mu, PDMat(C))
end

function fit_mle(D::Type{DiagNormal}, x::AbstractMatrix{Float64})
    m = size(x, 1)
    n = size(x, 2)

    mu = vec(mean(x, 2))
    va = zeros(Float64, m)
    for j = 1:n
        for i = 1:m
            @inbounds va[i] += abs2(x[i,j] - mu[i])
        end
    end
    multiply!(va, inv(n))
    MvNormal(mu, PDiagMat(va))
end

function fit_mle(D::Type{DiagNormal}, x::AbstractMatrix{Float64}, w::AbstractArray{Float64})
    m = size(x, 1)
    n = size(x, 2)
    length(w) == n || throw(DimensionMismatch("Inconsistent argument dimensions"))

    inv_sw = 1.0 / sum(w)
    mu = Base.LinAlg.BLAS.gemv('N', inv_sw, x, w)

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

    mu = vec(mean(x, 2))
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

function fit_mle(D::Type{IsoNormal}, x::AbstractMatrix{Float64}, w::AbstractArray{Float64})
    m = size(x, 1)
    n = size(x, 2)
    length(w) == n || throw(DimensionMismatch("Inconsistent argument dimensions"))

    sw = sum(w)
    inv_sw = 1.0 / sw
    mu = Base.LinAlg.BLAS.gemv('N', inv_sw, x, w)

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
