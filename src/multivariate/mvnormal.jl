# Multivariate Normal distribution

## Generic multivariate normal class

abstract AbstractMvNormal{Cov<:AbstractPDMat} <: ContinuousMultivariateDistribution

immutable GenericMvNormal{Mean<:RealVector,Cov<:AbstractPDMat} <: AbstractMvNormal{Cov}
    μ::Mean
    Σ::Cov
    function GenericMvNormal(μ, Σ)
        dim(Σ) == length(μ) || throw(DimensionMismatch("The dimensions of μ and Σ are inconsistent."))
        new(μ, Σ)
    end
end
GenericMvNormal(μ::RealVector,Σ::AbstractPDMat) = GenericMvNormal{typeof(μ),typeof(Σ)}(μ, Σ)

immutable ZeroMeanMvNormal{Cov<:AbstractPDMat} <: AbstractMvNormal
    Σ::Cov
end


## Construction of multivariate normal with specific covariance type
typealias IsoNormal  AbstractMvNormal{ScalMat}
typealias DiagNormal AbstractMvNormal{PDiagMat}
typealias MvNormal AbstractMvNormal{PDMat}

mvnormal(μ::RealVector, C::PDMat) = GenericMvNormal(μ, C)
mvnormal(μ::RealVector, Σ::RealMatrix) = GenericMvNormal(μ, PDMat(Σ))
mvnormal(C::PDMat) = ZeroMeanMvNormal(C)
mvnormal(Σ::RealMatrix) = ZeroMeanMvNormal(PDMat(Σ))

diagnormal(μ::RealVector, C::PDiagMat) = GenericMvNormal(μ, C)
diagnormal(μ::RealVector, σ::RealVector) = GenericMvNormal(μ, PDiagMat(abs2(σ)))
diagnormal(C::PDiagMat) = ZeroMeanMvNormal(C)
diagnormal(σ::RealVector) = ZeroMeanMvNormal(PDiagMat(abs2(σ)))

isonormal(μ::RealVector, C::ScalMat) = GenericMvNormal(μ, C)
isonormal(μ::RealVector, σ::Real) = GenericMvNormal(μ, ScalMat(length(μ), abs2(float64(σ))))
isonormal(C::ScalMat) = ZeroMeanMvNormal(C)
isonormal(d::Int, σ::Real) = ZeroMeanMvNormal(ScalMat(d, abs2(float64(σ))))


## Support
insupport{T<:Real}(d::AbstractMvNormal, x::Vector{T}) = length(d) == length(x) && allfinite(x)
insupport{G<:AbstractMvNormal,T<:Real}(::Type{G}, x::Vector{T}) = allfinite(x)

## Properties
length(d::AbstractMvNormal) = dim(d.Σ)

mean(d::GenericMvNormal) = d.μ
mean(d::ZeroMeanMvNormal) = zeros(length(d))
mode(d::AbstractMvNormal) = mean(d)

var(d::AbstractMvNormal) = diag(d.Σ)
cov(d::AbstractMvNormal) = full(d.Σ)
invcov(d::AbstractMvNormal) = full(inv(d.Σ))
logdet_cov(d::AbstractMvNormal) = logdet(d.Σ)

entropy(d::AbstractMvNormal) = 0.5 * (length(d) * (log2π + 1.0) + logdet_cov(d))


## Functions
# sqmahal = (x-μ)' * inv(Σ) * (x-μ)
sqmahal(d::GenericMvNormal, x::RealVector) = invquad(d.Σ, x - d.μ)
sqmahal(d::ZeroMeanMvNormal, x::RealVector) = invquad(d.Σ, x)

# TODO: handle arbitrary dimensioned arrays
function sqmahal!(r::RealVector, d::GenericMvNormal, x::RealMatrix)
    (size(x, 1) == length(d) && size(x, 2) == length(r)) || throw(DimensionMismatch("Inconsistent argument dimensions"))
    z::Matrix{Float64} = bsubtract(x, d.μ, 1)
    invquad!(r, d.Σ, z)
end
function sqmahal!(r::RealVector, d::ZeroMeanMvNormal, x::RealMatrix)
    (size(x, 1) == length(d) && size(x, 2) == length(r)) || throw(DimensionMismatch("Inconsistent argument dimensions"))
    invquad!(r, d.Σ, x)
end
sqmahal(d::AbstractMvNormal, x::RealMatrix) = sqmahal!(Array(Float64, size(x,2)), d, x)

# log-normalisation constant
mvnormal_c0(g::AbstractMvNormal) = -0.5 * (dim(g) * float64(log2π) + logdet_cov(g))

logpdf(d::AbstractMvNormal, x::RealVector) = mvnormal_c0(d) - 0.5 * sqmahal(d, x) 
function logpdf!(r::AbstractArray{Float64}, d::AbstractMvNormal, x::AbstractMatrix{Float64})
    sqmahal!(r, d, x)
    c0::Float64 = mvnormal_c0(d)
    for i = 1:size(x, 2)
        r[i] = c0 - 0.5 * r[i]
    end 
    r
end

gradlogpdf(d::GenericMvNormal, x::RealVector) = -d.Σ \ (x - d.μ)
gradlogpdf(d::ZeroMeanMvNormal, x::RealVector) = -d.Σ \ x


## Sampling
rand!(d::GenericMvNormal, x::RealVector) = add!(unwhiten!(d.Σ, randn!(x)),d.μ)
rand!(d::ZeroMeanMvNormal, x::RealVector) = unwhiten!(d.Σ, randn!(x))

# TODO: this could be made to work for arrays.
rand!(d::GenericMvNormal, x::RealMatrix) = badd!(unwhiten!(d.Σ, randn!(x)),  d.μ, 1)
rand!(d::ZeroMeanMvNormal, x::RealMatrix) = unwhiten!(d.Σ, randn!(x))


# Computation of sufficient statistics (with known covariance)

immutable GenericMvNormalKnownSigma{Cov<:AbstractPDMat}
    Σ::Cov
end

typealias MvNormalKnownSigma   GenericMvNormalKnownSigma{PDMat}
typealias DiagNormalKnownSigma GenericMvNormalKnownSigma{PDiagMat}
typealias IsoNormalKnownSigma  GenericMvNormalKnownSigma{ScalMat}

IsoNormalKnownSigma(dimension::Int, σ::Float64) =
    IsoNormalKnownSigma(ScalMat(dimension, abs2(σ)))
DiagNormalKnownSigma(σ::Vector{Float64}) = DiagNormalKnownSigma(abs2(σ))
MvNormalKnownSigma(C::Matrix{Float64}) = MvNormalKnownSigma(PDMat(C)) 

length(g::GenericMvNormalKnownSigma) = dim(g.Σ)


immutable GenericMvNormalKnownSigmaStats{Cov<:AbstractPDMat}
    invΣ::Cov              # inverse covariance
    sx::Vector{Float64}    # (weighted) sum of vectors 
    tw::Float64            # sum of weights
end

typealias MvNormalKnownSigmaStats   GenericMvNormalKnownSigmaStats{PDMat}
typealias DiagNormalKnownSigmaStats GenericMvNormalKnownSigmaStats{PDiagMat}
typealias IsoNormalKnownSigmaStats  GenericMvNormalKnownSigmaStats{ScalMat}

function suffstats{Cov<:AbstractPDMat}(g::GenericMvNormalKnownSigma{Cov}, x::Matrix{Float64})
    size(x,1) == dim(g) || throw(DimensionMismatch("Inconsistent argument dimensions"))
    invΣ = inv(g.Σ)
    sx = vec(sum(x, 2))
    tw = float64(size(x, 2))
    GenericMvNormalKnownSigmaStats{Cov}(invΣ, sx, tw)
end

function suffstats{Cov<:AbstractPDMat}(g::GenericMvNormalKnownSigma{Cov}, x::Matrix{Float64}, w::Array{Float64})
    if !(size(x,1) == dim(g) && size(x,2) == length(w))
        throw(DimensionMismatch("Inconsistent argument dimensions"))
    end
    invΣ = inv(g.Σ)
    sx = x * vec(w)
    tw = sum(w)
    GenericMvNormalKnownSigmaStats{Cov}(invΣ, sx, tw)
end


## MLE estimation with covariance known

function fit_mle{C<:AbstractPDMat}(g::GenericMvNormalKnownSigma{C}, ss::GenericMvNormalKnownSigmaStats{C})
    GenericMvNormal(ss.sx * inv(ss.tw), g.Σ)
end

function fit_mle(g::GenericMvNormalKnownSigma, x::Matrix{Float64})
    d = dim(g)
    size(x,1) == d || throw(DimensionMismatch("Inconsistent argument dimensions"))
    μ = multiply!(vec(sum(x,2)), 1.0 / size(x,2))
    GenericMvNormal(μ, g.Σ)
end

function fit_mle(g::GenericMvNormalKnownSigma, x::Matrix{Float64}, w::Array{Float64})
    d = dim(g)
    if !(size(x,1) == d && size(x,2) == length(w))
        throw(DimensionMismatch("Inconsistent argument dimensions"))
    end
    μ = Base.LinAlg.BLAS.gemv('N', inv(sum(w)), x, vec(w))
    GenericMvNormal(μ, g.Σ)
end


# Computation of sufficient statistics (both μ and Σ are unknown)
#
# TODO: implement methods for DiagNormal and IsoNormal
#

immutable MvNormalStats <: SufficientStats
    s::Vector{Float64}  # (weighted) sum of x
    m::Vector{Float64}  # (weighted) mean of x
    s2::Matrix{Float64} # (weighted) sum of (x-μ) * (x-μ)'
    tw::Float64         # total sample weight
end

function suffstats(D::Type{MvNormal}, x::Matrix{Float64})
    d = size(x, 1)
    n = size(x, 2)

    s = vec(sum(x,2))
    m = s * inv(n)
    z = bsubtract(x, m, 1)
    s2 = A_mul_Bt(z, z)

    MvNormalStats(s, m, s2, float64(n))
end

function suffstats(D::Type{MvNormal}, x::Matrix{Float64}, w::Array{Float64})
    d = size(x, 1)
    n = size(x, 2)
    length(w) == n || throw(DimensionMismatch("Inconsistent argument dimensions"))

    tw = sum(w)
    s = x * vec(w)
    m = s * inv(tw)
    z = bmultiply!(bsubtract(x, m, 1), sqrt(w), 2)
    s2 = A_mul_Bt(z, z)

    MvNormalStats(s, m, s2, tw)
end


# Maximum Likelihood Estimation
#
# Specialized algorithms are respectively implemented for 
# each kind of covariance
#


fit_mle(D::Type{MvNormal}, ss::MvNormalStats) = MvNormal(ss.m, ss.s2 * inv(ss.tw))

function fit_mle(D::Type{MvNormal}, x::Matrix{Float64})
    n = size(x, 2)
    mu = vec(mean(x, 2))
    z = bsubtract(x, mu, 1)
    C = Base.LinAlg.BLAS.gemm('N', 'T', 1.0/n, z, z)   
    MvNormal(mu, PDMat(C)) 
end

function fit_mle(D::Type{MvNormal}, x::Matrix{Float64}, w::Vector{Float64})
    m = size(x, 1)
    n = size(x, 2)
    if length(w) != n
        throw(DimensionMismatch("Inconsistent argument dimensions"))
    end

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

function fit_mle(D::Type{DiagNormal}, x::Matrix{Float64})
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

    DiagNormal(mu, PDiagMat(va))
end

function fit_mle(D::Type{DiagNormal}, x::Matrix{Float64}, w::Vector{Float64})
    m = size(x, 1)
    n = size(x, 2)    
    if length(w) != n
        throw(DimensionMismatch("Inconsistent argument dimensions"))
    end

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

    DiagNormal(mu, PDiagMat(va))
end

function fit_mle(D::Type{IsoNormal}, x::Matrix{Float64})
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

    IsoNormal(mu, ScalMat(m, va / (m * n)))
end

function fit_mle(D::Type{IsoNormal}, x::Matrix{Float64}, w::Vector{Float64})
    m = size(x, 1)
    n = size(x, 2)    
    if length(w) != n
        throw(DimensionMismatch("Inconsistent argument dimensions"))
    end

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

    IsoNormal(mu, ScalMat(m, va / (m * sw)))
end

