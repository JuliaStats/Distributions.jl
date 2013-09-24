# Multivariate Normal distribution

## Generic multivariate normal class

abstract AbstractMvNormal <: ContinuousMultivariateDistribution

immutable GenericMvNormal{Cov<:AbstractPDMat} <: AbstractMvNormal
    dim::Int
    zeromean::Bool
    μ::Vector{Float64}
    Σ::Cov
end

function GenericMvNormal{Cov<:AbstractPDMat}(μ::Vector{Float64}, Σ::Cov, zmean::Bool)
    d = length(μ)
    dim(Σ) == d || throw(ArgumentError("The dimensions of μ and Σ are inconsistent."))
    GenericMvNormal{Cov}(d, zmean, μ, Σ)
end

function GenericMvNormal{Cov<:AbstractPDMat}(μ::Vector{Float64}, Σ::Cov)
    d = length(μ)
    dim(Σ) == d || throw(ArgumentError("The dimensions of μ and Σ are inconsistent."))
    GenericMvNormal{Cov}(d, allzeros(μ), μ, Σ)
end

function GenericMvNormal{Cov<:AbstractPDMat}(Σ::Cov)
    d = dim(Σ)
    GenericMvNormal{Cov}(d, true, zeros(d), Σ)    
end

## Construction of multivariate normal with specific covariance type

typealias IsoNormal  GenericMvNormal{ScalMat}
typealias DiagNormal GenericMvNormal{PDiagMat}
typealias MvNormal GenericMvNormal{PDMat}

MvNormal(μ::Vector{Float64}, C::PDMat) = GenericMvNormal(μ, C)
MvNormal(C::PDMat) = GenericMvNormal(C)
MvNormal(μ::Vector{Float64}, Σ::Matrix{Float64}) = GenericMvNormal(μ, PDMat(Σ))
MvNormal(Σ::Matrix{Float64}) = GenericMvNormal(PDMat(Σ))

DiagNormal(μ::Vector{Float64}, C::PDiagMat) = GenericMvNormal(μ, C)
DiagNormal(C::PDiagMat) = GenericMvNormal(C)
DiagNormal(μ::Vector{Float64}, σ::Vector{Float64}) = GenericMvNormal(μ, PDiagMat(abs2(σ)))

IsoNormal(μ::Vector{Float64}, C::ScalMat) = GenericMvNormal(μ, C)
IsoNormal(C::ScalMat) = GenericMvNormal(C)
IsoNormal(μ::Vector{Float64}, σ::Float64) = GenericMvNormal(μ, ScalMat(length(μ), abs2(σ)))
IsoNormal(d::Int, σ::Float64) = GenericMvNormal(ScalMat(d, abs2(σ)))

const MultivariateNormal = MvNormal  # for the purpose of backward compatibility

## convenient function to construct distributions of proper type based on arguments

gmvnormal(μ::Vector{Float64}, C::AbstractPDMat) = GenericMvNormal(μ, C)
gmvnormal(C::AbstractPDMat) = GenericMvNormal(C)

gmvnormal(μ::Vector{Float64}, σ::Float64) = IsoNormal(μ, σ)
gmvnormal(d::Int, σ::Float64) = IsoNormal(d, σ)
gmvnormal(μ::Vector{Float64}, σ::Vector{Float64}) = DiagNormal(μ, σ)
gmvnormal(μ::Vector{Float64}, Σ::Matrix{Float64}) = MvNormal(μ, Σ)
gmvnormal(Σ::Matrix{Float64}) = MvNormal(Σ)

# Basic statistics

dim(d::GenericMvNormal) = d.dim

mean(d::GenericMvNormal) = d.μ
mode(d::GenericMvNormal) = d.μ
modes(d::GenericMvNormal) = [mode(d)]

var(d::GenericMvNormal) = diag(d.Σ)
cov(d::GenericMvNormal) = full(d.Σ)
invcov(d::GenericMvNormal) = full(inv(d.Σ))
logdet_cov(d::GenericMvNormal) = logdet(d.Σ)

entropy(d::GenericMvNormal) = 0.5 * (dim(d) * (float64(log2π) + 1.0) + logdet_cov(d))


# evaluation (for GenericMvNormal)

function sqmahal(d::GenericMvNormal, x::Vector{Float64}) 
    z::Vector{Float64} = d.zeromean ? x : x - d.μ
    invquad(d.Σ, z) 
end

function sqmahal!(r::Array{Float64}, d::GenericMvNormal, x::Matrix{Float64})
    if !(size(x, 1) == dim(d) && size(x, 2) == length(r))
        throw(ArgumentError("Inconsistent argument dimensions."))
    end
    z::Matrix{Float64} = d.zeromean ? x : bsubtract(x, d.μ, 1)
    invquad!(r, d.Σ, z)
end


# generic PDF evaluation (appliable to AbstractMvNormal)

insupport{T<:Real}(d::AbstractMvNormal, x::Vector{T}) = dim(d) == length(x) && allfinite(x)
insupport{T<:Real}(d::AbstractMvNormal, x::Matrix{T}) = dim(d) == size(x,1) && allfinite(x)
insupport{G<:AbstractMvNormal,T<:Real}(::Type{G}, x::Vector{T}) = allfinite(x)
insupport{G<:AbstractMvNormal,T<:Real}(::Type{G}, x::Matrix{T}) = allfinite(x)

mvnormal_c0(g::AbstractMvNormal) = -0.5 * (dim(g) * float64(log2π) + logdet_cov(g))

sqmahal(d::AbstractMvNormal, x::Matrix{Float64}) = sqmahal!(Array(Float64, size(x, 2)), d, x)

logpdf(d::AbstractMvNormal, x::Vector{Float64}) = mvnormal_c0(d) - 0.5 * sqmahal(d, x) 

function logpdf!(r::Array{Float64}, d::AbstractMvNormal, x::Matrix{Float64})
    sqmahal!(r, d, x)
    c0::Float64 = mvnormal_c0(d)
    for i = 1:size(x, 2)
        r[i] = c0 - 0.5 * r[i]
    end 
    r
end


# Sampling (for GenericMvNormal)

function rand!(d::GenericMvNormal, x::Vector{Float64})
    unwhiten!(d.Σ, randn!(x))
    if !d.zeromean
        add!(x, d.μ)
    end
    x
end

function rand!(d::GenericMvNormal, x::Matrix{Float64})
    unwhiten!(d.Σ, randn!(x))
    if !d.zeromean
        badd!(x, d.μ, 1)
    end
    x
end


# Computation of sufficient statistics (with known covariance)
#

immutable GenericMvNormalKnownSigma{Cov<:AbstractPDMat}
    Σ::Cov
end

typealias MvNormalKnownSigma   GenericMvNormalKnownSigma{PDMat}
typealias DiagNormalKnownSigma GenericMvNormalKnownSigma{PDiagMat}
typealias IsoNormalKnownSigma  GenericMvNormalKnownSigma{ScalMat}

IsoNormalKnownSigma(σ::Float64) = IsoNormalKnownSigma(abs2(σ))
DiagNormalKnownSigma(σ::Vector{Float64}) = DiagNormalKnownSigma(abs2(σ))
MvNormalKnownSigma(C::Matrix{Float64}) = MvNormalKnownSigma(PDMat(C)) 

dim(g::GenericMvNormalKnownSigma) = dim(g.Σ)


immutable GenericMvNormalKnownSigmaStats{Cov<:AbstractPDMat}
    invΣ::Cov              # inverse covariance
    sx::Vector{Float64}    # (weighted) sum of vectors 
    tw::Float64            # sum of weights
end

typealias MvNormalKnownSigmaStats   GenericMvNormalKnownSigmaStats{PDMat}
typealias DiagNormalKnownSigmaStats GenericMvNormalKnownSigmaStats{PDiagMat}
typealias IsoNormalKnownSigmaStats  GenericMvNormalKnownSigmaStats{ScalMat}

function suffstats{Cov<:AbstractPDMat}(g::GenericMvNormalKnownSigma{Cov}, x::Matrix{Float64})
    size(x,1) == dim(g) || throw(ArgumentError("Invalid argument dimensions."))
    invΣ = inv(g.Σ)
    sx = vec(sum(x, 2))
    tw = float64(size(x, 2))
    GenericMvNormalKnownSigmaStats{Cov}(invΣ, sx, tw)
end

function suffstats{Cov<:AbstractPDMat}(g::GenericMvNormalKnownSigma{Cov}, x::Matrix{Float64}, w::Array{Float64})
    if !(size(x,1) == dim(g) && size(x,2) == length(w))
        throw(ArgumentError("Inconsistent argument dimensions."))
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
    size(x,1) == d || throw(ArgumentError("Invalid argument dimensions."))
    μ = multiply!(vec(sum(x,2)), 1.0 / size(x,2))
    GenericMvNormal(μ, g.Σ)
end

function fit_mle(g::GenericMvNormalKnownSigma, x::Matrix{Float64}, w::Array{Float64})
    d = dim(g)
    if !(size(x,1) == d && size(x,2) == length(w))
        throw(ArgumentError("Inconsistent argument dimensions."))
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
    length(w) == n || throw(ArgumentError("Inconsistent argument dimensions."))

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
        throw(ArgumentError("Inconsistent argument dimensions"))
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
        throw(ArgumentError("Inconsistent argument dimensions"))
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
        throw(ArgumentError("Inconsistent argument dimensions"))
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

