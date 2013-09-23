# Multivariate Normal distribution

## Generic multivariate normal class

immutable GenericMultivariateNormal{Cov<:AbstractPDMat} <: ContinuousMultivariateDistribution
    dim::Int
    zeromean::Bool
    μ::Vector{Float64}
    Σ::Cov
end

function GenericMultivariateNormal{Cov<:AbstractPDMat}(μ::Vector{Float64}, Σ::Cov, zmean::Bool)
    d = length(μ)
    dim(Σ) == d || throw(ArgumentError("The dimensions of μ and Σ are inconsistent."))
    GenericMultivariateNormal{Cov}(d, zmean, μ, Σ)
end

function GenericMultivariateNormal{Cov<:AbstractPDMat}(μ::Vector{Float64}, Σ::Cov)
    d = length(μ)
    dim(Σ) == d || throw(ArgumentError("The dimensions of μ and Σ are inconsistent."))
    zmean::Bool = true
    for i = 1:d
        if μ[i] != 0.
            zmean = false
            break
        end
    end
    GenericMultivariateNormal{Cov}(d, zmean, μ, Σ)
end

function GenericMultivariateNormal{Cov<:AbstractPDMat}(Σ::Cov)
    d = dim(Σ)
    GenericMultivariateNormal{Cov}(d, true, zeros(d), Σ)    
end

## Construction of multivariate normal with specific covariance type

typealias IsoNormal  GenericMultivariateNormal{ScalMat}
typealias DiagNormal GenericMultivariateNormal{PDiagMat}
typealias MultivariateNormal GenericMultivariateNormal{PDMat}

IsoNormal(μ::Vector{Float64}, C::ScalMat) = GenericMultivariateNormal(μ, C)
IsoNormal(C::ScalMat) = GenericMultivariateNormal(C)
IsoNormal(μ::Vector{Float64}, σ::Float64) = GenericMultivariateNormal(μ, ScalMat(length(μ), abs2(σ)))
IsoNormal(d::Int, σ::Float64) = GenericMultivariateNormal(ScalMat(d, abs2(σ)))

DiagNormal(μ::Vector{Float64}, C::PDiagMat) = GenericMultivariateNormal(μ, C)
DiagNormal(C::PDiagMat) = GenericMultivariateNormal(C)
DiagNormal(μ::Vector{Float64}, σ::Vector{Float64}) = GenericMultivariateNormal(μ, PDiagMat(abs2(σ)))

MultivariateNormal(μ::Vector{Float64}, C::PDMat) = GenericMultivariateNormal(μ, C)
MultivariateNormal(C::PDMat) = GenericMultivariateNormal(C)
MultivariateNormal(μ::Vector{Float64}, Σ::Matrix{Float64}) = GenericMultivariateNormal(μ, PDMat(Σ))
MultivariateNormal(Σ::Matrix{Float64}) = GenericMultivariateNormal(PDMat(Σ))

const MvNormal = MultivariateNormal

## convenient function to construct distributions of proper type based on arguments

gmvnormal(μ::Vector{Float64}, C::AbstractPDMat) = GenericMultivariateNormal(μ, C)
gmvnormal(C::AbstractPDMat) = GenericMultivariateNormal(C)

gmvnormal(μ::Vector{Float64}, σ::Float64) = IsoNormal(μ, σ)
gmvnormal(d::Int, σ::Float64) = IsoNormal(d, σ)
gmvnormal(μ::Vector{Float64}, σ::Vector{Float64}) = DiagNormal(μ, σ)
gmvnormal(μ::Vector{Float64}, Σ::Matrix{Float64}) = MultivariateNormal(μ, Σ)
gmvnormal(Σ::Matrix{Float64}) = MultivariateNormal(Σ)

## canonical forms

immutable GenericMvNormalCanon{Prec<:AbstractPDMat}
    h::Vector{Float64}   # potential vector, i.e. inv(Σ) * μ
    J::Prec              # precision matrix, i.e. inv(Σ)
end

GenericMvNormalCanon{Prec<:AbstractPDMat}(h::Vector{Float64}, J::Prec) = GenericMvNormalCanon{Prec}(h, J)

dim(cf::GenericMvNormalCanon) = length(cf.h)

function Base.convert{C<:AbstractPDMat}(D::Type{GenericMultivariateNormal{C}}, cf::GenericMvNormalCanon{C})
    Σ::C = inv(cf.J)
    μ = Σ * cf.h
    GenericMultivariateNormal(μ, Σ)
end

mean(cf::GenericMvNormalCanon) = cf.J \ cf.h
mode(cf::GenericMvNormalCanon) = cf.J \ cf.h

rand{C<:AbstractPDMat}(cf::GenericMvNormalCanon{C}) = rand(convert(GenericMultivariateNormal{C}, cf))

# Basic statistics

dim(d::GenericMultivariateNormal) = d.dim

mean(d::GenericMultivariateNormal) = d.μ

var(d::GenericMultivariateNormal) = diag(d.Σ)

cov(d::GenericMultivariateNormal) = full(d.Σ)

logdet_cov(d::GenericMultivariateNormal) = logdet(d.Σ)

mode(d::GenericMultivariateNormal) = d.μ

modes(d::GenericMultivariateNormal) = [mode(d)]

entropy(d::GenericMultivariateNormal) = 0.5 * (dim(d) * (float64(log2π) + 1.0) + logdet_cov(d))


# PDF evaluation

insupport{T<:Real}(d::GenericMultivariateNormal, x::Vector{T}) = dim(d) == length(x) && allfinite(x)
insupport{T<:Real}(d::GenericMultivariateNormal, x::Matrix{T}) = dim(d) == size(x,1) && allfinite(x)
insupport{G<:GenericMultivariateNormal,T<:Real}(::Type{G}, x::Vector{T}) = allfinite(x)
insupport{G<:GenericMultivariateNormal,T<:Real}(::Type{G}, x::Matrix{T}) = allfinite(x)


_mvnormal_c0(g::GenericMultivariateNormal) = -0.5 * (dim(g) * float64(log2π) + logdet_cov(g))

function sqmahal(d::GenericMultivariateNormal, x::Vector{Float64}) 
    z::Vector{Float64} = d.zeromean ? x : x - d.μ
    invquad(d.Σ, z) 
end

function sqmahal!(r::Array{Float64}, d::GenericMultivariateNormal, x::Matrix{Float64})
    if !(size(x, 1) == dim(d) && size(x, 2) == length(r))
        throw(ArgumentError("Inconsistent argument dimensions."))
    end
    z::Matrix{Float64} = d.zeromean ? x : bsubtract(x, d.μ, 1)
    invquad!(r, d.Σ, z)
end

sqmahal(d::GenericMultivariateNormal, x::Matrix{Float64}) = sqmahal!(Array(Float64, size(x, 2)), d, x)

logpdf(d::GenericMultivariateNormal, x::Vector{Float64}) = _mvnormal_c0(d) - 0.5 * sqmahal(d, x) 

function logpdf!(r::Array{Float64}, d::GenericMultivariateNormal, x::Matrix{Float64})
    sqmahal!(r, d, x)
    c0::Float64 = _mvnormal_c0(d)
    for i = 1:size(x, 2)
        r[i] = c0 - 0.5 * r[i]
    end 
    r
end


# Sampling

function rand!(d::GenericMultivariateNormal, x::Vector{Float64})
    unwhiten!(d.Σ, randn!(x))
    if !d.zeromean
        add!(x, d.μ)
    end
    x
end

function rand!(d::GenericMultivariateNormal, x::Matrix{Float64})
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
    GenericMultivariateNormal(ss.sx * inv(ss.tw), g.Σ)
end

function fit_mle(g::GenericMvNormalKnownSigma, x::Matrix{Float64})
    d = dim(g)
    size(x,1) == d || throw(ArgumentError("Invalid argument dimensions."))
    μ = multiply!(vec(sum(x,2)), 1.0 / size(x,2))
    GenericMultivariateNormal(μ, g.Σ)
end

function fit_mle(g::GenericMvNormalKnownSigma, x::Matrix{Float64}, w::Array{Float64})
    d = dim(g)
    if !(size(x,1) == d && size(x,2) == length(w))
        throw(ArgumentError("Inconsistent argument dimensions."))
    end
    μ = Base.LinAlg.BLAS.gemv('N', inv(sum(w)), x, vec(w))
    GenericMultivariateNormal(μ, g.Σ)
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

function suffstats(D::Type{MultivariateNormal}, x::Matrix{Float64})
    d = size(x, 1)
    n = size(x, 2)

    s = vec(sum(x,2))
    m = s * inv(n)
    z = bsubtract(x, m, 1)
    s2 = A_mul_Bt(z, z)

    MvNormalStats(s, m, s2, float64(n))
end

function suffstats(D::Type{MultivariateNormal}, x::Matrix{Float64}, w::Array{Float64})
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


fit_mle(D::Type{MultivariateNormal}, ss::MvNormalStats) = MvNormal(ss.m, ss.s2 * inv(ss.tw))

function fit_mle(D::Type{MultivariateNormal}, x::Matrix{Float64})
    n = size(x, 2)
    mu = vec(mean(x, 2))
    z = bsubtract(x, mu, 1)
    C = Base.LinAlg.BLAS.gemm('N', 'T', 1.0/n, z, z)   
    MvNormal(mu, PDMat(C)) 
end

function fit_mle(D::Type{MultivariateNormal}, x::Matrix{Float64}, w::Vector{Float64})
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

