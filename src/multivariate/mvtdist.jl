# Multivariate t-distribution

## Generic multivariate t-distribution class

abstract AbstractMvTDist <: ContinuousMultivariateDistribution

immutable GenericMvTDist{Cov<:AbstractPDMat} <: AbstractMvTDist
    df::Float64 # non-integer degrees of freedom allowed
    dim::Int
    zeromean::Bool
    μ::Vector{Float64}
    Σ::Cov

    function GenericMvTDist{Cov}(df::Float64, dim::Int, zmean::Bool, μ::Vector{Float64}, Σ::Cov)
      df > zero(df) || error("df must be positive")
      new(float64(df), dim, zmean, μ, Σ)
    end
end

function GenericMvTDist{Cov<:AbstractPDMat}(df::Float64, μ::Vector{Float64}, Σ::Cov, zmean::Bool)
    d = length(μ)
    dim(Σ) == d || throw(ArgumentError("The dimensions of μ and Σ are inconsistent."))
    GenericMvTDist{Cov}(df, d, zmean, μ, Σ)
end

function GenericMvTDist{Cov<:AbstractPDMat}(df::Float64, μ::Vector{Float64}, Σ::Cov)
    d = length(μ)
    dim(Σ) == d || throw(ArgumentError("The dimensions of μ and Σ are inconsistent."))
    GenericMvTDist{Cov}(df, d, allzeros(μ), μ, Σ)
end

function GenericMvTDist{Cov<:AbstractPDMat}(df::Float64, Σ::Cov)
    d = dim(Σ)
    GenericMvTDist{Cov}(df, d, true, zeros(d), Σ)    
end

## Construction of multivariate normal with specific covariance type

typealias IsoTDist  GenericMvTDist{ScalMat}
typealias DiagTDist GenericMvTDist{PDiagMat}
typealias MvTDist GenericMvTDist{PDMat}

MvTDist(df::Float64, μ::Vector{Float64}, C::PDMat) = GenericMvTDist(df, μ, C)
MvTDist(df::Float64, C::PDMat) = GenericMvTDist(df, C)
MvTDist(df::Float64, μ::Vector{Float64}, Σ::Matrix{Float64}) = GenericMvTDist(df, μ, PDMat(Σ))
MvTDist(df::Float64, Σ::Matrix{Float64}) = GenericMvTDist(df, PDMat(Σ))

DiagTDist(df::Float64, μ::Vector{Float64}, C::PDiagMat) = GenericMvTDist(df, μ, C)
DiagTDist(df::Float64, C::PDiagMat) = GenericMvTDist(df, C)
DiagTDist(df::Float64, μ::Vector{Float64}, σ::Vector{Float64}) = GenericMvTDist(df, μ, PDiagMat(abs2(σ)))

IsoTDist(df::Float64, μ::Vector{Float64}, C::ScalMat) = GenericMvTDist(df, μ, C)
IsoTDist(df::Float64, C::ScalMat) = GenericMvTDist(df, C)
IsoTDist(df::Float64, μ::Vector{Float64}, σ::Real) = GenericMvTDist(df, μ, ScalMat(length(μ), abs2(float64(σ))))
IsoTDist(df::Float64, d::Int, σ::Real) = GenericMvTDist(df, ScalMat(d, abs2(float64(σ))))

## convenient function to construct distributions of proper type based on arguments

mvtdist(df::Float64, μ::Vector{Float64}, C::AbstractPDMat) = GenericMvTDist(df, μ, C)
mvtdist(df::Float64, C::AbstractPDMat) = GenericMvTDist(df, C)

mvtdist(df::Float64, μ::Vector{Float64}, σ::Real) = IsoTDist(df, μ, float64(σ))
mvtdist(df::Float64, d::Int, σ::Float64) = IsoTDist(d, σ)
mvtdist(df::Float64, μ::Vector{Float64}, σ::Vector{Float64}) = DiagTDist(df, μ, σ)
mvtdist(df::Float64, μ::Vector{Float64}, Σ::Matrix{Float64}) = MvTDist(df, μ, Σ)
mvtdist(df::Float64, Σ::Matrix{Float64}) = MvTDist(df, Σ)

# Basic statistics

length(d::GenericMvTDist) = d.dim

mean(d::GenericMvTDist) = d.df>1 ? d.μ : NaN
mode(d::GenericMvTDist) = d.μ
modes(d::GenericMvTDist) = [mode(d)]

var(d::GenericMvTDist) = d.df>2 ? (d.df/(d.df-2))*diag(d.Σ) : Float64[NaN for i = 1:d.dim]
scale(d::GenericMvTDist) = full(d.Σ)
cov(d::GenericMvTDist) = d.df>2 ? (d.df/(d.df-2))*full(d.Σ) : NaN*ones(d.dim, d.dim)
invscale(d::GenericMvTDist) = full(inv(d.Σ))
invcov(d::GenericMvTDist) = d.df>2 ? ((d.df-2)/d.df)*full(inv(d.Σ)) : NaN*ones(d.dim, d.dim)
logdet_cov(d::GenericMvTDist) = d.df>2 ? logdet((d.df/(d.df-2))*d.Σ) : NaN

# For entropy calculations see "Multivariate t Distributions and their Applications", S. Kotz & S. Nadarajah
function entropy(d::GenericMvTDist)
    hdf, hdim = 0.5*d.df, 0.5*d.dim
    shdfhdim = hdf+hdim
    0.5*logdet(d.Σ)+hdim*log(d.df*pi)+lbeta(hdim, hdf)-lgamma(hdim)+shdfhdim*(digamma(shdfhdim)-digamma(hdf))
end

# evaluation (for GenericMvTDist)

insupport{T<:Real}(d::AbstractMvTDist, x::AbstractVector{T}) = 
  length(d) == length(x) && allfinite(x)

function sqmahal{T<:Real}(d::GenericMvTDist, x::DenseVector{T}) 
    z::Vector{Float64} = d.zeromean ? x : x - d.μ
    invquad(d.Σ, z) 
end

function sqmahal!{T<:Real}(r::DenseArray, d::GenericMvTDist, x::DenseMatrix{T})
    z::Matrix{Float64} = d.zeromean ? x : x .- d.μ
    invquad!(r, d.Σ, z)
end

sqmahal{T<:Real}(d::AbstractMvTDist, x::DenseMatrix{T}) = sqmahal!(Array(Float64, size(x, 2)), d, x)


function mvtdist_consts(d::AbstractMvTDist)
    hdf = 0.5 * d.df
    hdim = 0.5 * d.dim
    shdfhdim = hdf + hdim
    v = lgamma(shdfhdim) - lgamma(hdf) - hdim*log(d.df) - hdim*log(pi) - 0.5*logdet(d.Σ)
    return (shdfhdim, v)
end

function _logpdf{T<:Real}(d::AbstractMvTDist, x::DenseVector{T})
    shdfhdim, v = mvtdist_consts(d)
    v - shdfhdim * log1p(sqmahal(d, x) / d.df)
end

function _logpdf!{T<:Real}(r::DenseArray, d::AbstractMvTDist, x::DenseMatrix{T})
    sqmahal!(r, d, x)
    shdfhdim, v = mvtdist_consts(d)
    for i = 1:size(x, 2)
        r[i] = v - shdfhdim * log1p(r[i] / d.df)
    end
    return r
end

_pdf!{T<:Real}(r::DenseArray, d::AbstractMvNormal, x::DenseMatrix{T}) = exp!(_logpdf!(r, d, x))

function gradlogpdf{T<:Real}(d::GenericMvTDist, x::DenseVector{T})
    z::Vector{Float64} = d.zeromean ? x : x - d.μ
    prz = invscale(d)*z
    -((d.df + d.dim) / (d.df + dot(z, prz))) * prz
end

# Sampling (for GenericMvTDist)

function _rand!{T<:Real}(d::GenericMvTDist, x::DenseVector{T})
    chisqd = Chisq(d.df)
    y = sqrt(rand(chisqd)/(d.df))
    unwhiten!(d.Σ, randn!(x))
    broadcast!(/, x, x, y)
    if !d.zeromean
        broadcast!(+, x, x, d.μ)
    end
    x
end

function _rand!{T<:Real}(d::GenericMvTDist, x::DenseMatrix{T})
    cols = size(x,2)
    chisqd = Chisq(d.df)
    y = Array(Float64, 1, cols)
    unwhiten!(d.Σ, randn!(x))
    rand!(chisqd, y)
    y = sqrt(y/(d.df))
    broadcast!(/, x, x, y)
    if !d.zeromean
        broadcast!(+, x, x, d.μ)
    end
    x
end
