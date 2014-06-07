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

dim(d::GenericMvTDist) = d.dim

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

function sqmahal(d::GenericMvTDist, x::Vector{Float64}) 
    z::Vector{Float64} = d.zeromean ? x : x - d.μ
    invquad(d.Σ, z) 
end

function sqmahal!(r::Array{Float64}, d::GenericMvTDist, x::Matrix{Float64})
    if !(size(x, 1) == dim(d) && size(x, 2) == length(r))
        throw(ArgumentError("Inconsistent argument dimensions."))
    end
    z::Matrix{Float64} = d.zeromean ? x : bsubtract(x, d.μ, 1)
    invquad!(r, d.Σ, z)
end

# generic PDF evaluation (appliable to AbstractMvTDist)

insupport{T<:Real}(d::AbstractMvTDist, x::Vector{T}) = dim(d) == length(x) && allfinite(x)
insupport{G<:AbstractMvTDist,T<:Real}(::Type{G}, x::Vector{T}) = allfinite(x)


sqmahal(d::AbstractMvTDist, x::Matrix{Float64}) = sqmahal!(Array(Float64, size(x, 2)), d, x)

function logpdf(d::AbstractMvTDist, x::Vector{Float64})
  hdf, hdim = 0.5*d.df, 0.5*d.dim
  shdfhdim = hdf+hdim
  lgamma(shdfhdim)-lgamma(hdf)-hdim*log(d.df)-hdim*log(pi)-0.5*logdet(d.Σ)-shdfhdim*log(1+sqmahal(d, x)/d.df)
end

function logpdf!(r::Array{Float64}, d::AbstractMvTDist, x::Matrix{Float64})
  sqmahal!(r, d, x)
  hdf, hdim = 0.5*d.df, 0.5*d.dim
  shdfhdim = hdf+hdim
  for i = 1:size(x, 2)
    r[i] = lgamma(shdfhdim)-lgamma(hdf)-hdim*log(d.df)-hdim*log(pi)-0.5*logdet(d.Σ)-shdfhdim*log(1+r[i]/d.df)
  end 
  r
end

function gradlogpdf(d::GenericMvTDist, x::Vector{Float64})
  z::Vector{Float64} = d.zeromean ? x : x - d.μ
  prz = invscale(d)*z
  -((d.df + d.dim) / (d.df + dot(z, prz))) * prz
end

# Sampling (for GenericMvTDist)

function rand!(d::GenericMvTDist, x::Vector{Float64})
  normdim = d.dim
  normd = GenericMvNormal{typeof(d.Σ)}(normdim, true, zeros(normdim), d.Σ)
  chisqd = Chisq(d.df)
  y = Array(Float64, d.dim)
  unwhiten!(normd.Σ, randn!(x))
  rand!(chisqd, y)
  y = sqrt(y/(d.df))
  x = x./y
  if !d.zeromean
    x = x+d.μ
  end
  x
end

function rand!(d::GenericMvTDist, x::Matrix{Float64})
  normdim = d.dim
  normd = GenericMvNormal{typeof(d.Σ)}(normdim, true, zeros(normdim), d.Σ)
  chisqd = Chisq(d.df)
  y = Array(Float64, d.dim)
  unwhiten!(normd.Σ, randn!(x))
  rand!(chisqd, y)
  y = sqrt(y/(d.df))
  bdivide!(x, y, 1)
  x = x./y
  if !d.zeromean
    x = x+d.μ
  end
  x
end
