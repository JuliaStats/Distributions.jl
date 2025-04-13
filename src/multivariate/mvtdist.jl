# Multivariate t-distribution

## Generic multivariate t-distribution class

abstract type AbstractMvTDist <: ContinuousMultivariateDistribution end

struct GenericMvTDist{T<:Real, Cov<:AbstractPDMat, Mean<:AbstractVector} <: AbstractMvTDist
    df::T # non-integer degrees of freedom allowed
    dim::Int
    μ::Mean
    Σ::Cov

    function GenericMvTDist{T,Cov,Mean}(df::T, dim::Int, μ::Mean, Σ::AbstractPDMat{T}) where {T,Cov,Mean}
      df > zero(df) || error("df must be positive")
      new{T,Cov,Mean}(df, dim, μ, Σ)
    end
end

function GenericMvTDist(df::T, μ::Mean, Σ::Cov) where {Cov<:AbstractPDMat, Mean<:AbstractVector, T<:Real}
    d = length(μ)
    size(Σ, 1) == d || throw(DimensionMismatch("The dimensions of μ and Σ are inconsistent."))
    R = Base.promote_eltype(T, μ, Σ)
    S = convert(AbstractArray{R}, Σ)
    m = convert(AbstractArray{R}, μ)
    GenericMvTDist{R, typeof(S), typeof(m)}(R(df), d, m, S)
end

function GenericMvTDist(df::Real, Σ::AbstractPDMat)
    R = Base.promote_eltype(df, Σ)
    GenericMvTDist(df, Zeros{R}(size(Σ, 1)), Σ)
end

GenericMvTDist{T,Cov,Mean}(df, μ, Σ) where {T,Cov,Mean} =
    GenericMvTDist(convert(T,df), convert(Mean, μ), convert(Cov, Σ))

### Conversion
function convert(::Type{GenericMvTDist{T}}, d::GenericMvTDist) where T<:Real
    S = convert(AbstractArray{T}, d.Σ)
    m = convert(AbstractArray{T}, d.μ)
    GenericMvTDist{T, typeof(S), typeof(m)}(T(d.df), d.dim, m, S)
end
Base.convert(::Type{GenericMvTDist{T}}, d::GenericMvTDist{T}) where {T<:Real} = d

function convert(::Type{GenericMvTDist{T}}, df, dim, μ::AbstractVector, Σ::AbstractPDMat) where T<:Real
    S = convert(AbstractArray{T}, Σ)
    m = convert(AbstractArray{T}, μ)
    GenericMvTDist{T, typeof(S), typeof(m)}(T(df), dim, m, S)
end

## Construction of multivariate normal with specific covariance type

const IsoTDist  = GenericMvTDist{Float64, ScalMat{Float64}, Vector{Float64}}
const DiagTDist = GenericMvTDist{Float64, PDiagMat{Float64,Vector{Float64}}, Vector{Float64}}
const MvTDist = GenericMvTDist{Float64, PDMat{Float64,Matrix{Float64}}, Vector{Float64}}

MvTDist(df::Real, μ::Vector{<:Real}, C::PDMat) = GenericMvTDist(df, μ, C)
MvTDist(df::Real, C::PDMat) = GenericMvTDist(df, C)
MvTDist(df::Real, μ::Vector{<:Real}, Σ::Matrix{<:Real}) = GenericMvTDist(df, μ, PDMat(Σ))
MvTDist(df::Real, Σ::Matrix{<:Real}) = GenericMvTDist(df, PDMat(Σ))

DiagTDist(df::Real, μ::Vector{<:Real}, C::PDiagMat) = GenericMvTDist(df, μ, C)
DiagTDist(df::Real, C::PDiagMat) = GenericMvTDist(df, C)
DiagTDist(df::Real, μ::Vector{<:Real}, σ::Vector{<:Real}) = GenericMvTDist(df, μ, PDiagMat(abs2.(σ)))

IsoTDist(df::Real, μ::Vector{<:Real}, C::ScalMat) = GenericMvTDist(df, μ, C)
IsoTDist(df::Real, C::ScalMat) = GenericMvTDist(df, C)
IsoTDist(df::Real, μ::Vector{<:Real}, σ::Real) = GenericMvTDist(df, μ, ScalMat(length(μ), abs2(σ)))
IsoTDist(df::Real, d::Int, σ::Real) = GenericMvTDist(df, ScalMat(d, abs2(σ)))

## convenient function to construct distributions of proper type based on arguments

mvtdist(df::Real, μ::Vector, C::AbstractPDMat) = GenericMvTDist(df, μ, C)
mvtdist(df::Real, C::AbstractPDMat) = GenericMvTDist(df, C)

mvtdist(df::Real, μ::Vector, σ::Real) = GenericMvTDist(df, μ, ScalMat(length(μ), abs2(σ)))
mvtdist(df::Real, d::Int, σ::Real) = GenericMvTDist(df, μ, ScalMat(d, abs2(σ)))
mvtdist(df::Real, μ::Vector, σ::Vector) = GenericMvTDist(df, μ, PDiagMat(abs2.(σ)))
mvtdist(df::Real, μ::Vector, Σ::Matrix) = GenericMvTDist(df, μ, PDMat(Σ))
mvtdist(df::Real, Σ::Matrix) = GenericMvTDist(df, PDMat(Σ))

# mvtdist(df::Real, μ::Vector{<:Real}, σ::Real) = IsoTDist(df, μ, σ)
# mvtdist(df::Real, d::Int, σ::Real) = IsoTDist(d, σ)
mvtdist(df::Real, μ::Vector{<:Real}, σ::Vector{<:Real}) = DiagTDist(df, μ, σ)
mvtdist(df::Real, μ::Vector{<:Real}, Σ::Matrix{<:Real}) = MvTDist(df, μ, Σ)
mvtdist(df::Real, Σ::Matrix{<:Real}) = MvTDist(df, Σ)

# Basic statistics

length(d::GenericMvTDist) = d.dim

mean(d::GenericMvTDist) = d.df>1 ? d.μ : NaN
mode(d::GenericMvTDist) = d.μ
modes(d::GenericMvTDist) = [mode(d)]

var(d::GenericMvTDist) = d.df>2 ? (d.df/(d.df-2))*diag(d.Σ) : Float64[NaN for i = 1:d.dim]
scale(d::GenericMvTDist) = Matrix(d.Σ)
cov(d::GenericMvTDist) = d.df>2 ? (d.df/(d.df-2))*Matrix(d.Σ) : NaN*ones(d.dim, d.dim)
invscale(d::GenericMvTDist) = Matrix(inv(d.Σ))
invcov(d::GenericMvTDist) = d.df>2 ? ((d.df-2)/d.df)*Matrix(inv(d.Σ)) : NaN*ones(d.dim, d.dim)
logdet_cov(d::GenericMvTDist) = d.df>2 ? logdet((d.df/(d.df-2))*d.Σ) : NaN

params(d::GenericMvTDist) = (d.df, d.μ, d.Σ)
@inline partype(d::GenericMvTDist{T}) where {T} = T
Base.eltype(::Type{<:GenericMvTDist{T}}) where {T} = T

# For entropy calculations see "Multivariate t Distributions and their Applications", S. Kotz & S. Nadarajah
function entropy(d::GenericMvTDist)
    hdf, hdim = 0.5*d.df, 0.5*d.dim
    shdfhdim = hdf+hdim
    0.5*logdet(d.Σ) + hdim*log(d.df*pi) + logbeta(hdim, hdf) - loggamma(hdim) + shdfhdim*(digamma(shdfhdim) - digamma(hdf))
end

# evaluation (for GenericMvTDist)

insupport(d::AbstractMvTDist, x::AbstractVector{T}) where {T<:Real} =
    length(d) == length(x) && all(isfinite, x)

sqmahal(d::GenericMvTDist, x::AbstractVector{<:Real}) = invquad(d.Σ, x - d.μ)

function sqmahal!(r::AbstractArray, d::GenericMvTDist, x::AbstractMatrix{<:Real})
    invquad!(r, d.Σ, x .- d.μ)
end

sqmahal(d::AbstractMvTDist, x::AbstractMatrix{T}) where {T<:Real} = sqmahal!(Vector{T}(undef, size(x, 2)), d, x)


function mvtdist_consts(d::AbstractMvTDist)
    H = convert(eltype(d), 0.5)
    logpi = convert(eltype(d), log(pi))
    hdf = H * d.df
    hdim = H * d.dim
    shdfhdim = hdf + hdim
    v = loggamma(shdfhdim) - loggamma(hdf) - hdim*log(d.df) - hdim*logpi - H*logdet(d.Σ)
    return (shdfhdim, v)
end

function _logpdf(d::AbstractMvTDist, x::AbstractVector{T}) where T<:Real
    shdfhdim, v = mvtdist_consts(d)
    v - shdfhdim * log1p(sqmahal(d, x) / d.df)
end

function _logpdf!(r::AbstractArray{<:Real}, d::AbstractMvTDist, x::AbstractMatrix{<:Real})
    sqmahal!(r, d, x)
    shdfhdim, v = mvtdist_consts(d)
    for i = 1:size(x, 2)
        r[i] = v - shdfhdim * log1p(r[i] / d.df)
    end
    return r
end

function gradlogpdf(d::GenericMvTDist, x::AbstractVector{<:Real})
    z = x - d.μ
    prz = invscale(d)*z
    -((d.df + d.dim) / (d.df + dot(z, prz))) * prz
end

# Sampling (for GenericMvTDist)
function _rand!(rng::AbstractRNG, d::GenericMvTDist, x::AbstractVector{<:Real})
    chisqd = Chisq{partype(d)}(d.df)
    y = sqrt(rand(rng, chisqd) / d.df)
    unwhiten!(d.Σ, randn!(rng, x))
    x .= x ./ y .+ d.μ
    x
end

function _rand!(rng::AbstractRNG, d::GenericMvTDist, x::AbstractMatrix{T}) where T<:Real
    cols = size(x,2)
    chisqd = Chisq{partype(d)}(d.df)
    y = Matrix{T}(undef, 1, cols)
    unwhiten!(d.Σ, randn!(rng, x))
    rand!(rng, chisqd, y)
    x .= x ./ sqrt.(y ./ d.df) .+ d.μ
    x
end
