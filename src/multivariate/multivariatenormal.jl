# Multivariate Normal distribution

immutable MultivariateNormal{Cov<:AbstractPDMat} <: ContinuousMultivariateDistribution
    dim::Int
    μ::Vector{Float64}
    Σ::Cov
end

function MultivariateNormal{Cov<:AbstractPDMat}(μ::Vector{Float64}, Σ::Cov)
    d = length(μ)
    if dim(Σ) != d
        throw(ArgumentError("The dimensions of μ and Σ are inconsistent."))
    end
    MultivariateNormal{Cov}(d, μ, Σ)
end

function MultivariateNormal{Cov<:AbstractPDMat}(Σ::Cov)
    d = dim(Σ)
    MultivariateNormal{Cov}(d, zeros(d), Σ)    
end

MultivariateNormal(μ::Vector{Float64}, σ2::Float64) = MultivariateNormal(μ, ScalMat(length(μ), σ2))
MultivariateNormal(μ::Vector{Float64}, σ2::Vector{Float64}) = MultivariateNormal(μ, PDiagMat(σ2))
MultivariateNormal(μ::Vector{Float64}, Σ::Matrix{Float64}) = MultivariateNormal(μ, PDMat(Σ))

const MvNormal = MultivariateNormal


# Basic statistics

dim(d::MvNormal) = d.dim

mean(d::MvNormal) = d.μ

var(d::MvNormal) = diag(d.Σ)

cov(d::MvNormal) = full(d.Σ)

logdet_cov(d::MvNormal) = logdet(d.Σ)

mode(d::MvNormal) = d.μ

modes(d::MvNormal) = [mode(d)]

entropy(d::MvNormal) = 0.5 * (dim(d) * (float64(log2π) + 1.0) + logdet_cov(d))


# PDF evaluation

_gauss_c0(g::MvNormal) = -0.5 * (dim(g) * float64(log2π) + logdet_cov(g))

sqmahal(d::MvNormal, x::Vector{Float64}) = invquad(d.Σ, x - d.μ)

function sqmahal!(r::Array{Float64}, d::MvNormal, x::Matrix{Float64})
    if !(size(x, 1) == dim(d) && size(x, 2) == length(r))
        throw(ArgumentError("Inconsistent argument dimensions."))
    end
    z::Matrix{Float64} = bsubtract(x, d.μ, 1)
    invquad!(r, d.Σ, z)
end

sqmahal(d::MvNormal, x::Matrix{Float64}) = sqmahal!(Array(Float64, size(x, 2)), d, x)

logpdf(d::MvNormal, x::Vector{Float64}) = _gauss_c0(d) - 0.5 * sqmahal(d, x) 

function logpdf!(r::Array{Float64}, d::MvNormal, x::Matrix{Float64})
    sqmahal!(r, d, x)
    c0::Float64 = _gauss_c0(d)
    for i = 1:size(x, 2)
        r[i] = c0 - 0.5 * r[i]
    end 
    r
end

