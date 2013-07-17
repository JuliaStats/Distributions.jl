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

entropy(d::MvNormal) = 0.5 * (log2π + 1.0 + logdet_cov(d))


# PDF evaluation

