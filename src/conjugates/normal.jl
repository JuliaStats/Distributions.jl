# Conjugates related to normal distribution
#
#	Normal - Normal (with known sigma)
#	InverseGamma - Normal (with known mu)
#	NormalInverseGamma - Normal
#

# known sigma (prior on mu)

function posterior(prior::Normal, ss::NormalKnownSigmaStats)
	μ0 = prior.μ
	c0 = 1.0 / abs2(prior.σ)
	c1 = 1.0 / abs2(ss.σ)

	τ = c0 + ss.tw * c1
	μ1 = (μ0 * c0 + ss.s * c1) / τ
	σ1 = sqrt(1 / τ)
	return Normal(μ1, σ1)	
end

function posterior(prior::InverseGamma, ss::NormalKnownMuStats)
	α1 = prior.shape + 0.5*ss.tw
	β1 = rate(prior) + 0.5*ss.s2
	return InverseGamma(α1, 1.0 / β1)
end

function posterior(prior::Gamma, ss::NormalKnownMuStats)
    a = prior.shape + 0.5*ss.tw
    b = rate(prior) + 0.5*ss.s2
    return Gamma(a, 1. / b)
end

function posterior{T<:Real}(prior::(Normal, Float64), ::Type{Normal}, x::Array{T}) 
	pri_μ::Normal, σ::Float64 = prior
	posterior(pri_μ, suffstats(NormalKnownSigma(σ), x))
end


function posterior{T<:Real}(prior::(Normal, Float64), ::Type{Normal}, x::Array{T}, w::Array{Float64}) 
	pri_μ::Normal, σ::Float64 = prior
	posterior(pri_μ, suffstats(NormalKnownSigma(σ), x, w))
end

function posterior_mode(prior::Normal, ss::NormalKnownSigmaStats)
	μ0 = prior.μ
	c0 = 1.0 / abs2(prior.σ)
	c1 = 1.0 / abs2(ss.σ)
	return (μ0 * c0 + ss.s * c1) / (c0 + ss.tw * c1)
end

function posterior_mode{T<:Real}(prior::(Normal, Float64), ::Type{Normal}, x::Array{T}) 
	pri_μ::Normal, σ::Float64 = prior
	posterior_mode(pri_μ, suffstats(NormalKnownSigma(σ), x))
end

function posterior_mode{T<:Real}(prior::(Normal, Float64), ::Type{Normal}, x::Array{T}, w::Array{Float64}) 
	pri_μ::Normal, σ::Float64 = prior
	posterior_mode(pri_μ, suffstats(NormalKnownSigma(σ), x, w))
end

function fit_map{T<:Real}(prior::(Normal, Float64), ::Type{Normal}, x::Array{T})
	pri_μ::Normal, σ::Float64 = prior
	μ = posterior_mode(pri_μ, suffstats(NormalKnownSigma(σ), x))
	Normal(μ, σ)
end

function fit_map{T<:Real}(prior::(Normal, Float64), ::Type{Normal}, x::Array{T}, w::Array{Float64})
	pri_μ::Normal, σ::Float64 = prior
	μ = posterior_mode(pri_μ, suffstats(NormalKnownSigma(σ), x, w))
	Normal(μ, σ)
end


# known mu (prior on sigma)

function posterior(prior::InverseGamma, ss::NormalKnownMuStats)
	α1 = prior.shape + ss.tw / 2
	β1 = prior.scale + ss.s2 / 2
	return InverseGamma(α1, β1)
end

function posterior{T<:Real}(prior::(Float64, InverseGamma), ::Type{Normal}, x::Array{T}) 
	μ::Float64 = prior[1]
	pri_σ::InverseGamma = prior[2]
	posterior(pri_σ, suffstats(NormalKnownMu(μ), x))
end

function posterior{T<:Real}(prior::(Float64, InverseGamma), ::Type{Normal}, x::Array{T}, w::Array{Float64}) 
	μ::Float64 = prior[1]
	pri_σ::InverseGamma = prior[2]
	posterior(pri_σ, suffstats(NormalKnownMu(μ), x, w))
end

function posterior{T<:Real}(prior::(Float64, Gamma), ::Type{Normal}, x::Array{T}) 
	μ::Float64 = prior[1]
	pri_tau::Gamma = prior[2]
	posterior(pri_tau, suffstats(NormalKnownMu(μ), x))
end

function posterior{T<:Real}(prior::(Float64, Gamma), ::Type{Normal}, x::Array{T}, w::Array{Float64}) 
	μ::Float64 = prior[1]
	pri_tau::Gamma = prior[2]
	posterior(pri_tau, suffstats(NormalKnownMu(μ), x, w))
end



# The NormalInverseGamma version is in normalinversegamma.jl

function posterior_sample{T<:Real}(prior::(Normal, Float64), ::Type{Normal}, x::Array{T})
    mu = rand(posterior(prior, suffstats(Normal, x)))
    return Normal(mu, prior[2]) 
end

function posterior_sample{T<:Real}(prior::(Normal, Float64), ::Type{Normal}, x::Array{T}, w::Array{Float64})
    mu = rand(posterior(prior, suffstats(Normal, x, w)))
    return Normal(mu, prior[2])
end

function posterior_sample{T<:Real}(prior::(Float64, InverseGamma), ::Type{Normal}, x::Array{T})
    sig2 = rand(posterior(prior, suffstats(Normal, x)))
    return Normal(prior[1], sqrt(sig2)) 
end

function posterior_sample{T<:Real}(prior::(Float64, InverseGamma), ::Type{Normal}, x::Array{T}, w::Array{Float64})
    sig2 = rand(posterior(prior, suffstats(Normal, x, w)))
    return Normal(prior[1], sqrt(sig2))
end

function posterior_sample{T<:Real}(prior::(Float64, Gamma), ::Type{Normal}, x::Array{T})
    tau2 = rand(posterior(prior, suffstats(Normal, x)))
    return Normal(prior[1], sqrt(1./tau2)) 
end

function posterior_sample{T<:Real}(prior::(Float64, Gamma), ::Type{Normal}, x::Array{T}, w::Array{Float64})
    tau2 = rand(posterior(prior, suffstats(Normal, x, w)))
    return Normal(prior[1], 1./sqrt(tau2))
end

