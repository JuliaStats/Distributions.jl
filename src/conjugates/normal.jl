# Conjugates related to normal distribution
#
#	Normal - Normal (with known sigma)
#	InverseGamma - Normal (with known mu)
#	NormalInverseGamma - Normal
#

#### Normal prior on μ (with σ^2 known) 

function posterior_canon(pri::Normal, ss::NormalKnownSigmaStats)
	p0 = 1.0 / abs2(pri.σ)
	p1 = 1.0 / abs2(ss.σ)
	h = pri.μ * p0 + ss.sx * p1
	prec = p0 + ss.tw * p1
	NormalCanon(h, prec)	
end

function posterior_canon(pri::(Normal, Float64), G::Type{Normal}, x::Array)
	μpri::Normal, σ::Float64 = pri
	posterior_canon(μpri, suffstats(NormalKnownSigma(σ), x))
end

function posterior_canon(pri::(Normal, Float64), G::Type{Normal}, x::Array, w::Array{Float64})
	μpri::Normal, σ::Float64 = pri
	posterior_canon(μpri, suffstats(NormalKnownSigma(σ), x, w))
end

function posterior(pri::(Normal, Float64), G::Type{Normal}, x::Array)
	convert(Normal, posterior_canon(pri, G, x))
end

function posterior(pri::(Normal, Float64), G::Type{Normal}, x::Array, w::Array{Float64})
	convert(Normal, posterior_canon(pri, G, x, w))
end

complete(G::Type{Normal}, pri::(Normal, Float64), μ::Float64) = Normal(μ, pri[2])


#### InverseGamma on σ^2 (with μ known)

function posterior_canon(pri::InverseGamma, ss::NormalKnownMuStats)
	α1 = pri.shape + 0.5 * ss.tw
	β1 = pri.scale + 0.5 * ss.s2
	return InverseGamma(α1, β1)
end

function posterior_canon(pri::(Float64, InverseGamma), G::Type{Normal}, x::Array)
	μ::Float64, σ2pri::InverseGamma = pri
	posterior_canon(σ2pri, suffstats(NormalKnownMu(μ), x))
end

function posterior_canon(pri::(Float64, InverseGamma), G::Type{Normal}, x::Array, w::Array{Float64})
	μ::Float64, σ2pri::InverseGamma = pri
	posterior_canon(σ2pri, suffstats(NormalKnownMu(μ), x, w))
end

posterior(pri::(Float64, InverseGamma), G::Type{Normal}, x::Array) = posterior_canon(pri, G, x)
posterior(pri::(Float64, InverseGamma), G::Type{Normal}, x::Array, w::Array{Float64}) = posterior_canon(pri, G, x, w)

complete(G::Type{Normal}, pri::(Float64, InverseGamma), σ2::Float64) = Normal(pri[1], sqrt(σ2))


#### Gamma on precision σ^(-2) (with μ known)

function posterior_canon(pri::Gamma, ss::NormalKnownMuStats)
	α1 = pri.shape + 0.5 * ss.tw
	β1 = pri.scale + 0.5 * ss.s2
	return Gamma(α1, β1)
end

function posterior_canon(pri::(Float64, Gamma), G::Type{Normal}, x::Array)
	μ::Float64, τpri::Gamma = pri
	posterior_canon(τpri, suffstats(NormalKnownMu(μ), x))
end

function posterior_canon(pri::(Float64, Gamma), G::Type{Normal}, x::Array, w::Array{Float64})
	μ::Float64, τpri::Gamma = pri
	posterior_canon(τpri, suffstats(NormalKnownMu(μ), x, w))
end

posterior(pri::(Float64, Gamma), G::Type{Normal}, x::Array) = posterior_canon(pri, G, x)
posterior(pri::(Float64, Gamma), G::Type{Normal}, x::Array, w::Array{Float64}) = posterior_canon(pri, G, x, w)

complete(G::Type{Normal}, pri::(Float64, Gamma), τ::Float64) = Normal(pri[1], 1.0 / sqrt(τ))

