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

typealias NormalWithFloat64 @compat Tuple{Normal, Float64}

function posterior_canon(pri::NormalWithFloat64, G::Type{Normal}, x::Array)
	μpri::Normal, σ::Float64 = pri
	posterior_canon(μpri, suffstats(NormalKnownSigma(σ), x))
end

function posterior_canon(pri::NormalWithFloat64, G::Type{Normal}, x::Array, w::Array{Float64})
	μpri::Normal, σ::Float64 = pri
	posterior_canon(μpri, suffstats(NormalKnownSigma(σ), x, w))
end

function posterior(pri::NormalWithFloat64, G::Type{Normal}, x::Array)
	convert(Normal, posterior_canon(pri, G, x))
end

function posterior(pri::NormalWithFloat64, G::Type{Normal}, x::Array, w::Array{Float64})
	convert(Normal, posterior_canon(pri, G, x, w))
end

complete(G::Type{Normal}, pri::NormalWithFloat64, μ::Float64) = Normal(μ, pri[2])


#### InverseGamma on σ^2 (with μ known)

typealias Float64WithInverseGamma @compat Tuple{Float64, InverseGamma}

function posterior_canon(pri::InverseGamma, ss::NormalKnownMuStats)
	α1 = shape(pri) + 0.5 * ss.tw
	β1 = scale(pri) + 0.5 * ss.s2
	return InverseGamma(α1, β1)
end

function posterior_canon(pri::Float64WithInverseGamma, G::Type{Normal}, x::Array)
	μ::Float64, σ2pri::InverseGamma = pri
	posterior_canon(σ2pri, suffstats(NormalKnownMu(μ), x))
end

function posterior_canon(pri::Float64WithInverseGamma, G::Type{Normal}, x::Array, w::Array{Float64})
	μ::Float64, σ2pri::InverseGamma = pri
	posterior_canon(σ2pri, suffstats(NormalKnownMu(μ), x, w))
end

posterior(pri::Float64WithInverseGamma, G::Type{Normal}, x::Array) = posterior_canon(pri, G, x)
posterior(pri::Float64WithInverseGamma, G::Type{Normal}, x::Array, w::Array{Float64}) = posterior_canon(pri, G, x, w)

complete(G::Type{Normal}, pri::Float64WithInverseGamma, σ2::Float64) = Normal(pri[1], sqrt(σ2))


#### Gamma on precision σ^(-2) (with μ known)

typealias Float64WithGamma @compat Tuple{Float64, Gamma}

function posterior_canon(pri::Gamma, ss::NormalKnownMuStats)
	α1 = shape(pri) + 0.5 * ss.tw
	β1 = scale(pri) + 0.5 * ss.s2
	return Gamma(α1, β1)
end

function posterior_canon(pri::Float64WithGamma, G::Type{Normal}, x::Array)
	μ::Float64, τpri::Gamma = pri
	posterior_canon(τpri, suffstats(NormalKnownMu(μ), x))
end

function posterior_canon(pri::Float64WithGamma, G::Type{Normal}, x::Array, w::Array{Float64})
	μ::Float64, τpri::Gamma = pri
	posterior_canon(τpri, suffstats(NormalKnownMu(μ), x, w))
end

posterior(pri::Float64WithGamma, G::Type{Normal}, x::Array) = posterior_canon(pri, G, x)
posterior(pri::Float64WithGamma, G::Type{Normal}, x::Array, w::Array{Float64}) = posterior_canon(pri, G, x, w)

complete(G::Type{Normal}, pri::Float64WithGamma, τ::Float64) = Normal(pri[1], 1.0 / sqrt(τ))


#### NormalInverseGamma on (μ, σ^2)

function posterior_canon(prior::NormalInverseGamma, ss::NormalStats)
    mu0 = prior.mu
    v0 = prior.v0
    shape0 = prior.shape
    scale0 = prior.scale

    # ss.tw contains the number of observations if weight wasn't used to
    # compute the sufficient statistics.

    vn_inv = 1./v0 + ss.tw
    mu = (mu0/v0 + ss.s) / vn_inv  # ss.s = ss.tw*ss.m = n*xbar
    shape = shape0 + 0.5*ss.tw
    scale = scale0 + 0.5*ss.s2 + 0.5/(vn_inv*v0)*ss.tw*(ss.m-mu0).^2

    return NormalInverseGamma(mu, 1./vn_inv, shape, scale)
end

complete(G::Type{Normal}, pri::NormalInverseGamma, t::NTuple{2,Float64}) = Normal(t[1], sqrt(t[2]))


#### NormalGamma on (μ, σ^(-2))

function posterior_canon(prior::NormalGamma, ss::NormalStats)
    mu0 = prior.mu
    nu0 = prior.nu
    shape0 = prior.shape
    rate0 = prior.rate

    # ss.tw contains the number of observations if weight wasn't used to
    # compute the sufficient statistics.

    nu = nu0 + ss.tw
    mu = (nu0*mu0 + ss.s) / nu
    shape = shape0 + 0.5*ss.tw
    rate = rate0 + 0.5*ss.s2 + 0.5*nu0/nu*ss.tw*(ss.m-mu0).^2

    return NormalGamma(mu, nu, shape, rate)
end

complete(G::Type{Normal}, pri::NormalGamma, t::NTuple{2,Float64}) = Normal(t[1], 1. / sqrt(t[2]))
