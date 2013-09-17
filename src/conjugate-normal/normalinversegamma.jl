
# Used "Conjugate Bayesian analysis of the Gaussian distribution" by Murphy as
# a reference.  Note that there were some typos in that document so the code
# here may not correspond exactly.

immutable NormalInverseGamma <: Distribution
    mu::Float64
    v0::Float64     # scales variance of Normal
    shape::Float64  
    scale::Float64

    function NormalInverseGamma(mu::Real, v0::Real, sh::Real, r::Real)
    	v0 > zero(v0) && sh > zero(sh) && r > zero(r) || error("Both shape and scale must be positive")
    	new(mu, v0, sh, r)
    end
end

mu(d::NormalInverseGamma) = d.mu
v0(d::NormalInverseGamma) = d.v0
shape(d::NormalInverseGamma) = d.shape
scale(d::NormalInverseGamma) = d.scale
rate(d::NormalInverseGamma) = 1. / d.scale

insupport(::Type{NormalInverseGamma}, x::Real, sig2::Real) = 
    isfinite(x) && zero(sig2) <= sig2 < Inf 

# Probably should guard agains dividing by and taking the log of 0.
function pdf(d::NormalInverseGamma, x::Real, sig2::Real)
    Zinv = d.scale.^d.shape / gamma(d.shape) / sqrt(d.v0 * 2.*pi)
    return Zinv * 1./(sqrt(sig2)*sig2.^(d.shape+1.)) * exp(-d.scale/sig2 - 0.5/(sig2*d.v0)*(x-d.mu).^2)
end
function logpdf(d::NormalInverseGamma, x::Real, sig2::Real)
    lZinv = d.shape*log(d.scale) - lgamma(d.shape) - 0.5*(log(d.v0) + log(2pi))
    return lZinv - 0.5*log(sig2) - (d.shape+1.)*log(sig2) - d.scale/sig2 - 0.5/(sig2*d.v0)*(x-d.mu).^2
end

function rand(d::NormalInverseGamma)
    # Guard against invalid precisions
    sig2 = rand(InverseGamma(d.shape, d.scale))
    if sig2 <= zero(Float64)
        sig2 = eps(Float64)
    end
    mu = rand(Normal(d.mu, sig2*d.v0))
    return mu, sig2
end

function posterior(prior::NormalInverseGamma, ss::NormalStats)
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

function posterior{T<:Real}(prior::NormalInverseGamma, ::Type{Normal}, x::Array{T})
    return posterior(prior, suffstats(Normal, x))
end

function posterior{T<:Real}(prior::NormalInverseGamma, ::Type{Normal}, x::Array{T}, w::Array{Float64})
    return posterior(prior, suffstats(Normal, x, w))
end

function posterior_sample{T<:Real}(prior::NormalInverseGamma, ::Type{Normal}, x::Array{T})
    mu, sig2 = rand(posterior(prior, suffstats(Normal, x)))
    return Normal(mu, sqrt(sig2))
end

function posterior_sample{T<:Real}(prior::NormalInverseGamma, ::Type{Normal}, x::Array{T}, w::Array{Float64})
    mu, sig2 = rand(posterior(prior, suffstats(Normal, x, w)))
    return Normal(mu, sqrt(sig2))
end
