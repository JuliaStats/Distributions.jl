
# Used "Conjugate Bayesian analysis of the Gaussian distribution" by Murphy as
# a reference.  Note that there were some typos in that document so the code
# here may not correspond exactly.

immutable NormalGamma <: Distribution
    mu::Float64
    nu::Float64     # scales precision of Normal
    shape::Float64  
    rate::Float64

    function NormalGamma(mu::Real, nu::Real, sh::Real, r::Real)
    	nu > zero(nu) && sh > zero(sh) && r > zero(r) || error("Both shape and scale must be positive")
    	new(mu, nu, sh, r)
    end
end

mu(d::NormalGamma) = d.mu
nu(d::NormalGamma) = d.nu
shape(d::NormalGamma) = d.shape
scale(d::NormalGamma) = 1. / d.rate
rate(d::NormalGamma) = d.rate

insupport(::Type{NormalGamma}, x::Real, tau2::Real) = 
    isfinite(x) && zero(tau2) <= tau2 < Inf

# Probably should guard agains dividing by and taking the log of 0.
function pdf(d::NormalGamma, x::Real, tau2::Real)
    Zinv = d.rate.^d.shape / gamma(d.shape) * sqrt(d.nu / (2.*pi))
    return Zinv * tau2.^(d.shape-0.5) * exp(-0.5*tau2*(d.nu*(x-d.mu).^2 + 2.*d.rate))
end
function logpdf(d::NormalGamma, x::Real, tau2::Real)
    lZinv = d.shape*log(d.rate) - lgamma(d.shape) + 0.5*(log(d.nu) - log(2.*pi))
    return lZinv + (d.shape-0.5)*log(tau2) - 0.5*tau2*(d.nu*(x-d.mu).^2 + 2*d.rate)
end

function rand(d::NormalGamma)
    # Guard against invalid precisions
    tau2 = rand(Gamma(d.shape, d.rate))
    if tau2 <= zero(Float64)
        tau2 = eps(Float64)
    end
    mu = rand(Normal(d.mu, 1./(tau2*d.nu)))
    return mu, tau2
end

function posterior(prior::NormalGamma, ss::NormalStats)
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

function posterior{T<:Real}(prior::NormalGamma, ::Type{Normal}, x::Array{T})
    return posterior(prior, suffstats(Normal, x))
end

function posterior{T<:Real}(prior::NormalGamma, ::Type{Normal}, x::Array{T}, w::Array{Float64})
    return posterior(prior, suffstats(Normal, x, w))
end

function posterior_sample{T<:Real}(prior::NormalGamma, ::Type{Normal}, x::Array{T})
    mu, tau2 = rand(posterior(prior, suffstats(Normal, x)))
    return Normal(mu, 1./sqrt(tau2))
end

function posterior_sample{T<:Real}(prior::NormalGamma, ::Type{Normal}, x::Array{T}, w::Array{Float64})
    mu, tau2 = rand(posterior(prior, suffstats(Normal, x, w)))
    return Normal(mu, 1./sqrt(tau2))
end
