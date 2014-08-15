
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
    tau2 = rand(Gamma(d.shape, scale(d)))
    if tau2 <= zero(Float64)
        tau2 = eps(Float64)
    end
    mu = rand(Normal(d.mu, sqrt(1./(tau2*d.nu))))
    return mu, tau2
end

