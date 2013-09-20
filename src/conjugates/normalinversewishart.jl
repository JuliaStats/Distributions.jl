
# Used "Conjugate Bayesian analysis of the Gaussian distribution" by Murphy as
# a reference.  Note that there were some typos in that document so the code
# here may not correspond exactly.

import NumericExtensions.PDMat
import NumericExtensions.invquad

immutable NormalInverseWishart <: Distribution
    dim::Int
    zeromean::Bool
    mu::Vector{Float64}
    kappa::Float64              # This scales precision (inverse covariance)
    Lamchol::Cholesky{Float64}  # Covariance matrix (well, sqrt of one)
    nu::Float64

    function NormalInverseWishart(mu::Vector{Float64}, kappa::Real,
                                  Lamchol::Cholesky{Float64}, nu::Real)
        # Probably should put some error checking in here
        d = length(mu)
        zmean::Bool = true
        for i = 1:d
            if mu[i] != 0.
                zmean = false
                break
            end
        end
        new(d, zmean, mu, float64(kappa), Lamchol, float64(nu))
    end
end

function NormalInverseWishart(mu::Vector{Float64}, kappa::Real,
                              Lambda::Matrix{Float64}, nu::Real)
    NormalInverseWishart(mu, kappa, cholfact(Lambda), nu)
    
end

function insupport(::Type{NormalInverseWishart}, x::Vector{Float64}, Sig::Matrix{Float64})
    return (all(isfinite(x)) &&
           size(Sig, 1) == size(Sig, 2) &&
           isApproxSymmmetric(Sig) &&
           size(Sig, 1) == length(x) &&
           hasCholesky(Sig))
end

pdf(niw::NormalInverseWishart, x::Vector{Float64}, Sig::Matrix{Float64}) =
        exp(logpdf(niw, x, Sig))

function logpdf(niw::NormalInverseWishart, x::Vector{Float64}, Sig::Matrix{Float64})
    if !insupport(NormalInverseWishart, x, Sig)
        return -Inf
    else
        p = size(x, 1)

        nu = niw.nu
        kappa = niw.kappa
        mu = niw.mu
        Lamchol = niw.Lamchol
        hnu = 0.5 * nu
        hp = 0.5 * p
    
        # Normalization
        logp::Float64 = hnu * logdet(Lamchol)
        logp -= hnu * p * log(2.)
        logp -= lpgamma(p, hnu)
        logp -= hp * (log(2.*pi) - log(kappa))
        
        # Inverse-Wishart
        logp -= (hnu + hp + 1.) * logdet(Sig)
        logp -= 0.5 * trace(Sig \ (Lamchol[:U]' * Lamchol[:U]))
        
        # Normal
        z = niw.zeromean ? x : x - mu
        logp -= 0.5 * kappa * invquad(PDMat(Sig), z) 

        return logp

    end
end

function rand(niw::NormalInverseWishart)
    Sig = rand(InverseWishart(niw.nu, niw.Lamchol))
    mu = rand(MvNormal(niw.mu, Sig ./ niw.kappa))
    return (mu, Sig)
end


function posterior(prior::NormalInverseWishart, ss::MvNormalStats)
    mu0 = prior.mu
    kappa0 = prior.kappa
    LamC0 = prior.Lamchol
    nu0 = prior.nu

    kappa = kappa0 + ss.tw
    mu = (kappa0.*mu0 + ss.s) ./ kappa
    nu = nu0 + ss.tw

    Lam0 = LamC0[:U]'*LamC0[:U]
    z = prior.zeromean ? ss.m : ss.m - mu0
    Lam = Lam0 + ss.s2 + kappa0*ss.tw/kappa*(z*z')

    return NormalInverseWishart(mu, kappa, cholfact(Lam), nu)
end

function posterior{T<:Real}(prior::NormalInverseWishart, ::Type{MvNormal}, X::Matrix{T})
    return posterior(prior, suffstats(MvNormal, X))
end

function posterior{T<:Real}(prior::NormalInverseWishart, ::Type{MvNormal}, X::Matrix{T}, w::Array{Float64})
    return posterior(prior, suffstats(MvNormal, X, w))
end

function posterior_rand{T<:Real}(prior::NormalInverseWishart, ::Type{MvNormal}, X::Matrix{T})
    return rand(posterior(prior, suffstats(MvNormal, X)))
end

function posterior_rand{T<:Real}(prior::NormalInverseWishart, ::Type{MvNormal}, x::Matrix{T}, w::Array{Float64})
    return rand(posterior(prior, suffstats(MvNormal, X, w)))
end

function posterior_sample{T<:Real}(prior::NormalInverseWishart, ::Type{MvNormal}, X::Matrix{T})
    mu, Sig = rand(posterior(prior, suffstats(MvNormal, X)))
    return MvNormal(mu, Sig)
end

function posterior_sample{T<:Real}(prior::NormalInverseWishart, ::Type{MvNormal}, X::Matrix{T}, w::Array{Float64})
    mu, Sig = rand(posterior(prior, suffstats(MvNormal, X, w)))
    return Normal(mu, Sig)
end
