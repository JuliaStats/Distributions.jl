# Conjugates for Multivariate Normal

#### Generic MvNormal -- Generic MvNormal (Σ is known)

function posterior_canon(prior::GenericMultivariateNormal, ss::GenericMvNormalKnownSigmaStats)
    invΣ0 = inv(prior.Σ)
    μ0 = prior.μ
    invΣp = add_scal(invΣ0, ss.invΣ, ss.tw)
    h = add!(invΣ0 * μ0, ss.invΣ * ss.sx)
	return GenericMvNormalCanon(h, invΣp)
end

function posterior_canon{Pri<:GenericMultivariateNormal,Cov<:AbstractPDMat}(
    prior::(Pri, Cov), 
    G::Type{GenericMultivariateNormal{Cov}}, 
    x::Matrix) 

	μpri::Pri, Σ::Cov = prior
	posterior_canon(μpri, suffstats(GenericMvNormalKnownSigma{Cov}(Σ), x))
end

function posterior_canon{Pri<:GenericMultivariateNormal,Cov<:AbstractPDMat}(
    prior::(Pri, Cov), 
    G::Type{GenericMultivariateNormal{Cov}}, 
    x::Matrix, w::Array{Float64}) 

    μpri::Pri, Σ::Cov = prior
    posterior_canon(μpri, suffstats(GenericMvNormalKnownSigma{Cov}(Σ), x, w))
end

function posterior{Pri<:GenericMultivariateNormal,Cov<:AbstractPDMat}(
    prior::(Pri, Cov), 
    G::Type{GenericMultivariateNormal{Cov}}, 
    x::Matrix) 

    convert(Pri, posterior_canon(prior, G, x))
end

function posterior{Pri<:GenericMultivariateNormal,Cov<:AbstractPDMat}(
    prior::(Pri, Cov), 
    G::Type{GenericMultivariateNormal{Cov}}, 
    x::Matrix, w::Array{Float64}) 

    convert(Pri, posterior_canon(prior, G, x, w))
end

function complete{Pri<:GenericMultivariateNormal,Cov<:AbstractPDMat}(
    G::Type{GenericMultivariateNormal{Cov}},
    pri::(Pri, Cov), 
    μ::Vector{Float64})

    GenericMultivariateNormal(μ, pri[2]::Cov)
end


function posterior_canon(pri::(MvNormal, Matrix{Float64}), G::Type{MvNormal}, args...) 
    μpri::MvNormal, Σmat::Matrix{Float64} = pri
    posterior_canon((μpri, PDMat(Σmat)), G, args...)
end

function posterior(pri::(MvNormal, Matrix{Float64}), G::Type{MvNormal}, args...) 
    μpri::MvNormal, Σmat::Matrix{Float64} = pri
    posterior((μpri, PDMat(Σmat)), G, args...)
end

function complete(G::Type{MvNormal}, pri::(MvNormal, Matrix{Float64}), μ::Vector{Float64})
    GenericMultivariateNormal(μ, PDMat(pri[2]))
end


#### NormalInverseWishart -- Normal

function posterior_canon(prior::NormalInverseWishart, ss::MvNormalStats)
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

complete(G::Type{MvNormal}, pri::NormalInverseWishart, s::(Vector{Float64}, Matrix{Float64})) = MvNormal(s...)


#### NormalWishart -- Normal

function posterior_canon(prior::NormalWishart, ss::MvNormalStats)
    mu0 = prior.mu
    kappa0 = prior.kappa
    TC0 = prior.Tchol
    nu0 = prior.nu

    kappa = kappa0 + ss.tw
    nu = nu0 + ss.tw
    mu = (kappa0.*mu0 + ss.s) ./ kappa

    Lam0 = TC0[:U]'*TC0[:U]
    z = prior.zeromean ? ss.m : ss.m - mu0
    Lam = Lam0 + ss.s2 + kappa0*ss.tw/kappa*(z*z')

    return NormalWishart(mu, kappa, cholfact(Lam), nu)
end

complete(G::Type{MvNormal}, pri::NormalWishart, s::(Vector{Float64}, Matrix{Float64})) = MvNormal(s[1], inv(PDMat(s[2])))

