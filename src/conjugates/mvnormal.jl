# Conjugates for Multivariate Normal

#### Generic MvNormal -- Generic MvNormal (Σ is known)

typealias MvNormalWithCov{Pri<:MvNormal, Cov<:AbstractPDMat} @compat Tuple{Pri,Cov}

function posterior_canon(prior::MvNormal, ss::MvNormalKnownCovStats)
    invΣ0 = inv(prior.Σ)
    μ0 = prior.μ
    invΣp = pdadd(invΣ0, ss.invΣ, ss.tw)
    h = add!(invΣ0 * μ0, ss.invΣ * ss.sx)
	return MvNormalCanon(h, invΣp)
end

function posterior_canon(
    prior::MvNormalWithCov,
    G::Type{MvNormal},
    x::Matrix)

	μpri::Pri, Σ::Cov = prior
	posterior_canon(μpri, suffstats(MvNormalKnownCov{Cov}(Σ), x))
end

function posterior_canon(
    prior::MvNormalWithCov,
    G::Type{MvNormal},
    x::Matrix, w::Array{Float64})

    μpri::Pri, Σ::Cov = prior
    posterior_canon(μpri, suffstats(MvNormalKnownSigma{Cov}(Σ), x, w))
end

function posterior(
    prior::MvNormalWithCov,
    G::Type{MvNormal},
    x::Matrix)

    meanform(posterior_canon(prior, G, x))
end

function posterior(
    prior::MvNormalWithCov,
    G::Type{MvNormal},
    x::Matrix, w::Array{Float64})

    meanform(posterior_canon(prior, G, x, w))
end

function complete(
    G::Type{MvNormal},
    pri::MvNormalWithCov,
    μ::Vector{Float64})

    MvNormal(μ, pri[2]::Cov)
end

typealias MvNormalWithCovMat{Pri<:MvNormal} @compat Tuple{MvNormal, Matrix{Float64}}

function posterior_canon(pri::MvNormalWithCovMat, G::Type{MvNormal}, args...)
    μpri::MvNormal, Σmat::Matrix{Float64} = pri
    posterior_canon((μpri, PDMat(Σmat)), G, args...)
end

function posterior(pri::MvNormalWithCovMat, G::Type{MvNormal}, args...)
    μpri::MvNormal, Σmat::Matrix{Float64} = pri
    posterior((μpri, PDMat(Σmat)), G, args...)
end

function complete(G::Type{MvNormal}, pri::MvNormalWithCovMat, μ::Vector{Float64})
    MvNormal(μ, PDMat(pri[2]))
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

typealias MeanAndCovMat @compat Tuple{Vector{Float64}, Matrix{Float64}}
complete(G::Type{MvNormal}, pri::NormalInverseWishart, s::MeanAndCovMat) = MvNormal(s...)


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

complete(G::Type{MvNormal}, pri::NormalWishart, s::MeanAndCovMat) = MvNormal(s[1], inv(PDMat(s[2])))
