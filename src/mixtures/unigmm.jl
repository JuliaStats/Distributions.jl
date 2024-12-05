# Univariate Gaussian Mixture Models

struct UnivariateGMM{VT1<:AbstractVector{<:Real},VT2<:AbstractVector{<:Real},C<:Categorical} <: UnivariateMixture{Continuous,Normal}
    K::Int
    means::VT1
    stds::VT2
    prior::C

    function UnivariateGMM(ms::VT1, ss::VT2, pri::C) where {VT1<:AbstractVector{<:Real},VT2<:AbstractVector{<:Real},C<:Categorical}
        K = length(ms)
        length(ss) == K || throw(DimensionMismatch())
        ncategories(pri) == K ||
            error("The number of categories in pri should be equal to the number of components.")
        new{VT1,VT2,C}(K, ms, ss, pri)
    end
end

@distr_support UnivariateGMM -Inf Inf

ncomponents(d::UnivariateGMM) = d.K

component(d::UnivariateGMM, k::Int) = Normal(d.means[k], d.stds[k])

probs(d::UnivariateGMM) = probs(d.prior)

mean(d::UnivariateGMM) = dot(d.means, probs(d))

function rand(rng::AbstractRNG, d::UnivariateGMM)
    k = rand(rng, d.prior)
    μ = d.means[k]
    σ = d.stds[k]
    return muladd(randn(rng, float(Base.promote_typeof(μ, σ))), σ, μ)
end

params(d::UnivariateGMM) = (d.means, d.stds, d.prior)

struct UnivariateGMMSampler{VT1<:AbstractVector{<:Real},VT2<:AbstractVector{<:Real}} <: Sampleable{Univariate,Continuous}
    means::VT1
    stds::VT2
    psampler::AliasTable
end

function rand(rng::AbstractRNG, s::UnivariateGMMSampler)
    k = rand(rng, s.psampler)
    μ = d.means[k]
    σ = d.stds[k]
    return muladd(randn(rng, float(Base.promote_typeof(μ, σ))), σ, μ)
end
function rand!(rng::AbstractRNG, s::UnivariateGMMSampler, x::AbstractArray{<:Real})
    psampler = s.psampler
    means = s.means
    stds = s.stds
    randn!(rng, x)
    for i in eachindex(x)
        k = rand(rng, psampler)
        x[i] = muladd(x[i], stds[k], means[k])
    end
    return x
end

sampler(d::UnivariateGMM) = UnivariateGMMSampler(d.means, d.stds, sampler(d.prior))
