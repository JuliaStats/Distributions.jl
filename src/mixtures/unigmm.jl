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
    koffset = rand(rng, d.prior) - 1
    μ = d.means[begin+koffset]
    σ = d.stds[begin+koffset]
    return muladd(randn(rng, float(Base.promote_typeof(μ, σ))), σ, μ)
end

params(d::UnivariateGMM) = (d.means, d.stds, d.prior)

struct UnivariateGMMSampler{VT1<:AbstractVector{<:Real},VT2<:AbstractVector{<:Real}} <: Sampleable{Univariate,Continuous}
    means::VT1
    stds::VT2
    psampler::AliasTable
end

function rand(rng::AbstractRNG, s::UnivariateGMMSampler)
    koffset = rand(rng, s.psampler) - 1
    μ = d.means[begin+koffset]
    σ = d.stds[begin+koffset]
    return muladd(randn(rng, float(Base.promote_typeof(μ, σ))), σ, μ)
end
function rand!(rng::AbstractRNG, s::UnivariateGMMSampler, x::AbstractArray{<:Real})
    (; means, stds, psampler) = s
    randn!(rng, x)
    for i in eachindex(x)
        koffset = rand(rng, psampler) - 1
        x[i] = muladd(x[i], stds[begin+koffset], means[begin+koffset])
    end
    return x
end

sampler(d::UnivariateGMM) = UnivariateGMMSampler(d.means, d.stds, sampler(d.prior))
