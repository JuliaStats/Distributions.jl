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

rand(d::UnivariateGMM) = (k = rand(d.prior); d.means[k] + randn() * d.stds[k])

rand(rng::AbstractRNG, d::UnivariateGMM) =
    (k = rand(rng, d.prior); d.means[k] + randn(rng) * d.stds[k])

params(d::UnivariateGMM) = (d.means, d.stds, d.prior)

struct UnivariateGMMSampler{VT1<:AbstractVector{<:Real},VT2<:AbstractVector{<:Real}} <: Sampleable{Univariate,Continuous}
    means::VT1
    stds::VT2
    psampler::AliasTable
end

rand(rng::AbstractRNG, s::UnivariateGMMSampler) =
    (k = rand(rng, s.psampler); s.means[k] + randn(rng) * s.stds[k])
sampler(d::UnivariateGMM) = UnivariateGMMSampler(d.means, d.stds, sampler(d.prior))
