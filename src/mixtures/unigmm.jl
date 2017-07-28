# Univariate Gaussian Mixture Models

immutable UnivariateGMM{T<:Real} <: UnivariateMixture{Continuous,Normal}
    K::Int
    means::Vector{T}
    stds::Vector{T}
    prior::Categorical

    (::Type{UnivariateGMM{T}}){T}(ms::Vector{T}, ss::Vector{T}, pri::Categorical) = begin
        K = length(ms)
        length(ss) == K || throw(DimensionMismatch())
        ncategories(pri) == K ||
            error("The number of categories in pri should be equal to the number of components.")
        new{T}(K, ms, ss, pri)
    end
end

UnivariateGMM{T<:Real}(ms::Vector{T}, ss::Vector{T}, pri::Categorical) = UnivariateGMM{T}(ms, ss, pri)

@distr_support UnivariateGMM -Inf Inf

ncomponents(d::UnivariateGMM) = d.K

component(d::UnivariateGMM, k::Int) = Normal(d.means[k], d.stds[k])

probs(d::UnivariateGMM) = probs(d.prior)

mean(d::UnivariateGMM) = dot(d.means, probs(d))

rand(d::UnivariateGMM) = (k = rand(d.prior); d.means[k] + randn() * d.stds[k])

params(d::UnivariateGMM) = (d.means, d.stds, d.prior)

immutable UnivariateGMMSampler{T<:Real} <: Sampleable{Univariate,Continuous}
    means::Vector{T}
    stds::Vector{T}
    psampler::AliasTable
end

rand(s::UnivariateGMMSampler) = (k = rand(s.psampler); s.means[k] + randn() * s.stds[k])
sampler(d::UnivariateGMM) = UnivariateGMMSampler(d.means, d.stds, sampler(d.prior))
