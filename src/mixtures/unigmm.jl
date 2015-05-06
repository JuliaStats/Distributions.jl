# Univariate Gaussian Mixture Models

immutable UnivariateGMM <: UnivariateMixture{Continuous,Normal}
    K::Int
    means::Vector{Float64}
    stds::Vector{Float64}
    prior::Categorical

    function UnivariateGMM(ms::Vector{Float64}, ss::Vector{Float64}, pri::Categorical)
        K = length(ms)
        length(ss) == K || throw(DimensionMismatch())
        ncategories(pri) == K ||
            error("The number of categories in pri should be equal to the number of components.")
        new(K, ms, ss, pri)
    end
end

ncomponents(d::UnivariateGMM) = d.K

component(d::UnivariateGMM, k::Int) = Normal(d.means[k], d.stds[k])

probs(d::UnivariateGMM) = probs(d.prior)

mean(d::UnivariateGMM) = dot(d.means, probs(d))

rand(d::UnivariateGMM) = (k = rand(d.prior); d.means[k] + randn() * d.stds[k])

immutable UnivariateGMMSampler <: Sampleable{Univariate,Continuous}
    means::Vector{Float64}
    stds::Vector{Float64}
    psampler::AliasTable
end

rand(s::UnivariateGMMSampler) = (k = rand(s.psampler); s.means[k] + randn() * s.stds[k])
sampler(d::UnivariateGMM) = UnivariateGMMSampler(d.means, d.stds, sampler(d.prior))
