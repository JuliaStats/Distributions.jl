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
