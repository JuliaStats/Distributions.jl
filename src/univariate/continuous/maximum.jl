"""
The maximum of n iid random variables with continuous univariate distribution
"""
struct Maximum{D<:ContinuousUnivariateDistribution} <: ContinuousUnivariateDistribution
    dist::D
    n::Int
    Maximum{D}(dist, n) where {D<:ContinuousUnivariateDistribution} = new{D}(dist, n)
end

function Maximum(dist::D, n::Integer; check_args::Bool=true) where {D <: ContinuousUnivariateDistribution}
    @check_args Maximum (n, n >= one(n))
    return Maximum{D}(dist, n)
end

rand(rng::AbstractRNG, d::Maximum{D}) where {D} = maximum([rand(rng, d.dist) for _ in 1:d.n])

#### Evaluation

cdf(d::Maximum{D}, x::Real) where {D} = cdf(d.dist, x)^d.n
pdf(d::Maximum{D}, x::Real) where {D} = d.n*pdf(d.dist, x)*cdf(d.dist, x)^(d.n-1)
logpdf(d::Maximum{D}, x::Real) where {D} = log(d.n)+logpdf(d.dist, x)+(d.n-1)*logcdf(d.dist, x)
quantile(d::Maximum{D}, q::Real) where {D} = quantile(d.dist, q^(1/d.n))
minimum(d::Maximum{D}) where {D} = minimum(d.dist)
maximum(d::Maximum{D}) where {D} = maximum(d.dist)
insupport(d::Maximum{D}, x::Real) where {D} = insupport(d.dist, x)