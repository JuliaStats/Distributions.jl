struct EmpiricalUnivariateDistribution <: DiscreteUnivariateDistribution
    values::Vector{Float64}
    cdf::Function
end

@distr_support EmpiricalUnivariateDistribution d.values[1] d.values[end]

EmpiricalUnivariateDistribution(x::Vector) = EmpiricalUnivariateDistribution(sort(x), ecdf(x))

for f in (:entropy, :mean, :var, :skewness, :kurtosis)
    @eval ($f)(d::EmpiricalUnivariateDistribution) = ($f)(d.values)
end

function median(d::DiscreteUnivariateDistribution)
    v = d.values
    n = length(v)
    return (v[(n + 1) >> 1] + v[(n + 2) >> 1]) / 2
end

### Evaluation

cdf(d::EmpiricalUnivariateDistribution, x::Real) = d.cdf(x)

pdf(d::EmpiricalUnivariateDistribution, x::Real) = mean(t -> t == x, d.values)

quantile(d::EmpiricalUnivariateDistribution, p::Real) = quantile(d.values, p)

function rand(d::EmpiricalUnivariateDistribution)
    d.values[rand(1:length(d.values))]
end
