##############################################################################
#
# REFERENCES: "Statistical Distributions"
#
##############################################################################

struct EmpiricalUnivariateDistribution <: ContinuousUnivariateDistribution
    values::Vector{Float64}
    support::Vector{Float64}
    cdf::Function
    entropy::Float64
    kurtosis::Float64
    mean::Float64
    median::Float64
    modes::Vector{Float64}
    skewness::Float64
    var::Float64
end

@distr_support EmpiricalUnivariateDistribution d.values[1] d.values[end]

function EmpiricalUnivariateDistribution(x::Vector)
    sx = sort(x)
    EmpiricalUnivariateDistribution(sx,
                                    unique(sx),
                                    ecdf(x),
                                    NaN,
                                    NaN,
                                    mean(x),
                                    median(x),
                                    Float64[],
                                    NaN,
                                    var(x))
end

entropy(d::EmpiricalUnivariateDistribution) = d.entropy

kurtosis(d::EmpiricalUnivariateDistribution) = d.kurtosis

mean(d::EmpiricalUnivariateDistribution) = d.mean

median(d::EmpiricalUnivariateDistribution) = d.median

modes(d::EmpiricalUnivariateDistribution) = Float64[]

skewness(d::EmpiricalUnivariateDistribution) = NaN

var(d::EmpiricalUnivariateDistribution) = d.var


### Evaluation

cdf(d::EmpiricalUnivariateDistribution, x::Float64) = d.cdf(x)

function pdf(d::EmpiricalUnivariateDistribution, x::Float64)
    ## TODO: Create lookup table for discrete case
    1.0 / length(d.values)
end

function quantile(d::EmpiricalUnivariateDistribution, p::Float64)
    n = length(d.values)
    index = floor(Int,p * n) + 1
    index > n ? d.values[n] : d.values[index]
end

function rand(d::EmpiricalUnivariateDistribution)
    d.values[rand(1:length(d.values))]
end


### fit model

function fit_mle(::Type{EmpiricalUnivariateDistribution},
             x::Vector{T}) where T <: Real
    EmpiricalUnivariateDistribution(x)
end
