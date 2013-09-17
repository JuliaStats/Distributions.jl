##############################################################################
#
# REFERENCES: "Statistical Distributions"
#
##############################################################################

immutable EmpiricalUnivariateDistribution <: ContinuousUnivariateDistribution
	values::Vector{Float64}
	cdf::Function
	entropy::Float64
	kurtosis::Float64
	mean::Float64
	median::Float64
	modes::Vector{Float64}
	skewness::Float64
	var::Float64
end

function EmpiricalUnivariateDistribution(x::Vector)
	EmpiricalUnivariateDistribution(sort(x),
	                                ecdf(x),
	                                NaN,
	                                NaN,
	                                mean(x),
	                                median(x),
	                                Float64[],
	                                NaN,
	                                var(x))
end

cdf(d::EmpiricalUnivariateDistribution, q::Real) = d.cdf(q)

entropy(d::EmpiricalUnivariateDistribution) = d.entropy

function insupport(d::EmpiricalUnivariateDistribution, x::Number)
    contains(d.values, x)
end

kurtosis(d::EmpiricalUnivariateDistribution) = d.kurtosis

mean(d::EmpiricalUnivariateDistribution) = d.mean

median(d::EmpiricalUnivariateDistribution) = d.median

modes(d::EmpiricalUnivariateDistribution) = Float64[]

function pdf(d::EmpiricalUnivariateDistribution, x::Real)
    ## TODO: Create lookup table for discrete case
    1.0 / length(d.values)
end

function quantile(d::EmpiricalUnivariateDistribution, p::Real)
    n = length(d.values)
    index = ifloor(p * n) + 1
    index > n ? d.values[n] : d.values[index]
end

function rand(d::EmpiricalUnivariateDistribution)
    d.values[rand(1:length(d.values))]
end

skewness(d::EmpiricalUnivariateDistribution) = NaN

var(d::EmpiricalUnivariateDistribution) = d.var

function fit_mle{T <: Real}(::Type{EmpiricalUnivariateDistribution},
	                    x::Vector{T})
	EmpiricalUnivariateDistribution(x)
end
