##############################################################################
#
# REFERENCES: "Statistical Distributions"
#
##############################################################################

immutable EmpiricalUnivariateDistribution <: ContinuousUnivariateDistribution
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

cdf(d::EmpiricalUnivariateDistribution, q::Real) = d.cdf(q)

entropy(d::EmpiricalUnivariateDistribution) = d.entropy

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

### handling support

insupport(d::EmpiricalUnivariateDistribution, x::Number) = contains(d.support, x)

isupperbounded(::Union(EmpiricalUnivariateDistribution, Type{EmpiricalUnivariateDistribution})) = true
islowerbounded(::Union(EmpiricalUnivariateDistribution, Type{EmpiricalUnivariateDistribution})) = true
isbounded(::Union(EmpiricalUnivariateDistribution, Type{EmpiricalUnivariateDistribution})) = true

hasfinitesupport(d::Union(EmpiricalUnivariateDistribution, Type{EmpiricalUnivariateDistribution})) = true
min(d::EmpiricalUnivariateDistribution) = min(d.values[1])
max(d::EmpiricalUnivariateDistribution) = max(d.values[end])
support(d::EmpiricalUnivariateDistribution) = d.support

### fit model

function fit_mle{T <: Real}(::Type{EmpiricalUnivariateDistribution},
	                    x::Vector{T})
	EmpiricalUnivariateDistribution(x)
end