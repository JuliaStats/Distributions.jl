##############################################################################
#
# REFERENCES: "Statistical Distributions"
#
##############################################################################

struct EmpiricalUnivariateDistribution{T<:Real} <: DiscreteUnivariateDistribution
    values::Vector{T}
    support::Vector{T}
    cdf::Function
    entropy::Void
    kurtosis::Void
    mean::T
    median::T
    modes::Void
    skewness::Void
    var::T
end

@distr_support EmpiricalUnivariateDistribution d.values[1] d.values[end]

function EmpiricalUnivariateDistribution(x::Vector{T}) where {T<:Real}
    sx = sort(x)
    EmpiricalUnivariateDistribution{T}(sx,
                                       unique(sx),
                                       ecdf(x),
                                       nothing,
                                       nothing,
                                       mean(x),
                                       median(x),
                                       nothing,
                                       nothing,
                                       var(x))
end

entropy(d::EmpiricalUnivariateDistribution) = d.entropy

kurtosis(d::EmpiricalUnivariateDistribution) = d.kurtosis

mean(d::EmpiricalUnivariateDistribution) = d.mean

median(d::EmpiricalUnivariateDistribution) = d.median

modes(d::EmpiricalUnivariateDistribution) = d.modes

skewness(d::EmpiricalUnivariateDistribution) = d.skewness

var(d::EmpiricalUnivariateDistribution) = d.var


### Evaluation

cdf(d::EmpiricalUnivariateDistribution, x::Real) = d.cdf(x)

function pdf(d::EmpiricalUnivariateDistribution{T}, x::Real) where {T<:Real}
    ## TODO: Create lookup table for discrete case
    one(T) / length(d.values)
end

# bisection step for quantile lookup
function bisect(f::Function, p::Real, values::Vector{<:Real},
                idx₁::Integer, idx₂::Integer)
    x₁ = values[idx₁]
    x₂ = values[idx₂]

    xₘ = (x₁ + x₂) / 2
    fₘ = f(xₘ)

    if fₘ > p
        idx₁, findfirst(x -> x > xₘ, values)
    else
        findlast(x -> x < xₘ, values), idx₂
    end
end

function quantile(d::EmpiricalUnivariateDistribution, p::Real)
    n = length(d.values)

    if n == 1 # trivial
        d.values[1]
    elseif n == 2 # easy choice
        p < 1 ? d.values[1] : d.values[2]
    else # bisect until p ∈ [cdf(x₁), cdf(x₂))
        i, j = bisect(d.cdf, p, d.values, 1, n)
        a, b = 1, n
        while i ≠ a || j ≠ b
            a, b = i, j
            i, j = bisect(d.cdf, p, d.values, i, j)
        end

        # the quantile is in one of the indexes a, a+1, ..., b
        interval = a:b
        idx = findlast(x -> d.cdf(x) ≤ p, d.values[interval])

        if idx ≠ 0 # case p = 0
            d.values[interval[idx]]
        else
            d.values[a]
        end
    end
end

rand(d::EmpiricalUnivariateDistribution) = quantile(d, rand())


### fit model

function fit_mle(::Type{EmpiricalUnivariateDistribution},
             x::Vector{T}) where T <: Real
    EmpiricalUnivariateDistribution(x)
end
