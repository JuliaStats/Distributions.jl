# Bernoulli distribution


#### Type and Constructor

immutable Bernoulli <: DiscreteUnivariateDistribution
    p::Float64

    function Bernoulli(p::Float64)
        0.0 <= p <= 1.0 || error("p must be in [0, 1].")
        new(p)
    end

    @compat Bernoulli(p::Real) = Bernoulli(Float64(p))
    Bernoulli() = new(0.5)
end

@distr_support Bernoulli 0 1


#### Parameters

succprob(d::Bernoulli) = d.p
failprob(d::Bernoulli) = 1.0 - d.p

params(d::Bernoulli) = (d.p,)


#### Properties

mean(d::Bernoulli) = succprob(d)
var(d::Bernoulli) =  succprob(d) * failprob(d)
skewness(d::Bernoulli) = (p0 = failprob(d); p1 = succprob(d); (p0 - p1) / sqrt(p0 * p1))
kurtosis(d::Bernoulli) = 1.0 / var(d) - 6.0


mode(d::Bernoulli) = ifelse(succprob(d) > 0.5, 1, 0)

function modes(d::Bernoulli)
    p = succprob(d)
    p < 0.5 ? [0] :
    p > 0.5 ? [1] : [0, 1]
end

function median(d::Bernoulli)
    p = succprob(d)
    p < 0.5 ? 0.0 :
    p > 0.5 ? 1.0 : 0.5
end

function entropy(d::Bernoulli) 
    p0 = failprob(d)
    p1 = succprob(d)
    (p0 == 0.0 || p0 == 1.0) ? 0.0 : -(p0 * log(p0) + p1 * log(p1))
end

#### Evaluation

pdf(d::Bernoulli, x::Bool) = x ? succprob(d) : failprob(d)
pdf(d::Bernoulli, x::Int) = x == 0 ? failprob(d) : 
                            x == 1 ? succprob(d) : 0.0

pdf(d::Bernoulli) = Float64[failprob(d), succprob(d)]

cdf(d::Bernoulli, x::Bool) = x ? failprob(d) : 1.0
cdf(d::Bernoulli, x::Int) = x < 0 ? 0.0 :
                            x < 1 ? failprob(d) : 1.0

ccdf(d::Bernoulli, x::Bool) = x ? succprob(d) : 1.0
ccdf(d::Bernoulli, x::Int) = x < 0 ? 1.0 :
                             x < 1 ? succprob(d) : 0.0

quantile(d::Bernoulli, p::Float64) = 0.0 <= p <= 1.0 ? (p <= failprob(d) ? 0 : 1) : NaN
cquantile(d::Bernoulli, p::Float64) = 0.0 <= p <= 1.0 ? (p >= succprob(d) ? 0 : 1) : NaN

mgf(d::Bernoulli, t::Real) = failprob(d) + succprob(d) * exp(t)
cf(d::Bernoulli, t::Real) = failprob(d) + succprob(d) * cis(t)


#### Sampling

rand(d::Bernoulli) = round(Int, rand() <= succprob(d))


#### MLE fitting

immutable BernoulliStats <: SufficientStats
    cnt0::Float64
    cnt1::Float64

    @compat BernoulliStats(c0::Real, c1::Real) = new(Float64(c0), Float64(c1))
end

fit_mle(::Type{Bernoulli}, ss::BernoulliStats) = Bernoulli(ss.cnt1 / (ss.cnt0 + ss.cnt1))

function suffstats{T<:Integer}(::Type{Bernoulli}, x::AbstractArray{T})
    n = length(x)
    c0 = c1 = 0
    for i = 1:n
        @inbounds xi = x[i]
        if xi == 0
            c0 += 1
        elseif xi == 1
            c1 += 1
        else
            throw(DomainError())
        end
    end
    BernoulliStats(c0, c1)
end

function suffstats{T<:Integer}(::Type{Bernoulli}, x::AbstractArray{T}, w::AbstractArray{Float64})
    n = length(x)
    length(w) == n || throw(DimensionMismatch("Inconsistent argument dimensions."))
    c0 = c1 = 0.0
    for i = 1:n
        @inbounds xi = x[i]
        @inbounds wi = w[i]
        if xi == 0
            c0 += wi
        elseif xi == 1
            c1 += wi
        else
            throw(DomainError())
        end
    end
    BernoulliStats(c0, c1)
end


