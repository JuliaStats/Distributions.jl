immutable Categorical <: DiscreteUnivariateDistribution
    prob::Vector{Float64}
    aliastable::AliasTable
    function Categorical{T <: Real}(p::Vector{T})
        length(p) > 1 || error("Categorical: there must be at least two categories")
        pv = T <: Float64 ? copy(p) : float64(p)
        all(pv .>= 0.) || error("Categorical: probabilities must be non-negative")
        sump = sum(pv); sump > 0. || error("Categorical: sum(p) > 0.")
        pv ./= sump
        new(pv, AliasTable(pv))
    end
end

Categorical(d::Integer) = Categorical(ones(d))

min(d::Categorical) = 1
max(d::Categorical) = length(d.prob)

function cdf(d::Categorical, x::Integer)
    if !insupport(d, x)
        error("$x is not in the support")
    else
        p = 0.0
        for i in 1:x
            p += d.prob[i]
        end
        return p
    end
end

entropy(d::Categorical) = NumericExtensions.entropy(d.prob)

function insupport(d::Categorical, x::Real)
    return isinteger(x) && 1 <= x <= length(d.prob) && d.prob[x] != 0.0
end

function kurtosis(d::Categorical)
    m = mean(d)
    s = 0.0
    for i in 1:length(d.prob)
        s += (i - m)^4 * d.prob[i]
    end
    return s / var(d)^2 - 3.0
end

function mean(d::Categorical)
    s = 0.0
    for i in 1:length(d.prob)
        s += i * d.prob[i]
    end
    return s
end

function median(d::Categorical)
    p, n = 0.0, length(d.prob)
    i = 0
    while p < 0.5 && i <= n
        i += 1
        p += d.prob[i]
    end
    return i
end

function mgf(d::Categorical, t::AbstractVector)
    s = 0.0
    for i in 1:length(d.prob)
        s += d.prob[i] * exp(t[i])
    end
    return s
end

function cf(d::Categorical, t::AbstractVector)
    s = 0.0 + 0.0im
    for i in 1:length(d.prob)
        s += d.prob[i] * exp(im * t[i])
    end
    return s
end

modes(d::Categorical) = [indmax(d.prob)]

pdf(d::Categorical, x::Real) = !insupport(d, x) ? 0.0 : d.prob[x]

rand(d::Categorical) = rand(d.aliastable)

function skewness(d::Categorical)
    m = mean(d)
    s = 0.0
    for i in 1:length(d.prob)
        s += (i - m)^3 * d.prob[i]
    end
    return s / std(d)^3
end

var(d::Categorical) = var(d, mean(d))

function var(d::Categorical, m::Number)
    s = 0.0
    for i in 1:length(d.prob)
        s += (i - m)^2 * d.prob[i]
    end
    return s
end

function fit_mle{T <: Real}(::Type{Categorical}, x::Array{T})
    # Counts for all categories
    return Categorical()
end



