immutable Categorical <: DiscreteUnivariateDistribution
    prob::Vector{Float64}
    drawtable::DiscreteDistributionTable
    function Categorical{T <: Real}(p::Vector{T})
        if length(p) <= 1
            error("Categorical: there must be at least two categories")
        end
        sump = 0.0
        for i in 1:length(p)
            if p[i] < 0.0
                error("Categorical: probabilities must be non-negative")
            end
            sump += p[i]
        end
        for i in 1:length(p)
            p[i] /= sump
        end
        new(p, DiscreteDistributionTable(p))
    end
end

function Categorical(d::Integer)
    if d <= 1
        error("d must be greater than 1")
    end
    prob = Array(Float64, d)
    fill!(prob, 1.0 / d)
    Categorical(prob)
end

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

entropy(d::Categorical) = pventropy(d.prob)

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

rand(d::Categorical) = draw(d.drawtable)

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

function fit{T <: Real}(::Type{Categorical}, x::Array{T})
    # Counts for all categories
    return Categorical()
end
