immutable Multinomial <: DiscreteMultivariateDistribution
    n::Int
    prob::Vector{Float64}
    drawtable::DiscreteDistributionTable
    function Multinomial{T <: Real}(n::Integer, p::Vector{T})
        if n <= 0
            error("Multinomial: n must be positive")
        end
        sump = 0.0
        for i in 1:length(p)
            if p[i] < 0.0
                error("Multinomial: probabilities must be non-negative")
            end
            sump += p[i]
        end
        for i in 1:length(p)
            p[i] /= sump
        end
        new(int(n), p, DiscreteDistributionTable(p))
    end
end

function Multinomial(n::Integer, d::Integer)
    if d <= 1
        error("d must be greater than 1")
    end
    Multinomial(n, ones(d) / d)
end

Multinomial(d::Integer) = Multinomial(1, d)

function insupport{T <: Real}(d::Multinomial, x::Vector{T})
    n = length(x)
    if length(d.prob) != n
        return false
    end
    s = 0.0
    for i in 1:n
        if x[i] < 0.0 || !isinteger(x[i])
            return false
        end
        s += x[i]
    end
    if abs(s - d.n) > 10e-8
        return false
    end
    return true
end

mean(d::Multinomial) = d.n .* d.prob

pdf{T <: Real}(d::Multinomial, x::Vector{T}) = exp(logpdf(d, x))

function logpdf{T <: Real}(d::Multinomial, x::Vector{T})
    if !insupport(d, x)
        return -Inf
    else
        return lgamma(d.n + 1.0) - sum(lgamma(x + 1.0)) + sum(x .* log(d.prob))
    end
end

function rand(d::Multinomial)
    s = zeros(Int, length(d.prob))
    psum = 1.0
    for index in 1:d.n
        i = draw(d.drawtable)
        s[i] += 1
    end
    return s
end

function var(d::Multinomial)
    s = 0.0
    for i in 1:length(d.prob)
        s += d.n * d.prob[i] .* (1.0 - d.prob[i])
    end
    return s
end
