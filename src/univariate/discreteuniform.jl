immutable DiscreteUniform <: DiscreteUnivariateDistribution
    a::Int
    b::Int
    function DiscreteUniform(a::Real, b::Real)
        if a < b
            new(int(a), int(b))
        else
            error("a must be less than b")
        end
    end
end

DiscreteUniform(b::Integer) = DiscreteUniform(0, b)
DiscreteUniform() = DiscreteUniform(0, 1)

entropy(d::DiscreteUniform) = log(d.b - d.a + 1.0)

insupport(d::DiscreteUniform, x::Number) = isinteger(x) && d.a <= x <= d.b

function kurtosis(d::DiscreteUniform)
    n = d.b - d.a + 1.0
    return -(6.0 / 5.0) * (n^2 + 1.0) / (n^2 - 1.0)
end

mean(d::DiscreteUniform) = (d.a + d.b) / 2.0

median(d::DiscreteUniform) = (d.a + d.b) / 2.0

modes(d::DiscreteUniform) = [d.a:d.b]

function pdf(d::DiscreteUniform, x::Real)
    if insupport(d, x)
        return (1.0 / (d.b - d.a + 1.))
    else
        return 0.0
    end
end

function quantile(d::DiscreteUniform, k::Real)
    if k < d.a
        return 0.0
    elseif <= d.b
        return (floor(k) - d.a + 1.) / (d.b - d.a + 1.)
    else
        return 1.0
    end
end

rand(d::DiscreteUniform) = d.a + rand(0:(d.b - d.a))

skewness(d::DiscreteUniform) = 0.0

var(d::DiscreteUniform) = ((d.b - d.a + 1.0)^2 - 1.0) / 12.0
