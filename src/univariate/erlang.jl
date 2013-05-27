##############################################################################
#
# REFERENCES: "Statistical Distributions"
#
##############################################################################

immutable Erlang <: ContinuousUnivariateDistribution
    shape::Float64
    rate::Float64
end

Erlang(scale::Real) = Erlang(1.0, 1.0)
Erlang() = Erlang(1.0, 1.0)

insupport(d::Erlang, x::Number) = isreal(x) && isfinite(x) && 0.0 <= x

mean(d::Erlang) = d.scale * d.shape

function pdf(d::Erlang, x::Real)
    b, c = d.scale, d.shape
    return ((x / b)^(c - 1.0) * exp(-x / b)) / (b * gamma(c))
end

function rand(d::Erlang)
    b, c = d.scale, d.shape
    z = 0.0
    for i in 1:c
        z *= rand()
    end
    return -b * log(z)
end

var(d::Erlang) = d.scale^2 * d.shape
