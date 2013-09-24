##############################################################################
#
# REFERENCES: "Statistical Distributions"
#
##############################################################################

immutable Erlang <: ContinuousUnivariateDistribution
    shape::Int
    scale::Float64
    nested_gamma::Gamma

    function Erlang(shape::Real, scale::Real)
        isinteger(shape) || error("Erlang shape parameter must be an integer")
        new(int(shape), float64(scale), Gamma(shape, scale))
    end

    Erlang(scale::Real) = Erlang(scale, 1.0)
    Erlang() = Erlang(1, 1.0)
end

@continuous_distr_support Erlang 0.0 Inf

cdf(d::Erlang, x::Real) = cdf(d.nested_gamma, x)

entropy(d::Erlang) = entropy(d.nested_gamma)

kurtosis(d::Erlang) = kurtosis(d.nested_gamma)

mean(d::Erlang) = d.shape * d.scale

median(d::Erlang) = median(d.nested_gamma)

mgf(d::Erlang, t::Real) = mgf(d.nested_gamma, t)
cf(d::Erlang, t::Real) = cf(d.nested_gamma, t)

mode(d::Erlang) = mode(d.nested_gamma)
modes(d::Erlang) = modes(d.nested_gamma)

function pdf(d::Erlang, x::Real)
    b, c = d.scale, d.shape
    ((x / b)^(c - 1.0) * exp(-x / b)) / (b * gamma(c))
end

quantile(d::Erlang, p::Real) = quantile(d.nested_gamma, p)

function rand(d::Erlang)
    b, c = d.scale, d.shape
    z = 1.0
    for i in 1:c
        z *= rand()
    end
    -b * log(z)
end

skewness(d::Erlang) = skewness(d.nested_gamma)

var(d::Erlang) = d.scale^2 * d.shape
