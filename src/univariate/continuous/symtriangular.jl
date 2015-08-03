immutable SymTriangularDist <: ContinuousUnivariateDistribution
    μ::Float64
    σ::Float64

    function SymTriangularDist(μ::Real, σ::Real)
        @check_args(SymTriangularDist, σ > zero(σ))
        new(μ, σ)
    end
    SymTriangularDist(μ::Real) = new(μ, 1.0)
    SymTriangularDist() = new(0.0, 1.0)
end

@distr_support SymTriangularDist d.μ - d.σ d.μ + d.σ


#### Parameters

location(d::SymTriangularDist) = d.μ
scale(d::SymTriangularDist) = d.σ

params(d::SymTriangularDist) = (d.μ, d.σ)


#### Statistics

mean(d::SymTriangularDist) = d.μ
median(d::SymTriangularDist) = d.μ
mode(d::SymTriangularDist) = d.μ

var(d::SymTriangularDist) = d.σ^2 / 6.0
skewness(d::SymTriangularDist) = 0.0
kurtosis(d::SymTriangularDist) = -0.6

entropy(d::SymTriangularDist) = 0.5 + log(d.σ)


#### Evaluation

zval(d::SymTriangularDist, x::Float64) = (x - d.μ) / d.σ
xval(d::SymTriangularDist, z::Float64) = d.μ + z * d.σ


pdf(d::SymTriangularDist, x::Float64) = insupport(d, x) ? (1.0 - abs(zval(d, x))) / scale(d) : 0.0

logpdf(d::SymTriangularDist, x::Float64) = insupport(d, x) ? log((1.0 - abs(zval(d, x))) / scale(d)) : -Inf

function cdf(d::SymTriangularDist, x::Float64)
    (μ, σ) = params(d)
    x <= μ - σ ? 0.0 :
    x <= μ ? 0.5 * (1.0 + zval(d, x))^2 :
    x < μ + σ ? 1.0 - 0.5 * (1.0 - zval(d, x))^2 : 1.0
end

function ccdf(d::SymTriangularDist, x::Float64)
    (μ, σ) = params(d)
    x <= μ - σ ? 1.0 :
    x <= μ ? 1.0 - 0.5 * (1.0 + zval(d, x))^2 :
    x < μ + σ ? 0.5 * (1.0 - zval(d, x))^2 : 0.0
end

function logcdf(d::SymTriangularDist, x::Float64)
    (μ, σ) = params(d)
    x <= μ - σ ? -Inf :
    x <= μ ? loghalf + 2.0 * log1p(zval(d, x)) :
    x < μ + σ ? log1p(-0.5 * (1.0 - zval(d, x))^2) : 0.0
end

function logccdf(d::SymTriangularDist, x::Float64)
    (μ, σ) = params(d)
    x <= μ - σ ? 0.0 :
    x <= μ ? log1p(-0.5 * (1.0 + zval(d, x))^2) :
    x < μ + σ ? loghalf + 2.0 * log1p(-zval(d, x)) : -Inf
end

quantile(d::SymTriangularDist, p::Float64) = p < 0.5 ? xval(d, sqrt(2.0 * p) - 1.0) :
                                                       xval(d, 1.0 - sqrt(2.0 * (1.0 - p)))

cquantile(d::SymTriangularDist, p::Float64) = p > 0.5 ? xval(d, sqrt(2.0 * (1.0-p)) - 1.0) :
                                                        xval(d, 1.0 - sqrt(2.0 * p))

invlogcdf(d::SymTriangularDist, lp::Float64) = lp < loghalf ? xval(d, expm1(0.5*(lp - loghalf))) :
                                                              xval(d, 1.0 - sqrt(-2.0 * expm1(lp)))

invlogccdf(d::SymTriangularDist, lp::Float64) = lp > loghalf ? xval(d, sqrt(-2.0 * expm1(lp)) - 1.0) :
                                                               xval(d, -(expm1(0.5 * (lp - loghalf))))


function mgf(d::SymTriangularDist, t::Float64)
    (μ, σ) = params(d)
    a = σ * t
    a == zero(a) && return one(a)
    4.0 * exp(μ * t) * (sinh(0.5 * a) / a)^2
end

function cf(d::SymTriangularDist, t::Float64)
    (μ, σ) = params(d)
    a = σ * t
    a == zero(a) && return complex(one(a))
    4.0 * cis(μ * t) * (sin(0.5 * a) / a)^2
end


#### Sampling

rand(d::SymTriangularDist) = xval(d, rand() - rand())
