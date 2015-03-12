immutable Laplace <: ContinuousUnivariateDistribution
    μ::Float64
    β::Float64

    function Laplace(μ::Real, β::Real)
        β > zero(β) || error("Laplace's scale must be positive")
        @compat new(Float64(μ), Float64(β))
    end

    @compat Laplace(μ::Real) = new(Float64(μ), 1.0)
    Laplace() = new(0.0, 1.0)
end

typealias Biexponential Laplace

@distr_support Laplace -Inf Inf


#### Parameters

location(d::Laplace) = d.μ
scale(d::Laplace) = d.β
params(d::Laplace) = (d.μ, d.β)


#### Statistics

mean(d::Laplace) = d.μ
median(d::Laplace) = d.μ
mode(d::Laplace) = d.μ

var(d::Laplace) = 2.0 * d.β^2
std(d::Laplace) = sqrt2 * d.β
skewness(d::Laplace) = 0.0
kurtosis(d::Laplace) = 3.0

entropy(d::Laplace) = log(2.0 * d.β) + 1.0


#### Evaluations

zval(d::Laplace, x::Float64) = (x - d.μ) / d.β
xval(d::Laplace, z::Float64) = d.μ + z * d.β

pdf(d::Laplace, x::Float64) = 0.5 * exp(-abs(zval(d, x))) / scale(d)
logpdf(d::Laplace, x::Float64) = - (abs(zval(d, x)) + log(2.0 * scale(d)))

cdf(d::Laplace, x::Float64) = (z = zval(d, x); z < 0.0 ? 0.5 * exp(z) : 1.0 - 0.5 * exp(-z))
ccdf(d::Laplace, x::Float64) = (z = zval(d, x); z > 0.0 ? 0.5 * exp(-z) : 1.0 - 0.5 * exp(z))
logcdf(d::Laplace, x::Float64) = (z = zval(d, x); z < 0.0 ? loghalf + z : loghalf + log2mexp(-z))
logccdf(d::Laplace, x::Float64) = (z = zval(d, x); z > 0.0 ? loghalf - z : loghalf + log2mexp(z))

quantile(d::Laplace, p::Float64) = p < 0.5 ? xval(d, log(2.0 * p)) : xval(d, -log(2.0 * (1.0 - p)))
cquantile(d::Laplace, p::Float64) = p > 0.5 ? xval(d, log(2.0 * (1.0 - p))) : xval(d, -log(2.0 * p))
invlogcdf(d::Laplace, lp::Float64) = lp < loghalf ? xval(d, logtwo + lp) : xval(d, -(logtwo + log1mexp(lp)))
invlogccdf(d::Laplace, lp::Float64) = lp > loghalf ? xval(d, logtwo + log1mexp(lp)) : xval(d, -(logtwo + lp))

function gradlogpdf(d::Laplace, x::Float64)
    μ, β = params(d)
    x == μ && error("Gradient is undefined at the location point")
    g = 1.0 / β
    x > μ ? -g : g
end

function mgf(d::Laplace, t::Real)
    st = d.β * t
    exp(t * d.μ) / ((1.0 - st) * (1.0 + st))
end
function cf(d::Laplace, t::Real)
    st = d.β * t
    cis(t * d.μ) / (1+st*st)
end


#### Sampling

function rand(d::Laplace) 
    z = randexp()
    rand(Bool) ? d.μ + d.β * z : d.μ - d.β * z 
end


#### Fitting

function fit_mle(::Type{Laplace}, x::Array)
    a = median(x)
    Laplace(a, mad(x, a))
end


