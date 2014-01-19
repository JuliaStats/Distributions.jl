## Canonical Form of Normal distribution

immutable NormalCanon <: ContinuousUnivariateDistribution
    h::Float64       # σ^(-2) * μ
    prec::Float64    # σ^(-2)
    μ::Float64       # μ

    function NormalCanon(h::Float64, prec::Float64)
    	prec > 0. || throw(ArgumentError("prec should be positive."))
    	new(h, prec, h / prec)
    end

    NormalCanon(h::Real, prec::Real) = NormalCanon(float64(h), float64(prec))
    NormalCanon() = new(0.0, 1.0, 0.0)
end

## conversion between Normal and NormalCanon

Base.convert(::Type{Normal}, cf::NormalCanon) = Normal(cf.μ, 1.0 / sqrt(cf.prec))
Base.convert(::Type{NormalCanon}, d::Normal) = (J = 1.0 / abs2(σ); NormalCanon(J * d.μ, J))
canonform(d::Normal) = convert(NormalCanon, d)

## Basic properties

@continuous_distr_support NormalCanon -Inf Inf

mean(cf::NormalCanon) = cf.μ
median(cf::NormalCanon) = mean(cf)
mode(cf::NormalCanon) = mean(cf)
modes(cf::NormalCanon) = [mean(cf)]

skewness(cf::NormalCanon) = 0.0
kurtosis(cf::NormalCanon) = 0.0

var(cf::NormalCanon) = 1.0 / cf.prec
std(cf::NormalCanon) = sqrt(var(cf))

entropy(cf::NormalCanon) = 0.5 * (log2π + 1. - log(cf.prec))

# Evaluation

pdf(d::NormalCanon, x::Real) = (sqrt(d.prec) / √2π) * exp(-0.5 * d.prec * abs2(x - d.μ))
logpdf(d::NormalCanon, x::Real) = 0.5 * (log(d.prec) - log2π - d.prec * abs2(x - d.μ))

zval(d::NormalCanon, x::Real) = (x - d.μ) * sqrt(d.prec)
xval(d::NormalCanon, z::Real) = d.μ + z / sqrt(d.prec)

cdf(d::NormalCanon, x::Real) = Φ(zval(d,x))
ccdf(d::NormalCanon, x::Real) = Φc(zval(d,x))
logcdf(d::NormalCanon, x::Real) = logΦ(zval(d,x))
logccdf(d::NormalCanon, x::Real) = logΦc(zval(d,x))    

quantile(d::NormalCanon, p::Real) = xval(d, Φinv(p))
cquantile(d::NormalCanon, p::Real) = xval(d, -Φinv(p))
invlogcdf(d::NormalCanon, p::Real) = xval(d, logΦinv(p))
invlogccdf(d::NormalCanon, p::Real) = xval(d, -logΦinv(p))

## Sampling

rand(cf::NormalCanon) = cf.μ + randn() / sqrt(cf.prec)
rand!{T<:FloatingPoint}(cf::NormalCanon, r::Array{T}) = rand!(convert(Normal, cf), r)
