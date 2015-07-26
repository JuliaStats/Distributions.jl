## Canonical Form of Normal distribution

immutable NormalCanon <: ContinuousUnivariateDistribution
    h::Float64       # σ^(-2) * μ
    prec::Float64    # σ^(-2)
    μ::Float64       # μ

    function NormalCanon(h::Float64, prec::Float64)
    	prec > 0. || throw(ArgumentError("prec should be positive."))
    	new(h, prec, h / prec)
    end

    @compat NormalCanon(h::Real, prec::Real) = NormalCanon(Float64(h), Float64(prec))
    NormalCanon() = new(0.0, 1.0, 0.0)
end

@distr_support NormalCanon -Inf Inf

## conversion between Normal and NormalCanon

Base.convert(::Type{Normal}, cf::NormalCanon) = Normal(cf.μ, 1.0 / sqrt(cf.prec))
Base.convert(::Type{NormalCanon}, d::Normal) = (J = 1.0 / abs2(σ); NormalCanon(J * d.μ, J))
canonform(d::Normal) = convert(NormalCanon, d)


#### Parameters

params(d::NormalCanon) = (d.h, d.prec)


#### Statistics

mean(cf::NormalCanon) = cf.μ
median(cf::NormalCanon) = mean(cf)
mode(cf::NormalCanon) = mean(cf)

skewness(cf::NormalCanon) = 0.0
kurtosis(cf::NormalCanon) = 0.0

var(cf::NormalCanon) = 1.0 / cf.prec
std(cf::NormalCanon) = sqrt(var(cf))

entropy(cf::NormalCanon) = 0.5 * (log2π + 1. - log(cf.prec))


#### Evaluation

pdf(d::NormalCanon, x::Float64) = (sqrt(d.prec) / sqrt2π) * exp(-0.5 * d.prec * abs2(x - d.μ))
logpdf(d::NormalCanon, x::Float64) = 0.5 * (log(d.prec) - log2π - d.prec * abs2(x - d.μ))

zval(d::NormalCanon, x::Float64) = (x - d.μ) * sqrt(d.prec)
xval(d::NormalCanon, z::Float64) = d.μ + z / sqrt(d.prec)

cdf(d::NormalCanon, x::Float64) = normcdf(zval(d,x))
ccdf(d::NormalCanon, x::Float64) = normccdf(zval(d,x))
logcdf(d::NormalCanon, x::Float64) = normlogcdf(zval(d,x))
logccdf(d::NormalCanon, x::Float64) = normlogccdf(zval(d,x))

quantile(d::NormalCanon, p::Float64) = xval(d, norminvcdf(p))
cquantile(d::NormalCanon, p::Float64) = xval(d, norminvccdf(p))
invlogcdf(d::NormalCanon, lp::Float64) = xval(d, norminvlogcdf(lp))
invlogccdf(d::NormalCanon, lp::Float64) = xval(d, norminvlogccdf(lp))


#### Sampling

rand(cf::NormalCanon) = cf.μ + randn() / sqrt(cf.prec)
rand!{T<:Real}(cf::NormalCanon, r::AbstractArray{T}) = rand!(convert(Normal, cf), r)
