immutable Geometric <: DiscreteUnivariateDistribution
    prob::Float64
    function Geometric(p::Real)
        zero(p) < p < one(p) || error("prob must be in (0, 1)")
    	new(float64(p))
    end
end

Geometric() = Geometric(0.5) # Flips of a fair coin

## Support
insupport(::Geometric, x::Real) = isinteger(x) && x >= zero(x)
insupport(::Type{Geometric}, x::Real) = isinteger(x) && x >= zero(x)

isupperbounded(d::Union(Geometric, Type{Geometric})) = false
islowerbounded(d::Union(Geometric, Type{Geometric})) = true
isbounded(d::Union(Geometric, Type{Geometric})) = false

minimum(d::Union(Geometric, Type{Geometric})) = 0
maximum(d::Geometric) = Inf

## Properties
mean(d::Geometric) = (1.0 - d.prob) / d.prob

median(d::Geometric) = -fld(0.6931471805599453,log1p(-d.prob)) - 1.0

mode(d::Geometric) = 0
modes(d::Geometric) = [0]

var(d::Geometric) = (1.0 - d.prob) / d.prob^2
skewness(d::Geometric) = (2.0 - d.prob) / sqrt(1.0 - d.prob)
kurtosis(d::Geometric) = 6.0 + d.prob^2 / (1.0 - d.prob)

entropy(d::Geometric) = (-xlogx(1.0 - d.prob) - xlogx(d.prob)) / d.prob

## Functions
pdf(d::Geometric, x::Real) = insupport(d,x) ? d.prob*exp(log1p(-d.prob)*x) : 0.0
logpdf(d::Geometric, x::Real) = insupport(d,x) ? log(d.prob) + log1p(-d.prob)*x : -Inf

cdf(d::Geometric, q::Real) = q < zero(q) ? 0.0 : -expm1(log1p(-d.prob) * (floor(q) + 1.0))
ccdf(d::Geometric, q::Real) =  q < zero(q) ? 1.0 : exp(log1p(-d.prob) * (floor(q) + 1.0))
logcdf(d::Geometric, q::Real) = q < zero(q) ? -Inf : log1mexp(log1p(-d.prob) * (floor(q) + 1.0))
logccdf(d::Geometric, q::Real) =  q < zero(q) ? 0.0 : log1p(-d.prob) * (floor(q) + 1.0)

quantile(d::Geometric, p::Real) = invlogccdf(d,log1p(-p))
cquantile(d::Geometric, p::Real) = invlogccdf(d,log(p))
invlogcdf(d::Geometric, lp::Real) = invlogccdf(d,log1mexp(lp))

function invlogccdf(d::Geometric, lp::Real) 
    if (lp > zero(lp)) || isnan(lp)
        return NaN
    elseif isinf(lp)
        return Inf
    elseif lp == zero(lp)
        return 0.0
    end
    max(ceil(lp/log1p(-d.prob))-1.0,0.0)
end

function mgf(d::Geometric, t::Real)
    p = d.prob
    if t >= -log(1.0 - p)
        error("MGF does not exist for all t")
    end
    (p * exp(t)) / (1.0 - (1.0 - p) * exp(t))
end

function cf(d::Geometric, t::Real)
    p = d.prob
    (p * exp(im * t)) / (1.0 - (1.0 - p) * exp(im * t))
end

## Sampling
function rand(d::Geometric)
    e = Base.Random.randmtzig_exprnd()
    floor(-e/log1p(-d.prob))
end

## Fitting
immutable GeometricStats <: SufficientStats
    sx::Float64
    tw::Float64

    GeometricStats(sx::Real, tw::Real) = new(float64(sx), float64(tw))
end

suffstats{T<:Integer}(::Type{Geometric}, x::Array{T}) = GeometricStats(sum(x), length(x))

function suffstats{T<:Integer}(::Type{Geometric}, x::Array{T}, w::Array{Float64})
    n = length(x)
    if length(w) != n
        throw(ArgumentError("Inconsistent argument dimensions."))
    end
    sx = 0.
    tw = 0.
    for i = 1:n
        wi = w[i]
        sx += wi * x[i]
        tw += wi
    end
    GeometricStats(sx, tw)
end

fit_mle(::Type{Geometric}, ss::GeometricStats) = Geometric(1.0 / (ss.sx / ss.tw + 1.0))

