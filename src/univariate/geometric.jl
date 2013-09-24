immutable Geometric <: DiscreteUnivariateDistribution
    prob::Float64
    function Geometric(p::Real)
        zero(p) < p < one(p) || error("prob must be in (0, 1)")
    	new(float64(p))
    end
end

Geometric() = Geometric(0.5) # Flips of a fair coin

### handling support

isupperbounded(d::Union(Geometric, Type{Geometric})) = false
islowerbounded(d::Union(Geometric, Type{Geometric})) = true
isbounded(d::Union(Geometric, Type{Geometric})) = false

min(d::Union(Geometric, Type{Geometric})) = 0
max(d::Geometric) = Inf

insupport(::Geometric, x::Real) = isinteger(x) && zero(x) <= x
insupport(::Type{Geometric}, x::Real) = isinteger(x) && zero(x) <= x



mean(d::Geometric) = (1.0 - d.prob) / d.prob

median(d::Geometric) = -fld(0.6931471805599453,log1p(-d.prob)) - 1.0

mode(d::Geometric) = 0
modes(d::Geometric) = [0]

var(d::Geometric) = (1.0 - d.prob) / d.prob^2
skewness(d::Geometric) = (2.0 - d.prob) / sqrt(1.0 - d.prob)
kurtosis(d::Geometric) = 6.0 + d.prob^2 / (1.0 - d.prob)

entropy(d::Geometric) = (-xlogx(1.0 - d.prob) - xlogx(d.prob)) / d.prob



pdf(d::Geometric, x::Real) = insupport(d,x) ? d.prob*exp(log1p(-d.prob)*x) : 0.0
logpdf(d::Geometric, x::Real) = insupport(d,x) ? log(d.prob) + log1p(-d.prob)*x : -Inf

cdf(d::Geometric, q::Real) = q < zero(q) ? 0.0 : -expm1(log1p(-d.prob) * (floor(q) + 1.0))
ccdf(d::Geometric, q::Real) =  q < zero(q) ? 1.0 : exp(log1p(-d.prob) * (floor(q) + 1.0))
logcdf(d::Geometric, q::Real) = q < zero(q) ? -Inf : log1mexp(log1p(-d.prob) * (floor(q) + 1.0))
logccdf(d::Geometric, q::Real) =  q < zero(q) ? 0.0 : log1p(-d.prob) * (floor(q) + 1.0)

function quantile(d::Geometric, p::Real) 
    if isnan(p) || (p < zero(p)) || (p > one(p))
        return NaN
    elseif p == zero(p)
        return 0.0
    elseif p == one(p)
        return Inf
    end
    -fld(-log1p(-p),log1p(-d.prob))-1.0
end
function cquantile(d::Geometric, p::Real) 
    if isnan(p) || (p < zero(p)) || (p > one(p))
        return NaN
    elseif p == zero(p)
        return Inf
    elseif p == one(p)
        return 0.0
    end
    -fld(-log(p),log1p(-d.prob))-1.0
end
function invlogcdf(d::Geometric, p::Real) 
    if (p > zero(p)) || isnan(p)
        return NaN
    elseif isinf(p)
        return 0.0
    elseif p == zero(p)
        return Inf
    end
    -fld(-log1mexp(p),log1p(-d.prob))-1.0
end
function invlogccdf(d::Geometric, p::Real) 
    if (p > zero(p)) || isnan(p)
        return NaN
    elseif isinf(p)
        return Inf
    elseif p == zero(p)
        return 0.0
    end
    -fld(-p,log1p(-d.prob))-1.0
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


function rand(d::Geometric)
    e = Base.Random.randmtzig_exprnd()
    fld(e,-log1p(-d.prob))
end



## Fit model

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

