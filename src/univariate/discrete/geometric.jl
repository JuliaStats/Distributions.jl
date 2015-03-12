immutable Geometric <: DiscreteUnivariateDistribution
    p::Float64

    function Geometric(p::Real)
        zero(p) < p < one(p) || error("prob must be in (0, 1)")
    	@compat new(Float64(p))
    end

    Geometric() = Geometric(0.5) # Flips of a fair coin
end

@distr_support Geometric 0 Inf


### Parameters

succprob(d::Geometric) = d.p
failprob(d::Geometric) = 1.0 - d.p
params(d) = (d.p,)


### Statistics

mean(d::Geometric) = failprob(d) / succprob(d)

median(d::Geometric) = -fld(logtwo, log1p(-d.p)) - 1.0

mode(d::Geometric) = 0

var(d::Geometric) = (1.0 - d.p) / abs2(d.p)

skewness(d::Geometric) = (2.0 - d.p) / sqrt(1.0 - d.p)

kurtosis(d::Geometric) = 6.0 + abs2(d.p) / (1.0 - d.p)

entropy(d::Geometric) = (-xlogx(succprob(d)) - xlogx(failprob(d))) / d.p


### Evaluations

function pdf(d::Geometric, x::Int)
    if x >= 0 
        p = d.p
        return p < 0.1 ? p * exp(log1p(-p) * x) : d.p * (1.0 - p)^x
    else 
        return 0.0
    end
end

logpdf(d::Geometric, x::Int) = x >= 0 ? log(d.p) + log1p(-d.p) * x : -Inf

immutable RecursiveGeomProbEvaluator <: RecursiveProbabilityEvaluator
    p0::Float64
end

RecursiveGeomProbEvaluator(d::Geometric) = RecursiveGeomProbEvaluator(failprob(d))
nextpdf(s::RecursiveGeomProbEvaluator, p::Float64, x::Integer) = p * s.p0
_pdf!(r::AbstractArray, d::Geometric, rgn::UnitRange) = _pdf!(r, d, rgn, RecursiveGeomProbEvaluator(d))


function cdf(d::Geometric, x::Int) 
    x < 0 && return 0.0
    p = succprob(d)
    n = x + 1
    p < 0.5 ? -expm1(log1p(-p)*n) : 1.0-(1.0-p)^n
end

function ccdf(d::Geometric, x::Int)
    x < 0 && return 1.0 
    p = succprob(d)
    n = x + 1
    p < 0.5 ? exp(log1p(-p)*n) : (1.0-p)^n
end

logcdf(d::Geometric, x::Int) = x < 0 ? -Inf : log1mexp(log1p(-d.p) * (x + 1))

logccdf(d::Geometric, x::Int) =  x < 0 ? 0.0 : log1p(-d.p) * (x + 1)

quantile(d::Geometric, p::Float64) = invlogccdf(d, log1p(-p))

cquantile(d::Geometric, p::Float64) = invlogccdf(d, log(p))

invlogcdf(d::Geometric, lp::Float64) = invlogccdf(d, log1mexp(lp))

function invlogccdf(d::Geometric, lp::Float64) 
    if (lp > 0.0) || isnan(lp)
        return NaN
    elseif isinf(lp)
        return Inf
    elseif lp == 0.0
        return 0.0
    end
    max(ceil(lp/log1p(-d.p))-1.0,0.0)
end

function mgf(d::Geometric, t::Real)
    p = succprob(d)
    p / (expm1(-t) + p)
end

function cf(d::Geometric, t::Real)
    p = succprob(d)
    # replace with expm1 when complex version available
    p / (exp(-t*im) - 1.0 + p)
end


### Sampling

rand(d::Geometric) = floor(Int,-randexp() / log1p(-d.p))


### Model Fitting

immutable GeometricStats <: SufficientStats
    sx::Float64
    tw::Float64

    @compat GeometricStats(sx::Real, tw::Real) = new(Float64(sx), Float64(tw))
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

