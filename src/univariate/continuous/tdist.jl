doc"""
    TDist(ν,μ,σ)

The *Students T distribution* with `ν` degrees of freedom, location `μ`, and scale `σ` has probability density function

$f(x; ν, μ, σ) = \frac{1}{\sqrt{ν} B(1/2, ν/2) σ}
\left( 1 + \frac{(x-μ)^2}{ν \sigma^2} \right)^{-\frac{ν + 1}{2}}$

```julia
TDist(ν,μ,σ)      # t-distribution with ν degrees of freedom, location μ, and scale σ
TDist(ν)      # t-distribution with ν degrees of freedom

params(d)     # Get the parameters, i.e. (ν,μ,σ)
dof(d)        # Get the degrees of freedom, i.e. ν
location(d)   # Get the location, i.e. μ
scale(d)      # Get the scale, i.e. σ
```

External links

[Student's T distribution on Wikipedia](https://en.wikipedia.org/wiki/Student%27s_t-distribution)

"""
immutable TDist <: ContinuousUnivariateDistribution
    #TODO right order?
    ν::Float64
    μ::Float64
    σ::Float64

    function TDist(ν::Real, μ::Float64=0.0, σ::Float64=1.0)
        @check_args(NonstandardT, ν > zero(ν))
        @check_args(NonstandardT, σ > zero(σ))
        new(ν,μ,σ)
    end
end

@distr_support TDist -Inf Inf


#### Parameters

dof(d::TDist) = d.ν
location(d::TDist) = d.μ
scale(d::TDist) = d.σ
params(d::TDist) = (d.ν,d.μ,d.σ)


#### Statistics

mean(d::TDist) = d.ν > 1.0 ? d.μ : NaN
median(d::TDist) = d.μ
mode(d::TDist) = d.μ

function var(d::TDist)
    ν = d.ν
    σ = d.σ
    ν > 2.0 ? σ * σ * ν / (ν - 2.0) :
    ν > 1.0 ? Inf : NaN
end

skewness(d::TDist) = d.ν > 3.0 ? 0.0 : NaN

#TODO need to write some numerical tests to see if kurt changes
function kurtosis(d::TDist)
    ν = d.ν
    ν > 4.0 ? 6.0 / (ν - 4.0) :
    ν > 2.0 ? Inf : NaN
end

function entropy(d::TDist)
    h = 0.5 * d.ν
    h1 = h + 0.5
    h1 * (digamma(h1) - digamma(h)) + 0.5 * log(d.ν) + lbeta(h, 0.5) + log(d.σ)
end


#### Evaluation & Sampling

pdf(d::TDist,x::Real) = tdistpdf(d.ν,(x-d.μ)/d.σ)/d.σ
logpdf(d::TDist,x::Real) = tdistlogpdf(d.ν,(x-d.μ)/d.σ) - log(d.σ)
cdf(d::TDist,x::Real) = tdistcdf(d.ν,(x-d.μ)/d.σ)
ccdf(d::TDist,x::Real) = tdistccdf(d.ν,(x-d.μ)/d.σ)
logcdf(d::TDist,x::Real) = tdistlogcdf(d.ν,(x-d.μ)/d.σ)
logccdf(d::TDist,x::Real) = tdistlogccdf(d.ν,(x-d.μ)/d.σ)
quantile(d::TDist,q::Float64) = convert(Real,d.μ + d.σ * tdistinvcdf(d.ν,q))
cquantile(d::TDist,q::Float64) = convert(Real,d.μ + d.σ * tdistinvccdf(d.ν,q))
invlogcdf(d::TDist,lq::Float64) = convert(Real,d.μ + d.σ * tdistinvlogcdf(d.ν,lq))
invlogccdf(d::TDist,lq::Float64) = convert(Real,d.μ + d.σ * tdistinvlogccdf(d.ν,lq))

rand(d::TDist) = d.μ + d.σ * StatsFuns.Rmath.tdistrand(d.ν)

function cf(d::TDist, t::Real)
    t == 0 && return complex(1.0)
    h = d.ν * 0.5
    q = d.ν * 0.25
    t2 = t*t/d.σ/d.σ
    complex(2*(q*t2)^q*besselk(h,sqrt(d.ν)*abs(t/d.σ))/gamma(h)/d.σ) * exp(t*d.μ*im)
end

function gradlogpdf(d::TDist, x::Float64)
    z = x - d.μ
     -((d.ν + 1.0) * z) / (z^2 + d.ν) / d.σ / d.σ
end
