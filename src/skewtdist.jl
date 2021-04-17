"""
SkewTDist(ξ, ω, α, df)
    The *skew t distribution* is a continuous probability distribution
    that generalises the student's t distribution to allow for non-zero skewness.
#
External links
* [Skew t distribution on Wikipedia](https://en.wikipedia.org/wiki/Skewed_generalized_t_distribution#Skewed_t_distribution)
* [Dissertation](https://etd.ohiolink.edu/!etd.send_file?accession=bgsu1496762068499547&disposition=inline)
* [SkewDist.jl](https://github.com/STOR-i/SkewDist.jl)
"""

struct SkewTDist{T<:Real} <: ContinuousUnivariateDistribution
    ξ::T  # Location
    ω::T  # Scale
    α::T  # Skewness
    df::T # Degrees of freedom
    SkewTDist{T}(ξ::T, ω::T, α::T, df::T) where {T} = new{T}(ξ, ω, α, df)
end

function SkewTDist(ξ::T, ω::T, α::T, df::T; check_args=true) where {T <: Real}
    check_args && @check_args(SkewTDist, ω > zero(ω) && df>zero(df))
    return SkewTDist{T}(ξ, ω, α, df)
end

SkewTDist(ξ::Real, ω::Real, α::Real, df::Real) = SkewTDist(promote(ξ, ω, α, df)...)
SkewTDist(ξ::Integer, ω::Integer, α::Integer, df::Integer) = SkewTDist(float(ξ), float(ω), float(α), float(df))
SkewTDist(df::Real) = SkewTDist(0.0, 1.0, 0.0, df)

@distr_support SkewTDist -Inf Inf

#### Conversions
convert(::Type{SkewTDist{T}}, ξ::S, ω::S, α::S, df::S) where {T <: Real, S <: Real} = SkewTDist(T(ξ), T(ω), T(α), T(df))
convert(::Type{SkewTDist{T}}, d::SkewTDist{S}) where {T <: Real, S <: Real} = SkewTDist(T(d.ξ), T(d.ω), T(d.α), T(d.df), check_args=false)

#### Parameters
params(d::SkewTDist) = (d.ξ, d.ω, d.α, d.df)
@inline partype(d::SkewTDist{T}) where {T<:Real} = T

#### Statistics
minimum(dist::SkewTDist) = -Inf
maximum(dist::SkewTDist) = Inf
dof(d::SkewTDist) = d.df

δ(α::Real) = α/sqrt(1.0 + α^2)
function μ(d::SkewTDist)
    #δ(d.α)*sqrt(d.df/π)* exp(lgamma(0.5*(d.df-1.0)) - lgamma(0.5*d.df))
    a1 = δ(d.α) * sqrt(d.df/π)
    a2 = exp(logabsgamma(0.5*(d.df-1.0))[1] - logabsgamma(0.5*d.df)[1])
    return a1 * a2
end

mean(d::SkewTDist) = d.df > 1.0 ? d.ξ + d.ω * μ(d) : Inf
var(d::SkewTDist) = d.df > 2.0 ? (d.ω^2) * (d.df/(d.df - 2.0)) - d.ω^2 * μ(d)^2 : Inf
std(d::SkewTDist) = √var(d)

function skewness(d::SkewTDist)
    a0 = μ(d)/( (d.df/(d.df - 2.0)) - μ(d)^2 )^3
    a1 = (d.df*(3.0 -δ(d.α)^2 )/(d.df - 3.0)) - 3.0*d.df/(d.df-2.0) + 2.0*μ(d)^2
    return d.df > 3.0 ? a0*a1 : Inf
end

#kurtosis(d::SkewTDist) #see paper
#moment() #see paper
#mode(d::SkewNormal)

#### Evalution
# from paper
# function pdf(d::SkewTDist, x::Real)
#     a0 = (2.0/d.ω)
#     a1 = pdf(TDist(d.df), (x-d.ξ)/d.ω)
#     a2 = cdf(TDist(d.df + 1.), sqrt((d.df+1.)/(d.df + ((x-d.ξ)/d.ω)^2 )) * d.α*(x-d.ξ)/d.ω )
#     return a0 * a1 * a2
#     #2.0/d.ω * normpdf((x-d.ξ)/d.ω) * normcdf(d.α*(x-d.ξ)/d.ω)
# end

_t₁(x, df::Float64) = Distributions.tdistpdf(df, x)
_logt₁(x, df::Float64) = Distributions.tdistlogpdf(df, x)
_T₁(x, df::Float64) = Distributions.tdistcdf(df, x)
_logT₁(x, df::Float64) = Distributions.tdistlogcdf(df, x)

skewtdistpdf(α::Real, df::Real, z::Real) = 2.0* _t₁(z, df) * _T₁(α * z * sqrt((df + 1)/(z^2 + df)), df + 1)

function pdf(dist::SkewTDist, x::Real)
    z = (x - dist.ξ)/dist.ω
    skewtdistpdf(dist.α, dist.df, z)/dist.ω
end

logpdf(d::SkewTDist, x::Real) = log(pdf(d, x))

# CDF uses QuadGK
function cdf(dist::SkewTDist, x::Real)
    quadgk(t->pdf(dist,t), -Inf, x)[1]
end

#quantile/mgf/cf

# quantile uses Roots.fzero()
# function quantile(dist::SkewTDist, β::Float64)
#     if dist.α < 0
#         β = 1 - β
#     end
#     a = tdistinvcdf(dist.df, β)
#     b = sqrt(fdistinvcdf(1.0, dist.df, β))
#     qz = fzero(x->cdf(dist, x) - β, a, b)
# end

#### Sampling
# rand() need MVSkewNormal PR
# function rand(dist::SkewTDist)
#     chisqd = Chisq(dist.df)
#     w = rand(chisqd)/dist.df
#     Ω = Array(Float64, 1,1)
#     Ω[1,1] = dist.ω^2
#     sndist = MvSkewNormal(Ω, [dist.α])
#     x = rand(sndist)[1]
#     return dist.ξ + x/sqrt(w)
#     ## Ω = Array(Float64, 1,1)
#     ## Ω[1,1] = dist.ω^2
#     ## return rand(MvSkewTDist([dist.ξ], Ω, [dist.α], dist.df))[1]
# end


#### Fitting

