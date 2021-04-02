"""
    AbstractMoments{N}

A representation of statistical moments of a distribution

The following functions are supported
- `n_moments(m)`: get the number of recorded moments

The following getters return a single moment or 
throw an error if the moment has not been recorded
- `mean(m)`: get the mean
- `var(m)`: get the variance
- `skewness(m)`: get the variance
- `kurtosis(m)`: get the variance
- `getindex(m,i)`: get the ith moment, i.e. indexing m[i]

The basic implementation `Moments` is immutable and
`convert(AbstractArray, m::Moments)` returns an `SArray{N,T}`.

# Examples
```jldoctest am; output = false, setup = :(using Statistics,Distributions)
m = Moments(1,0.2);
n_moments(m) == 2
var(m) == m[2]
# output
true
```
```julia
kurtosis(m) # throws error because its above 2nd moment
```
"""
abstract type AbstractMoments{N} end
n_moments(::Type{<:AbstractMoments{N}}) where N = N
n_moments(m::AbstractMoments{N}) where N = N
Distributions.mean(m::AbstractMoments) = n_moments(m) >= 1 ? m[1] : 
    error("mean not recorded")
Distributions.var(m::AbstractMoments) = n_moments(m) >= 2 ? m[2] : 
    error("variance not recorded")
Distributions.std(m::AbstractMoments) = n_moments(m) >= 2 ? sqrt(m[2]) : 
    error("std not recorded")
Distributions.skewness(m::AbstractMoments) = n_moments(m) >= 3 ? m[3] : 
    error("skewness not recorded")
Distributions.kurtosis(m::AbstractMoments) = n_moments(m) >= 4 ? m[4] : 
    error("kurtosis not recorded")


struct Moments{N,T} <: AbstractMoments{N}
    all::SVector{N,T}
end
Moments(x...) = Moments(SVector{length(x)}(promote(x...)))
Moments() = Moments(SA[])
Base.getindex(m::Moments, i) = n_moments(m) >= i ? m.all[i] : 
    error("$(i)th moment not recorded.")
Base.convert(::Type{AbstractArray}, m::Moments) = m.all

"""
    moments(D, ::Val{N} = Val(2))

Get the first N moments of a distribution.

See also type [`AbstractMoments`](@ref).

## Examples
```julia
moments(LogNormal(), Val(4))  # first four moments 
moments(Normal())  # mean and variance
```
"""
function moments(d::Distribution, ::Val{N} = Val(2)) where N 
    N isa Integer || error("N must be a positive Integer")
    N > 4 && error("Getting moments above 4 not yet implemented for " * 
        "distribution $(typeof(d)).")
    N == 4 && return(Moments(mean(d), var(d), skewness(d), kurtosis(d)))
    N == 3 && return(Moments(mean(d), var(d), skewness(d)))
    N == 2 && return(Moments(mean(d), var(d)))
    N == 1 && return(Moments(mean(d)))
    N == 0 && return(Moments())
    error("N must be a positive Integer.")
end

"""
    fit(D, m)
    
Fit a statistical distribution of type `D` to given moments `m`.

# Arguments
- `D`: The type of distribution to fit
- `m`: The moments of the distribution

# Notes
This can be used to approximate one distribution by another.

See also [`AbstractMoments`](@ref), [`moments`](@ref). 


# Examples
```jldoctest fm1; output = false, setup = :(using Statistics,Distributions)
d = fit(LogNormal, Moments(3.2,4.6));
(mean(d), var(d)) .≈ (3.2,4.6)
# output
(true, true)
```
```jldoctest fm1; output = false, setup = :(using Statistics,Distributions)
d = fit(LogNormal, moments(Normal(3,1.2)));
(mean(d), std(d)) .≈ (3,1.2)
# output
(true, true)
```
```julia
plot(d); lines(!Normal(3,1.2))
```
"""
fit(::Type{D}, m::AbstractMoments) where {D<:Distribution} = 
    error("fitting to moments not implemented for distribution of type $D")



struct QuantilePoint{TQ,TP}
    q::TQ
    p::TP
    QuantilePoint{TQ,TP}(q,p) where {TQ,TP} = 0 < p < 1 ? new(q,p) : 
        error("p must be in (0,1)")
end
QuantilePoint(q,p) = QuantilePoint{typeof(q),typeof(p)}(q,p)

QuantilePoint(qp::QuantilePoint; q = qp.q, p = qp.p) = QuantilePoint(q,p)
Base.show(io::IO, qp::QuantilePoint) = print(io, "QuantilePoint($(qp.q),$(qp.p))")
function Base.isless(x::QuantilePoint,y::QuantilePoint)
    is_equal_q = (x.q == y.q)
    ((x.p == y.p) && !is_equal_q) && error("incompatible: $x,$y")
    isless = (x.q < y.q)
    # for different p, q needs to be different
    (isless && (x.p > y.p)) && error("incompatible: $(x),$(y)")
    (!isless && !is_equal_q && (x.p < y.p))  && error("incompatible: $x,$y")
    return(isless)
end

macro qp(q,p) :(QuantilePoint($(esc(q)), $(esc(p)))) end
macro qp_ll(q0_025) :(QuantilePoint($(esc(q0_025)),0.025)) end
macro qp_l(q0_05) :(QuantilePoint($(esc(q0_05)),0.05)) end
macro qp_m(median) :(QuantilePoint($(esc(median)),0.5)) end
macro qp_u(q0_95) :(QuantilePoint($(esc(q0_95)),0.95)) end
macro qp_uu(q0_975) :(QuantilePoint($(esc(q0_975)),0.975)) end

macro qs_cf90(q0_05,q0_95) 
    :(Set([QuantilePoint($(esc(q0_05)),0.05),QuantilePoint($(esc(q0_95)),0.95)])) end
macro qs_cf95(q0_025,q0_975) 
    :(Set([QuantilePoint($(esc(q0_025)),0.025),QuantilePoint($(esc(q0_975)),0.975)])) end


"""
    fit(D, lower, upper)

Fit a statistical distribution to a set of quantiles 

# Arguments
- `D`: The type of the distribution to fit
- `lower`:  lower QuantilePoint (p,q)
- `upper`:  upper QuantilePoint (p,q)

# Notes
Several macros help to construct QuantilePoints
- `@qp(q,p)`    quantile at specified p: `QuantilePoint(q,p)`
- `@qp_ll(q0_025)`  quantile at very low p: `QuantilePoint(q0_025,0.025)` 
- `@qp_l(q0_05)`    quantile at low p: `QuantilePoint(q0_05,0.05)` 
- `@qp_m(median)`   quantile at median: `QuantilePoint(median,0.5)` 
- `@qp_u(q0_95)`    quantile at high p: `QuantilePoint(q0_95,0.95)`  
- `@qp_uu(q0_975)`  quantile at very high p: `QuantilePoint(q0_975,0.975)` 

# Examples
```jldoctest; output = false, setup = :(using Statistics,Distributions)
d = fit(LogNormal, @qp_m(3), @qp_uu(5));
quantile.(d, [0.5, 0.975]) ≈ [3,5]
# output
true
```
"""
function fit(::Type{D}, lower::QuantilePoint, upper::QuantilePoint) where D<:Distribution 
    error("fitting to two quantile points not implemented for distribution of type $D")
end,
function fit_median_quantile(D::Type{DT}, median, qp::QuantilePoint) where {DT <: Distribution}
    return(fit(D, @qp_m(median), qp))
end


"""
    fit(D, val, qp, ::Val{stats} = Val(:mean))

Fit a statistical distribution to a quantile and given statistics

# Arguments
- `D`: The type of distribution to fit
- `val`: The value of statistics
- `qp`: QuantilePoint(q,p)
- `stats` Which statistics to fit: defaults to `Val(:mean)`. 
   Alternatives are: `Val(:mode)`, `Val(:median)`

# Examples
```jldoctest fm2; output = false, setup = :(using Statistics,Distributions)
d = fit(LogNormal, 5, @qp_uu(14));
(mean(d),quantile(d, 0.975)) .≈ (5,14)
# output
(true, true)
```
```jldoctest fm2; output = false, setup = :(using Statistics,Distributions)
d = fit(LogNormal, 5, @qp_uu(14), Val(:mode));
(mode(d),quantile(d, 0.975)) .≈ (5,14)
# output
(true, true)
```
"""
function fit(::Type{D}, val, qp::QuantilePoint, ::Val{stats} = Val(:mean)) where {D<:Distribution, stats}
    stats == :mean && return(fit_mean_quantile(D, val, qp))
    stats == :mode && return(fit_mode_quantile(D, val, qp))
    stats == :median && return(fit_median_quantile(D, val, qp))
    error("unknown stats: $stats")
end,
function fit_mean_quantile(::Type{D}, mean::Real, qp::QuantilePoint) where D<:Distribution  
    error("fit_mean_quantile not yet implemented for Distribution of type: $D")
end
function fit_mode_quantile(::Type{D}, mode::Real, qp::QuantilePoint)  where D<:Distribution
    error("fit_mode_quantile not yet implemented for Distribution of type: $D")
end



