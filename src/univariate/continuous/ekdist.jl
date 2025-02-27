"""
    kumaraswamy(α,β,γ,l,u)
The exponentiated (scaled) kumaraswamy (EK) distribution is 5 paramter two of which are normalizing constants (l,u) to a range.
```
```julia
kumaraswamy()          # distribution with zero log-mean and unit scale
params(d)            # Get the parameters, 
```
External links
* [Wikipedia](https://en.wikipedia.org/wiki/Kumaraswamy_distribution
* The exponentiated Kumaraswamy Distribution and its log-transform: Lemonte et al https://www.jstor.org/stable/43601233

"""
## NEED TO ACCOMONDATE LOG(1 -1) when x hits the upper bound of support
## I get errors in trying to extend Distiributions functions in the script because I don't understand modules lol
using Distributions
import Distributions: @check_args, @distr_support, params, convert, partype, beta, gamma
import Distributions: AbstractRNG,  maximum, minimum, convert
import Distributions:   pdf, logpdf, gradlogpdf, cdf, ccdf, logcdf, logccdf, quantile, cquantile, median, mean, var, skewness, kurtosis, mode
import Distributions: fit_mle


struct EKDist{T<:Real} <: ContinuousUnivariateDistribution
    α::T
    β::T
    γ::T
    l::T
    u::T
    EKDist{T}(α, β, γ, l, u) where {T} = new{T}(α, β, γ, l, u) 
end

function EKDist(α::T, β::T, γ::T, l::T, u::T; check_args::Bool=true) where {T <: Real}
    @check_args EKDist (α, α ≥ zero(α)) (β, β ≥ zero(β)) (γ, γ ≥ zero(γ)) (l < u)
    return EKDist{T}(α, β, γ, l, u) 
end

EKDist(α::Real, β::Real, γ::Real, l::Real, u::Real; check_args::Bool=true) = 
    EKDist(promote(α, β, γ, l, u)...; check_args=check_args)

EKDist(α::Integer, β::Integer, γ::Integer, l::Integer, u::Integer ; check_args::Bool=true) = 
    EKDist(float(α), float(β), float(γ), float(l), float(u); check_args=check_args)

EKDist(α::Real, β::Real, γ::Real; check_args::Bool=true) = EKDist(promote(α, β, γ, 0.0, 1.0)...; check_args=check_args)
EKDist(α::Real, β::Real; check_args::Bool=true) = EKDist(promote(α, β, 1.0, 0.0, 1.0)...; check_args=check_args)

# @distr_support macro would set constant bounds but we need to take bounds from input
# so we will just define them ourselves
maximum(d::EKDist) = d.u
minimum(d::EKDist) = d.l

#### Conversions
convert(::Type{EKDist{T}}, α::S, β::S, γ::S, l::S, u::S) where {T <: Real, S <: Real} = 
        EKDist(T(α), T(β), T(γ), T(l), T(u))
        
convert(::Type{EKDist{T}}, d::EKDist) where {T<:Real} = 
        EKDist{T}(T(d.α), T(d.β), T(d.γ), T(d.l), T(d.u))

convert(::Type{EKDist{T}}, d::EKDist{T}) where {T<:Real} = d

#### Parameters

params(d::EKDist) = (d.α, d.β, d.γ, d.l, d.u)
partype(::EKDist{T}) where {T} = T

#### Evaluation

function pdf(d::EKDist, x::Real)
      xs = (x - d.l) / (d.u - d.l)
      α = d.α
      β = d.β
      γ = d.γ
    return ( α * β * γ * 
             xs^(α - 1) * 
             (1 - xs^α)^(β - 1) * 
             (1 - (1 - xs^α)^β)^(γ - 1)
           )
end

function logpdf(d::EKDist, x::Real)
      xs =  (x - d.l) / (d.u - d.l) 
      α = d.α
      β = d.β
      γ = d.γ
    return  ( log(α) + log(β) + log(γ) + 
              log(xs)*(α - 1) + 
              log(1 - xs^α)*(β - 1) + 
              log(1 - (1 - xs^α)^β)*(γ - 1)
            ) 
end

function logpdf(d::EKDist, x::Real, edgebound::Real)
    # Failes at exactly 1 or 0, so I guess jitter it by small amount
    # this only matters to me in likelihood analysis for real data
    # that might get defined as the max of the data or Fitting
    # bound parameters as part of the process.
    xc = clamp( x, d.l + edgebound, d.u - edgebound)
    return logpdf(d, xc)  
end

# do we even need gradlogpdf?
# grad implied gradient but this is definitely derivative wrt x
function gradlogpdf(d::EKDist, x::Real)
    outofsupport = (x < d.l) | (x > d.u)
    xs = (x - d.l) / (d.u - d.l)
    α = d.α
    β = d.β
    γ = d.γ

    z = (α - 1) / xs +
        -(α * xs^(α - 1)) / (1 - xs^α) * (β - 1) +
        (α * β * xs^(α - 1)) * (1 - xs^α)^(β - 1) / (1 -(1 - xs^α)^β) * (γ - 1)
    
    return outofsupport ? zero(z) : z
end

function gradlogpdf(d::EKDist, x::Real, edgebound::Real)
    xc = clamp( x, d.l + edgebound, d.u - edgebound)
    return gradlogpdf(d, xc)
end



function cdf(d::EKDist, x::Real)
      xs = (x - d.l) / (d.u - d.l)
      α = d.α
      β = d.β
      γ = d.γ
    return (1 - ( 1 - xs^α)^β)^γ
end

function ccdf(d::EKDist, x::Real)
      xs = (x - d.l) / (d.u - d.l)
      α = d.α
      β = d.β
      γ = d.γ

    return 1 - (1 - ( 1 - xs^α)^β)^γ
end

function logcdf(d::EKDist, x::Real)
    return log(cdf(d, x))
end

function logccdf(d::EKDist, x::Real)
    return log(ccdf(d, x))
end

function quantile(d::EKDist, q::Real)
    α = d.α
    β = d.β
    γ = d.γ
    xs = ( 1 - (1 - q^(1/γ) )^(1/β) )^(1/α) 
    x = xs * (d.u - d.l) + d.l
    return x
end

cquantile(d::EKDist, q::Real) = quantile(d, 1 - q)


#### Sampling
# `rand`: Uses fallback inversion sampling method

## Fitting

function fit_mle(::Type{<:EKDist}, x::AbstractArray{T}) where T<:Real
    #incomplete
end


#### Statistics
median(d::EKDist) = quantile(d, .5) # right?

function ekdmoment(d::EKDist, k::Real)
    α = d.α
    β = d.β
    γ = d.γ

    # can't tell from paper if its infinite sum or stops at the floor of γ -1
    # if infinite, we need to do convergence to get there and reflexive function on gamma(γ - i) when negative
    if ( !isinteger(γ) | (γ < 0) ) return error("moment currently only implemented when γ is positive integer") end

    # has infinite sums so probably need use Richardson.jl per googling
     
     momentseries = [ β * gamma(γ+1) * (-1)^i * beta(1 + k/α, β * (i + 1) ) / (gamma(γ-i) * factorial(i)) for i in 0:Int(γ - 1)]
       
     return sum(momentseries) 

end

mean(d::EKDist) = ekdmoment(d, 1)*(d.u - d.l) + d.l # need to scale it

var(d::EKDist) = (ekdmoment(d, 2) - ekdmoment(d, 1)^2) * (d.u - d.l)^2 # no clue if this scaling makes sense...
std(d::EKDist) = var(d::EKDist) ^.5

# These aren't scaled for sure.
skewness(d::EKDist) = if (d.l == 0) & (d.u == 1 )
                        (ekdmoment(d, 3) - 3 * mean(d) * var(d) - mean(d)^3) / var(d)^(3/2) 
                      else error("I didn't scale these yet, so it's not remotely right")
                      end

kurtosis(d::EKDist) = if (d.l == 0) & (d.u == 1 )
                        a1 = ekdmoment(d, 1)
                        a2 = ekdmoment(d, 2)
                        a3 = ekdmoment(d, 3)
                        a4 = ekdmoment(d, 4)
                        
                        return (a4 + a1 * (-4a3 + a1 * (6a2 - 3a1^2))) / (a2-a1^2)^2 - 3

                    else return error("I didn't scale these yet, so it's not remotely right")
                    end
                    
function mode(d::EKDist)
    if (d.γ == 1) & (d.α >= 1) & (d.β >= 1) & !(d.α == d.β == 1)
        return ( (d.α -1) / (d.α * d.β - 1) ) ^ (1 / d.α)

    else (d.α > 1) & (d.β > 1) &  (d.γ > 1)
        error("You'll have to approximate via pdf samples because I can't find general roots from derivatives like form: x^(α-1) * (1-x^α)^(β-1).
                    \n Take caution because combinations of parameters can give you situations you  might not expect especially across the boundary of 1.0 for parameters. \n\n")

    end
end

#= function entropy(d::EKDist) μ # In paper but not sure if is the expected form of entropy for package.
end =#
