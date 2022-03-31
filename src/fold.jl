"""
    folded(d::UnivariateDistribution, crease::Real)
    folded(d::UnivariateDistribution, crease::Real, keep_right=false)

Creates a _folded distribution_ from the original distribution `d` about the value `crease`.
This function defaults to folding the left side onto the right, but by using `keep_right=false` one can reflect the right side onto the left.

##### Folded Distributions

A folded distribution ``F_{c}(D)`` of a distribution ``D`` at crease ``c \\in \\mathbb{R}`` is the reflection of the distribution below ``c`` onto the the distribution above ``c``. 
The pdf of such a distribution is given by:
```math
p(x | F_{c}(D)) = p(x | D) + p(x' | D)
```
where ``x' = 2c - x``

This can be implemented for any univariate continuous distribution by using the following method:

    folded(d::UnivariateDistribution, crease::Real)

If one wants to reflect points _above_ the crease ``c`` onto points _below_ ``c`` (such that the resultant distribution lives on points below ``c``) one can do that using the `keep_right=false` tag:

    folded(d::UnivariateDistribution, crease::Real, keep_right=false)

A very useful and oft-occuring example of this is the action of the absolute value function (``|\\cdot|``) on random variables. 
For example if we have a random variable obeying the normal distribution:

```math
\\hat{X} \\sim \\mathcal{N}(\\mu,\\sigma)
```

Taking the absolute value of this random variable always follows a _folded normal distribution_ with a crease at ``0``

```math
|\\hat{X}| \\sim F_{c=0}(\\mathcal{N}(\\alpha,\\beta))
```
In julia that looks like:

```julia
folded_normal = folded(Normal(μ,σ), 0.0)
```
and then we can ask for many things like the pdf, or sample the distribution. The samples ofcourse will then be positive:

```julia
all(rand(folded_normal, 1000) .≥ 0) # True
```

In general, one can take __any__ univariate continous distribution and fold it using this function.

Mathematically all this is doing is that the pdf of the folded distribution at point ``x`` is essentially the sum of the pdf at point ``x`` of the original distribution and the pdf at point ``x'``, reflected about the crease ``c``:

```math
p(x | F_{c}(D)) = p(x | D) + p(x' | D)
```
where ``x' = 2c - x``

##### Example
We can create the absolute value of the laplace distribution by:
```julia
folded_laplace = folded(Laplace(μ,b), 0.0)
```
as a quick check we can see if the 1000 samples generated from this are all greater than 0:
```julia
all(rand(folded(Lapace(μ,b),0),1000) .≥ 0) # -> true
```
"""
function folded end


#the folded function with no arguments will just return the function itself
folded(d::UnivariateDistribution, ::Nothing, ::Nothing) = d


##### Defining the function and the initialization functions #####
"""
Generic wrapper for the folded version of a distribution. 
Holds the original distribution, and two truncated copies of the 
distribution: one above the crease and one below. 
"""
struct Folded{D<:UnivariateDistribution, K<:Truncated, S<:ValueSupport, T <: Real} <: UnivariateDistribution{S}
    original::D      # the original distribution (unfolded)
    included::K     # The part of the old distribution left unchanged
    excluded::K     # The folded part that is now part of the new distribution
    crease::T     # Value at which to make the fold
    keep_right::Bool # true if distribution folds the left distribution on to the right distribution, false if otherwise

    # Internal constructor
    function Folded(unfolded::UnivariateDistribution, included::K, excluded::K,  crease::T, keep_right::Bool;  check_args=true) where {T <: Real, K <: Truncated}
        # Check if the crease is within the support of the original distiribution
        check_args && @check_args(Folded, insupport(unfolded, crease))
        new{typeof(unfolded), typeof(included), value_support(typeof(unfolded)), T}(unfolded, included, excluded, crease, keep_right)
    end
end

# This is the user facing initialization function that will be used in most cases
"""
   folded(d::UnivariateDistribution, crease::T; keep_right=true) 

Creates a folded distribution from the original distribution `d` by folding the distribution about the `crease`.
This function defaults to folding the left side onto the right, but by using `keep_right=false` one can reflect the right side onto the left.

"""
function folded(d::UnivariateDistribution, crease::T; keep_right=true) where {T <: Real}
    lower = minimum(d); 
    upper = maximum(d);
    left_distribution = truncated(d, lower, crease);
    right_distribution = truncated(d, crease, upper);
    if keep_right
        included = right_distribution
        excluded = left_distribution
    else
        included = left_distribution
        excluded = right_distribution
    end
    Folded(d, included, excluded, crease, keep_right)
end

############################################################################
# This section is there so that typeof(d)(params(d)...) == d works:
############################################################################

# Define params as the unfolded original parameters, just appended with the crease and keep_right variables
params(d::Folded) = tuple(params(d.original)..., d.crease, d.keep_right)
partype(d::Folded) = partype(d.original)

# Use the type as the constructor and construct the original distribution to feed into the folded (lowercase) constructor.
Folded{D,K,S,T}(p...) where {D<:UnivariateDistribution, K<:Truncated, S<:ValueSupport, T <: Real} = folded(D(p[1:(end-2)]...),p[end-1],keep_right=p[end])


######################################################
#             Convenience Functions
######################################################

"""
    original_support_error_check(x::Real, d::Folded)

This function is like `insupport`, but in this case,
returns an error if the function is not in the support of the __original__ distribution.
"""
function original_support_error_check(x::Real,d::Folded)
    if !insupport(d.original, x)
        throw(DomainError(x, "value $x is not inside the original distribution domain"))
    end
end

"""
    fold_value(x::Real, d::Folded)

This function takes in a point on the real axis and attempts to bring it __into__ the current
domain of the Folded distribution `d`:

- if `x` is inside the domain of the folded distribution, then this returns `x`
- if `x` is outside the domain of the folded distribution, then this reflects this value into the
current distribution by returning `2c - x`
- if `x` is not in the support of the parent distribution, then the function gives a DomainError 
"""
function fold_value(x::Real, d::Folded)
    original_support_error_check(x,d) # Will error if x ∉ domain(d.original)
    return insupport(d.excluded, x) ? (2*d.crease - x) : x
end


"""
    unfold_value(x::Real, d::Folded)

This function takes in a point on the real axis and attempts to fold it __out of__  the current
domain of the Folded distribution `d`:

- if `x` is inside the domain of the folded distribution, then this reflects this value into the
current distribution by returning `2c - x`
- if `x` is outside the domain of the folded distribution, but inside the parent distribution, then this
just returns `x`
- if `x` is not in the support of the parent distribution, then the function gives a DomainError 
"""
function unfold_value(x::Real, d::Folded)
    original_support_error_check(x,d) # Will error if x ∉ domain(d.original)
    return insupport(d.included, x) ? (2*d.crease - x) : x
end



######################################################
#         Distribution.jl extended methods
######################################################
"""
    insupport(d::Folded, x::Real)

Returns true if the value `x` is in the domain (or support) of the folded
distribution.
"""
insupport(d::Folded, x::Real) = insupport(d.included, x)


"""
    minimum(d::Folded)

Returns the lowest value of the domain of the distribution
"""
minimum(d::Folded) = minimum(d.included)



"""
    maximum(d::Folded)

Returns the highest value of the domain of the distribution
"""
maximum(d::Folded) = maximum(d.included)


"""
    rand(::AbstractRNG, d::Folded)

Returns samples from the folded distribution
"""
rand(::AbstractRNG, d::Folded) = fold_value(rand(d.original), d)

"""
   logpdf(d::Folded, x::Real)

Returns the logarithm of the pdf of the folded distribution. 
Returns -Inf if the value is out of the distribution.

Example:
```julia
logpdf(folded(Normal(μ,σ),0), -3) # -> -Inf
logpdf(folded(Normal(1.0,1.0),0), 3) # -> 0.05412479673895295
``` 
"""
function logpdf(d::Folded, x::Real)
    !insupport(d,x) && return -Inf
    return logaddexp(logpdf(d.original,x),logpdf(d.original,unfold_value(x,d)))
end


"""
   pdf(d::Folded, x::Real)

Returns the the pdf of the folded distribution. This is simply the
sum of the value of the pdf of the original distribution at two locations:
one at `x` and the other at the reflected point `2c-x`, where `c` is the crease.

Example:
```julia
pdf(folded(Normal(μ,σ),0), -3) # -> 0.0
pdf(folded(Normal(1.0,1.0),0), 3) # -> 0.05412479673895295
pdf(folded(Normal(3,4),0.0),0.1) ≈ pdf(Normal(3,4),-0.1) + pdf(Normal(3,4),0.1) # -> true
```
"""
function pdf(d::Folded, x::Real)
    !insupport(d,x) && return zero(x)
    return pdf(d.original,x) + pdf(d.original,unfold_value(x,d))
end

"""
    cdf(d::Folded, x::Real) 

Example:
cdf(folded(Beta(α,β),0.5), -3) # -> 0.0
cdf(folded(Beta(α,β),0.5), 3) # -> 1.0
cdf(folded(Normal(1.0,1.0),0), 1.0) # -> 0.4772498680518208
```
"""
function cdf(d::Folded, x::Real) 
    if !insupport(d,x)
        (x ≤ minimum(d)) && return zero(x) # Is below the support
        (x ≥ maximum(d)) && return one(x) # Is above the support
    end
    d.included.tp*cdf(d.included, x) + d.excluded.tp*ccdf(d.excluded, unfold_value(x,d))
end


# Quantiles will be calculated by the newton method:
@quantile_newton(Folded)


######################################################
#         Statistics extended methods
######################################################

"""
    mean(d::Folded)

Gives the mean of a folded distribution. This will only work if `mean` is implemented for
the truncated version of the parent distribution.

Example:
```julia
mean(folded(Normal(1,2),0)) # -> 1.7911862296052243
```

Note that if the mean for the truncated distribution is not implemented, then this will give a Method 
Error:
```
mean(folded(Beta(1,2)),0) # Errors out because mean(truncated(Beta(1,2),..)) is not implemented
# >> MethodError: no method matching iterate(::Truncated{Beta{...}...})
```
"""
mean(d::Folded) = mean(d.included)*d.included.tp - mean(d.excluded)*d.excluded.tp + 2*d.crease


######################################################
#         Show functions
######################################################

function show(io::IO, d::Folded)
    print(io, "Folded(")
    d0 = d.original
    uml, namevals = _use_multline_show(d0)
    uml ? show_multline(io, d0, namevals) :
          show_oneline(io, d0, namevals)
    print(io, "; crease = $(d.crease)")
    if !d.keep_right
        print(io, " | fold right onto left")
    else
        print(io, " | fold left onto right")
    end
    print(io, ")")
end


_use_multline_show(d::Folded) = _use_multline_show(d.original)






