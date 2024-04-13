## sample space/domain

"""
`F <: VariateForm` specifies the form or shape of the variate or a sample.
"""
abstract type VariateForm end

"""
`F <: ArrayLikeVariate{N}` specifies the number of axes of a variate or
a sample with an array-like shape, e.g. univariate (scalar, `N == 0`),
multivariate (vector, `N == 1`) or matrix-variate (matrix, `N == 2`).
"""
abstract type ArrayLikeVariate{N} <: VariateForm end

const Univariate    = ArrayLikeVariate{0}
const Multivariate  = ArrayLikeVariate{1}
const Matrixvariate = ArrayLikeVariate{2}

"""
`F <: CholeskyVariate` specifies that the variate or a sample is of type
`LinearAlgebra.Cholesky`.
"""
abstract type CholeskyVariate <: VariateForm end

"""
    ValueSupport

Abstract type that specifies the support of elements of samples.

It is either [`Discrete`](@ref) or [`Continuous`](@ref).
"""
abstract type ValueSupport end

"""
    Discrete <: ValueSupport

This type represents the support of a discrete random variable.

It is countable. For instance, it can be a finite set or a countably infinite set such as
the natural numbers.

See also: [`Continuous`](@ref), [`ValueSupport`](@ref)
"""
struct Discrete   <: ValueSupport end

"""
    Continuous <: ValueSupport

This types represents the support of a continuous random variable.

It is uncountably infinite. For instance, it can be an interval on the real line.

See also: [`Discrete`](@ref), [`ValueSupport`](@ref)
"""
struct Continuous <: ValueSupport end

# promotions (e.g., in product distribution):
# combination of discrete support (countable) and continuous support (uncountable) yields
# continuous support (uncountable)
Base.promote_rule(::Type{Continuous}, ::Type{Discrete}) = Continuous

## Sampleable

"""
    Sampleable{F<:VariateForm,S<:ValueSupport}

`Sampleable` is any type able to produce random values.
Parametrized by a `VariateForm` defining the dimension of samples
and a `ValueSupport` defining the domain of possibly sampled values.
Any `Sampleable` implements the `Base.rand` method.
"""
abstract type Sampleable{F<:VariateForm,S<:ValueSupport} end

variate_form(::Type{<:Sampleable{VF}}) where {VF} = VF
value_support(::Type{<:Sampleable{<:VariateForm,VS}}) where {VS} = VS

"""
    length(s::Sampleable)

The length of each sample. Always returns `1` when `s` is univariate.
"""
Base.length(s::Sampleable) = prod(size(s))
Base.length(::Sampleable{Univariate}) = 1
Base.length(s::Sampleable{Multivariate}) = throw(MethodError(length, (s,)))

"""
    size(s::Sampleable)

The size (i.e. shape) of each sample. Always returns `()` when `s` is univariate, and
`(length(s),)` when `s` is multivariate.
"""
Base.size(s::Sampleable)
Base.size(s::Sampleable{Univariate}) = ()
Base.size(s::Sampleable{Multivariate}) = (length(s),)

"""
    eltype(::Type{Sampleable})

The default element type of a sample. This is the type of elements of the samples generated
by the `rand` method. However, one can provide an array of different element types to
store the samples using `rand!`.
"""
Base.eltype(::Type{<:Sampleable{F,Discrete}}) where {F} = Int
Base.eltype(::Type{<:Sampleable{F,Continuous}}) where {F} = Float64

"""
    nsamples(s::Sampleable)

The number of values contained in one sample of `s`. Multiple samples are often organized
into an array, depending on the variate form.
"""
nsamples(t::Type{Sampleable}, x::Any)
nsamples(::Type{D}, x::Number) where {D<:Sampleable{Univariate}} = 1
nsamples(::Type{D}, x::AbstractArray) where {D<:Sampleable{Univariate}} = length(x)
nsamples(::Type{D}, x::AbstractVector) where {D<:Sampleable{Multivariate}} = 1
nsamples(::Type{D}, x::AbstractMatrix) where {D<:Sampleable{Multivariate}} = size(x, 2)
nsamples(::Type{D}, x::Number) where {D<:Sampleable{Matrixvariate}} = 1
nsamples(::Type{D}, x::Array{Matrix{T}}) where {D<:Sampleable{Matrixvariate},T<:Number} = length(x)

for func in (:(==), :isequal, :isapprox)
    @eval function Base.$func(s1::A, s2::B; kwargs...) where {A<:Sampleable, B<:Sampleable}
        nameof(A) === nameof(B) || return false
        fields = fieldnames(A)
        fields === fieldnames(B) || return false

        for f in fields
            isdefined(s1, f) && isdefined(s2, f) || return false
            # perform equivalence check to support types that have no defined equality, such
            # as `missing`
            getfield(s1, f) === getfield(s2, f) || $func(getfield(s1, f), getfield(s2, f); kwargs...) || return false
        end

        return true
    end
end

function Base.hash(s::S, h::UInt) where S <: Sampleable
    hashed = hash(Sampleable, h)
    hashed = hash(nameof(S), hashed)

    for f in fieldnames(S)
        hashed = hash(getfield(s, f), hashed)
    end

    return hashed
end

"""
    Distribution{F<:VariateForm,S<:ValueSupport} <: Sampleable{F,S}

`Distribution` is a `Sampleable` generating random values from a probability
distribution. Distributions define a Probability Distribution Function (PDF)
to implement with `pdf` and a Cumulative Distribution Function (CDF) to implement
with `cdf`.
"""
abstract type Distribution{F<:VariateForm,S<:ValueSupport} <: Sampleable{F,S} end

const UnivariateDistribution{S<:ValueSupport}   = Distribution{Univariate,S}
const MultivariateDistribution{S<:ValueSupport} = Distribution{Multivariate,S}
const MatrixDistribution{S<:ValueSupport}       = Distribution{Matrixvariate,S}
const NonMatrixDistribution = Union{UnivariateDistribution, MultivariateDistribution}

const DiscreteDistribution{F<:VariateForm}   = Distribution{F,Discrete}
const ContinuousDistribution{F<:VariateForm} = Distribution{F,Continuous}

const DiscreteUnivariateDistribution     = Distribution{Univariate,    Discrete}
const ContinuousUnivariateDistribution   = Distribution{Univariate,    Continuous}
const DiscreteMultivariateDistribution   = Distribution{Multivariate,  Discrete}
const ContinuousMultivariateDistribution = Distribution{Multivariate,  Continuous}
const DiscreteMatrixDistribution         = Distribution{Matrixvariate, Discrete}
const ContinuousMatrixDistribution       = Distribution{Matrixvariate, Continuous}

# allow broadcasting over distribution objects
# to be decided: how to handle multivariate/matrixvariate distributions?
Broadcast.broadcastable(d::UnivariateDistribution) = Ref(d)

"""
    minimum(d::Distribution)

Return the minimum of the support of `d`.
"""
minimum(d::Distribution)

"""
    maximum(d::Distribution)

Return the maximum of the support of `d`.
"""
maximum(d::Distribution)

"""
    extrema(d::Distribution)

Return the minimum and maximum of the support of `d` as a 2-tuple.
"""
Base.extrema(d::Distribution) = minimum(d), maximum(d)

"""
    pdf(d::Distribution{ArrayLikeVariate{N}}, x::AbstractArray{<:Real,N}) where {N}

Evaluate the probability density function of `d` at `x`.

This function checks if the size of `x` is compatible with distribution `d`. This check can
be disabled by using `@inbounds`.

# Implementation

Instead of `pdf` one should implement `_pdf(d, x)` which does not have to check the size of
`x`. However, since the default definition of `pdf(d, x)` falls back to `logpdf(d, x)`
usually it is sufficient to implement `logpdf`.

See also: [`logpdf`](@ref).
"""
@inline function pdf(
    d::Distribution{ArrayLikeVariate{N}}, x::AbstractArray{<:Real,M}
) where {N,M}
    if M == N
        @boundscheck begin
            size(x) == size(d) ||
                throw(DimensionMismatch("inconsistent array dimensions"))
        end
        return _pdf(d, x)
    else
        @boundscheck begin
            M > N ||
                throw(DimensionMismatch(
                    "number of dimensions of the variates ($M) must be greater than or equal to the dimension of the distribution ($N)"
                ))
            ntuple(i -> size(x, i), Val(N)) == size(d) ||
                throw(DimensionMismatch("inconsistent array dimensions"))
        end
        return @inbounds map(Base.Fix1(pdf, d), eachvariate(x, variate_form(typeof(d))))
    end
end

function _pdf(d::Distribution{ArrayLikeVariate{N}}, x::AbstractArray{<:Real,N}) where {N}
    return exp(@inbounds logpdf(d, x))
end

"""
    logpdf(d::Distribution{ArrayLikeVariate{N}}, x::AbstractArray{<:Real,N}) where {N}

Evaluate the logarithm of the probability density function of `d` at `x`.

This function checks if the size of `x` is compatible with distribution `d`. This check can
be disabled by using `@inbounds`.

# Implementation

Instead of `logpdf` one should implement `_logpdf(d, x)` which does not have to check the
size of `x`.

See also: [`pdf`](@ref).
"""
@inline function logpdf(
    d::Distribution{ArrayLikeVariate{N}}, x::AbstractArray{<:Real,M}
) where {N,M}
    if M == N
        @boundscheck begin
            size(x) == size(d) ||
                throw(DimensionMismatch("inconsistent array dimensions"))
        end
        return _logpdf(d, x)
    else
        @boundscheck begin
            M > N ||
                throw(DimensionMismatch(
                    "number of dimensions of the variates ($M) must be greater than or equal to the dimension of the distribution ($N)"
                ))
            ntuple(i -> size(x, i), Val(N)) == size(d) ||
                throw(DimensionMismatch("inconsistent array dimensions"))
        end
        return @inbounds map(Base.Fix1(logpdf, d), eachvariate(x, variate_form(typeof(d))))
    end
end

# `_logpdf` should be implemented and has no default definition
# _logpdf(d::Distribution{ArrayLikeVariate{N}}, x::AbstractArray{<:Real,N}) where {N}

# TODO: deprecate?
"""
    pdf(d::Distribution{ArrayLikeVariate{N}}, x) where {N}

Evaluate the probability density function of `d` at every element in a collection `x`.

This function checks for every element of `x` if its size is compatible with distribution
`d`. This check can be disabled by using `@inbounds`.

Here, `x` can be
- an array of dimension `> N` with `size(x)[1:N] == size(d)`, or
- an array of arrays `xi` of dimension `N` with `size(xi) == size(d)`.
"""
Base.@propagate_inbounds function pdf(
    d::Distribution{ArrayLikeVariate{N}}, x::AbstractArray{<:AbstractArray{<:Real,N}},
) where {N}
    return map(Base.Fix1(pdf, d), x)
end

"""
    logpdf(d::Distribution{ArrayLikeVariate{N}}, x) where {N}

Evaluate the logarithm of the probability density function of `d` at every element in a
collection `x`.

This function checks for every element of `x` if its size is compatible with distribution
`d`. This check can be disabled by using `@inbounds`.

Here, `x` can be
- an array of dimension `> N` with `size(x)[1:N] == size(d)`, or
- an array of arrays `xi` of dimension `N` with `size(xi) == size(d)`.
"""
Base.@propagate_inbounds function logpdf(
    d::Distribution{ArrayLikeVariate{N}}, x::AbstractArray{<:AbstractArray{<:Real,N}},
) where {N}
    return map(Base.Fix1(logpdf, d), x)
end

"""
    pdf!(out, d::Distribution{ArrayLikeVariate{N}}, x) where {N}

Evaluate the probability density function of `d` at every element in a collection `x` and
save the results in `out`.

This function checks if the size of `out` is compatible with `d` and `x` and for every
element of `x` if its size is compatible with distribution `d`. These checks can be disabled
by using `@inbounds`.

Here, `x` can be
- an array of dimension `> N` with `size(x)[1:N] == size(d)`, or
- an array of arrays `xi` of dimension `N` with `size(xi) == size(d)`.

# Implementation

Instead of `pdf!` one should implement `_pdf!(out, d, x)` which does not have to check the
size of `out` and `x`. However, since the default definition of `_pdf!(out, d, x)` falls
back to `logpdf!` usually it is sufficient to implement `logpdf!`.

See also: [`logpdf!`](@ref).
"""
Base.@propagate_inbounds function pdf!(
    out::AbstractArray{<:Real},
    d::Distribution{ArrayLikeVariate{N}},
    x::AbstractArray{<:AbstractArray{<:Real,N},M}
) where {N,M}
    return map!(Base.Fix1(pdf, d), out, x)
end

Base.@propagate_inbounds function logpdf!(
    out::AbstractArray{<:Real},
    d::Distribution{ArrayLikeVariate{N}},
    x::AbstractArray{<:AbstractArray{<:Real,N},M}
) where {N,M}
    return map!(Base.Fix1(logpdf, d), out, x)
end

@inline function pdf!(
    out::AbstractArray{<:Real},
    d::Distribution{ArrayLikeVariate{N}},
    x::AbstractArray{<:Real,M},
) where {N,M}
    @boundscheck begin
        M > N ||
            throw(DimensionMismatch(
                "number of dimensions of the variates ($M) must be greater than the dimension of the distribution ($N)"
            ))
        ntuple(i -> size(x, i), Val(N)) == size(d) ||
            throw(DimensionMismatch("inconsistent array dimensions"))
        length(out) == prod(i -> size(x, i), (N + 1):M) ||
            throw(DimensionMismatch("inconsistent array dimensions"))
    end
    return _pdf!(out, d, x)
end

function _pdf!(
    out::AbstractArray{<:Real},
    d::Distribution{<:ArrayLikeVariate},
    x::AbstractArray{<:Real},
)
    @inbounds logpdf!(out, d, x)
    map!(exp, out, out)
    return out
end

"""
    logpdf!(out, d::Distribution{ArrayLikeVariate{N}}, x) where {N}

Evaluate the logarithm of the probability density function of `d` at every element in a
collection `x` and save the results in `out`.

This function checks if the size of `out` is compatible with `d` and `x` and for every
element of `x` if its size is compatible with distribution `d`. These checks can be disabled
by using `@inbounds`.

Here, `x` can be
- an array of dimension `> N` with `size(x)[1:N] == size(d)`, or
- an array of arrays `xi` of dimension `N` with `size(xi) == size(d)`.

# Implementation

Instead of `logpdf!` one should implement `_logpdf!(out, d, x)` which does not have to check
the size of `out` and `x`.

See also: [`pdf!`](@ref).
"""
@inline function logpdf!(
    out::AbstractArray{<:Real},
    d::Distribution{ArrayLikeVariate{N}},
    x::AbstractArray{<:Real,M},
) where {N,M}
    @boundscheck begin
        M > N ||
            throw(DimensionMismatch(
                "number of dimensions of the variates ($M) must be greater than the dimension of the distribution ($N)"
            ))
        ntuple(i -> size(x, i), Val(N)) == size(d) ||
            throw(DimensionMismatch("inconsistent array dimensions"))
        length(out) == prod(i -> size(x, i), (N + 1):M) ||
            throw(DimensionMismatch("inconsistent array dimensions"))
    end
    return _logpdf!(out, d, x)
end

# default definition
function _logpdf!(
    out::AbstractArray{<:Real},
    d::Distribution{<:ArrayLikeVariate},
    x::AbstractArray{<:Real},
)
    @inbounds map!(Base.Fix1(logpdf, d), out, eachvariate(x, variate_form(typeof(d))))
    return out
end

"""
    loglikelihood(d::Distribution{ArrayLikeVariate{N}}, x) where {N}

The log-likelihood of distribution `d` with respect to all variate(s) contained in `x`.

Here, `x` can be any output of `rand(d, dims...)` and `rand!(d, x)`. For instance, `x` can
be
- an array of dimension `N` with `size(x) == size(d)`,
- an array of dimension `N + 1` with `size(x)[1:N] == size(d)`, or
- an array of arrays `xi` of dimension `N` with `size(xi) == size(d)`.
"""
Base.@propagate_inbounds @inline function loglikelihood(
    d::Distribution{ArrayLikeVariate{N}}, x::AbstractArray{<:Real,M},
) where {N,M}
    if M == N
        return logpdf(d, x)
    else
        @boundscheck begin
            M > N ||
                throw(DimensionMismatch(
                    "number of dimensions of the variates ($M) must be greater than or equal to the dimension of the distribution ($N)"
                ))
            ntuple(i -> size(x, i), Val(N)) == size(d) ||
                throw(DimensionMismatch("inconsistent array dimensions"))
        end
        return @inbounds sum(Base.Fix1(logpdf, d), eachvariate(x, ArrayLikeVariate{N}))
    end
end
Base.@propagate_inbounds function loglikelihood(
    d::Distribution{ArrayLikeVariate{N}}, x::AbstractArray{<:AbstractArray{<:Real,N}},
) where {N}
    return sum(Base.Fix1(logpdf, d), x)
end

## TODO: the following types need to be improved
abstract type SufficientStats end
abstract type IncompleteDistribution end

const DistributionType{D<:Distribution} = Type{D}
const IncompleteFormulation = Union{DistributionType,IncompleteDistribution}

"""
    succprob(d::DiscreteUnivariateDistribution)

Get the probability of success.
"""
succprob(d::DiscreteUnivariateDistribution)

"""
    failprob(d::DiscreteUnivariateDistribution)

Get the probability of failure.
"""
failprob(d::DiscreteUnivariateDistribution)

# Temporary fix to handle RFunctions dependencies
"""
    @rand_rdist(::Distribution)

Mark a `Distribution` subtype as requiring RFunction calls. Since these calls
cannot accept an arbitrary random number generator as an input, this macro
creates new `rand(::Distribution, n::Int)` and
`rand!(::Distribution, X::AbstractArray)` functions that call the relevant
RFunction. Calls using another random number generator still work, but rely on
a quantile function to operate.
"""
macro rand_rdist(D)
    esc(quote
        function rand(d::$D, n::Int)
            [rand(d) for i in Base.OneTo(n)]
        end
        function rand!(d::$D, X::AbstractArray)
            for i in eachindex(X)
                X[i] = rand(d)
            end
            return X
        end
    end)
end
