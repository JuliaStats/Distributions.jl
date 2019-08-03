## sample space/domain

"""
`F <: VariateForm` specifies the form of the variate or
dimension of a sample, univariate (scalar), multivariate (vector), matrix-variate (matrix).
"""
abstract type VariateForm end
struct Univariate    <: VariateForm end
struct Multivariate  <: VariateForm end
struct Matrixvariate <: VariateForm end

"""
`S <: ValueSupport` specifies the support of sample elements,
either discrete or continuous.
"""
abstract type ValueSupport{N} end
struct ContinuousSupport{N <: Number} <: ValueSupport{N} end
abstract type CountableSupport{C} <: ValueSupport{C} end
struct ContiguousSupport{C <: Integer} <: CountableSupport{C} end
struct UnionSupport{N1, N2,
                    S1 <: ValueSupport{N1},
                    S2 <: ValueSupport{N2}} <:
                        ValueSupport{Union{N1, N2}} end

const DiscontinuousSupport{I, F} =
    UnionSupport{I, F, <: CountableSupport{I},
                 ContinuousSupport{F}} where {I <: Number, F <: Number}

## Sampleable

"""
    Sampleable{F<:VariateForm,S<:ValueSupport}

`Sampleable` is any type able to produce random values.
Parametrized by a `VariateForm` defining the dimension of samples
and a `ValueSupport` defining the domain of possibly sampled values.
Any `Sampleable` implements the `Base.rand` method.
"""
abstract type Sampleable{F<:VariateForm,S<:ValueSupport} end

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
    eltype(s::Sampleable)
    eltype(::ValueSupport)

The default element type of a sample. This is the type of elements of the samples generated
by the `rand` method. However, one can provide an array of different element types to
store the samples using `rand!`.
"""
Base.eltype(::Sampleable{F, <: ValueSupport{N}}) where {F, N} = N
Base.eltype(::Type{<:Sampleable{F, <: ValueSupport{N}}}) where {F, N} = N
Base.eltype(::Type{<:ValueSupport{N}}) where {N} = N

"""
    nsamples(s::Sampleable)

The number of values contained in one sample of `s`. Multiple samples are often organized
into an array, depending on the variate form.
"""
nsamples(t::Type{Sampleable}, x::Any)
nsamples(::Type{D}, x::Number) where {D<:Sampleable{Univariate}} = 1
nsamples(::Type{D}, x::AbstractArray) where {D<:Sampleable{Univariate}} = length(x)
nsamples(::Type{D}, x::AbstractArray{<:AbstractVector}) where {D<:Sampleable{Multivariate}} = length(x)
nsamples(::Type{D}, x::AbstractVector{<:Number}) where {D<:Sampleable{Multivariate}} = 1
nsamples(::Type{D}, x::AbstractMatrix) where {D<:Sampleable{Multivariate}} = size(x, 2)
nsamples(::Type{D}, x::AbstractMatrix{<:Number}) where {D<:Sampleable{Matrixvariate}} = 1
nsamples(::Type{D}, x::AbstractArray{<:AbstractMatrix{T}}) where {D<:Sampleable{Matrixvariate},T<:Number} = length(x)

"""
    Distribution{F<:VariateForm,S<:ValueSupport} <: Sampleable{F,S}

`Distribution` is a `Sampleable` generating random values from a probability
distribution. Distributions define a Probability Distribution Function (PDF)
to implement with `pdf` and a Cumulated Distribution Function (CDF) to implement
with `cdf`.
"""
abstract type Distribution{F<:VariateForm,S<:ValueSupport} <: Sampleable{F,S} end

const UnivariateDistribution{S<:ValueSupport}   = Distribution{Univariate,S}
const MultivariateDistribution{S<:ValueSupport} = Distribution{Multivariate,S}
const MatrixDistribution{S<:ValueSupport}       = Distribution{Matrixvariate,S}

const CountableDistribution{F<:VariateForm,
                            C<:CountableSupport} = Distribution{F,C}
const ContiguousDistribution{F<:VariateForm, S<:Integer} =
    CountableDistribution{F,ContiguousSupport{S}}
const ContinuousDistribution{F<:VariateForm, T<:Number} =
    Distribution{F,ContinuousSupport{T}}

const CountableUnivariateDistribution{C<:CountableSupport} =
    UnivariateDistribution{C}
const ContiguousUnivariateDistribution{S<:Integer} =
    CountableUnivariateDistribution{ContiguousSupport{S}}
const ContinuousUnivariateDistribution{T<:Number} =
    UnivariateDistribution{ContinuousSupport{T}}

const CountableMultivariateDistribution{C<:CountableSupport} =
    MultivariateDistribution{C}
const ContiguousMultivariateDistribution{S<:Integer} =
    CountableMultivariateDistribution{ContiguousSupport{S}}
const ContinuousMultivariateDistribution{T<:Number} =
    MultivariateDistribution{ContinuousSupport{T}}

const CountableMatrixDistribution{C<:CountableSupport} =
    MatrixDistribution{C}
const ContiguousMatrixDistribution{S<:Integer} =
    CountableMatrixDistribution{ContiguousSupport{S}}
const ContinuousMatrixDistribution{T<:Number} =
    MatrixDistribution{ContinuousSupport{T}}

variate_form(::Type{<:Sampleable{VF, <:ValueSupport}}) where {VF<:VariateForm} = VF
value_support(::Type{<:Sampleable{<:VariateForm,VS}}) where {VS<:ValueSupport} = VS

# allow broadcasting over distribution objects
# to be decided: how to handle multivariate/matrixvariate distributions?
Broadcast.broadcastable(d::UnivariateDistribution) = Ref(d)


## TODO: the following types need to be improved
abstract type SufficientStats end
abstract type IncompleteDistribution end

const DistributionType{D<:Distribution} = Type{D}
const IncompleteFormulation = Union{DistributionType,IncompleteDistribution}

"""
    succprob(d::ContiguousUnivariateDistribution)

Get the probability of success.
"""
succprob(d::ContiguousUnivariateDistribution)

"""
    failprob(d::ContiguousUnivariateDistribution)

Get the probability of failure.
"""
failprob(d::ContiguousUnivariateDistribution)

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
