## sample space/domain

abstract VariateForm
type Univariate    <: VariateForm end
type Multivariate  <: VariateForm end
type Matrixvariate <: VariateForm end

abstract ValueSupport
type Discrete   <: ValueSupport end
type Continuous <: ValueSupport end

Base.eltype(::Type{Discrete}) = Int
Base.eltype(::Type{Continuous}) = Float64

## Sampleable

abstract Sampleable{F<:VariateForm,S<:ValueSupport}

Base.length(::Sampleable{Univariate}) = 1
Base.length(s::Sampleable{Multivariate}) = throw(MethodError(length, (s,)))
Base.length(s::Sampleable) = prod(size(s))

Base.size(s::Sampleable{Univariate}) = ()
Base.size(s::Sampleable{Multivariate}) = (length(s),)

nsamples{D<:Sampleable{Univariate}}(::Type{D}, x::AbstractArray) = length(x)
nsamples{D<:Sampleable{Multivariate}}(::Type{D}, x::AbstractVector) = 1
nsamples{D<:Sampleable{Multivariate}}(::Type{D}, x::AbstractMatrix) = size(x, 2)
nsamples{D<:Sampleable{Matrixvariate},T}(::Type{D}, x::Array{Matrix{T}}) = length(x)

## Distribution <: Sampleable

abstract Distribution{F<:VariateForm,S<:ValueSupport} <: Sampleable{F,S}

typealias UnivariateDistribution{S<:ValueSupport}   Distribution{Univariate,S}
typealias MultivariateDistribution{S<:ValueSupport} Distribution{Multivariate,S}
typealias MatrixDistribution{S<:ValueSupport}       Distribution{Matrixvariate,S}
typealias NonMatrixDistribution Union(UnivariateDistribution, MultivariateDistribution)

typealias DiscreteDistribution{F<:VariateForm}   Distribution{F,Discrete}
typealias ContinuousDistribution{F<:VariateForm} Distribution{F,Continuous}

typealias DiscreteUnivariateDistribution     Distribution{Univariate,    Discrete}
typealias ContinuousUnivariateDistribution   Distribution{Univariate,    Continuous}
typealias DiscreteMultivariateDistribution   Distribution{Multivariate,  Discrete}
typealias ContinuousMultivariateDistribution Distribution{Multivariate,  Continuous}
typealias DiscreteMatrixDistribution         Distribution{Matrixvariate, Discrete}
typealias ContinuousMatrixDistribution       Distribution{Matrixvariate, Continuous}

variate_form{VF<:VariateForm,VS<:ValueSupport}(::Type{Distribution{VF,VS}}) = VF
variate_form{T<:Distribution}(::Type{T}) = variate_form(super(T))

value_support{VF<:VariateForm,VS<:ValueSupport}(::Type{Distribution{VF,VS}}) = VS
value_support{T<:Distribution}(::Type{T}) = value_support(super(T))


## TODO: replace dim with length in specialized methods
Base.length(d::MultivariateDistribution) = dim(d)
Base.size(d::MatrixDistribution) = (dim(d), dim(d)) # override if the matrix isn't square

## TODO: the following types need to be improved
abstract SufficientStats
abstract IncompleteDistribution

typealias DistributionType{D<:Distribution} Type{D}
typealias IncompleteFormulation Union(DistributionType,IncompleteDistribution)

