## sample space/domain

@compat abstract type VariateForm end
type Univariate    <: VariateForm end
type Multivariate  <: VariateForm end
type Matrixvariate <: VariateForm end

@compat abstract type ValueSupport end
type Discrete   <: ValueSupport end
type Continuous <: ValueSupport end

Base.eltype(::Type{Discrete}) = Int
Base.eltype(::Type{Continuous}) = Float64

## Sampleable

@compat abstract type Sampleable{F<:VariateForm,S<:ValueSupport} end

Base.length(::Sampleable{Univariate}) = 1
Base.length(s::Sampleable{Multivariate}) = throw(MethodError(length, (s,)))
Base.length(s::Sampleable) = prod(size(s))

Base.size(s::Sampleable{Univariate}) = ()
Base.size(s::Sampleable{Multivariate}) = (length(s),)

Base.eltype{F,S}(s::Sampleable{F,S}) = eltype(S)
Base.eltype{F}(s::Sampleable{F,Discrete}) = Int
Base.eltype{F}(s::Sampleable{F,Continuous}) = Float64

nsamples{D<:Sampleable{Univariate}}(::Type{D}, x::Number) = 1
nsamples{D<:Sampleable{Univariate}}(::Type{D}, x::AbstractArray) = length(x)
nsamples{D<:Sampleable{Multivariate}}(::Type{D}, x::AbstractVector) = 1
nsamples{D<:Sampleable{Multivariate}}(::Type{D}, x::AbstractMatrix) = size(x, 2)
nsamples{D<:Sampleable{Matrixvariate}}(::Type{D}, x::Number) = 1
nsamples{D<:Sampleable{Matrixvariate},T<:Number}(::Type{D}, x::Array{Matrix{T}}) = length(x)

## Distribution <: Sampleable

@compat abstract type Distribution{F<:VariateForm,S<:ValueSupport} <: Sampleable{F,S} end

typealias UnivariateDistribution{S<:ValueSupport}   Distribution{Univariate,S}
typealias MultivariateDistribution{S<:ValueSupport} Distribution{Multivariate,S}
typealias MatrixDistribution{S<:ValueSupport}       Distribution{Matrixvariate,S}
typealias NonMatrixDistribution Union{UnivariateDistribution, MultivariateDistribution}

typealias DiscreteDistribution{F<:VariateForm}   Distribution{F,Discrete}
typealias ContinuousDistribution{F<:VariateForm} Distribution{F,Continuous}

typealias DiscreteUnivariateDistribution     Distribution{Univariate,    Discrete}
typealias ContinuousUnivariateDistribution   Distribution{Univariate,    Continuous}
typealias DiscreteMultivariateDistribution   Distribution{Multivariate,  Discrete}
typealias ContinuousMultivariateDistribution Distribution{Multivariate,  Continuous}
typealias DiscreteMatrixDistribution         Distribution{Matrixvariate, Discrete}
typealias ContinuousMatrixDistribution       Distribution{Matrixvariate, Continuous}

variate_form{VF<:VariateForm,VS<:ValueSupport}(::Type{Distribution{VF,VS}}) = VF
variate_form{T<:Distribution}(::Type{T}) = variate_form(supertype(T))

value_support{VF<:VariateForm,VS<:ValueSupport}(::Type{Distribution{VF,VS}}) = VS
value_support{T<:Distribution}(::Type{T}) = value_support(supertype(T))

## TODO: the following types need to be improved
@compat abstract type SufficientStats end
@compat abstract type IncompleteDistribution end

typealias DistributionType{D<:Distribution} Type{D}
typealias IncompleteFormulation Union{DistributionType,IncompleteDistribution}
