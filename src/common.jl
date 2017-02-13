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

@compat const UnivariateDistribution{S<:ValueSupport}   = Distribution{Univariate,S}
@compat const MultivariateDistribution{S<:ValueSupport} = Distribution{Multivariate,S}
@compat const MatrixDistribution{S<:ValueSupport}       = Distribution{Matrixvariate,S}
const NonMatrixDistribution = Union{UnivariateDistribution, MultivariateDistribution}

@compat const DiscreteDistribution{F<:VariateForm}   = Distribution{F,Discrete}
@compat const ContinuousDistribution{F<:VariateForm} = Distribution{F,Continuous}

const DiscreteUnivariateDistribution     = Distribution{Univariate,    Discrete}
const ContinuousUnivariateDistribution   = Distribution{Univariate,    Continuous}
const DiscreteMultivariateDistribution   = Distribution{Multivariate,  Discrete}
const ContinuousMultivariateDistribution = Distribution{Multivariate,  Continuous}
const DiscreteMatrixDistribution         = Distribution{Matrixvariate, Discrete}
const ContinuousMatrixDistribution       = Distribution{Matrixvariate, Continuous}

variate_form{VF<:VariateForm,VS<:ValueSupport}(::Type{Distribution{VF,VS}}) = VF
variate_form{T<:Distribution}(::Type{T}) = variate_form(supertype(T))

value_support{VF<:VariateForm,VS<:ValueSupport}(::Type{Distribution{VF,VS}}) = VS
value_support{T<:Distribution}(::Type{T}) = value_support(supertype(T))

## TODO: the following types need to be improved
@compat abstract type SufficientStats end
@compat abstract type IncompleteDistribution end

@compat const DistributionType{D<:Distribution} = Type{D}
const IncompleteFormulation = Union{DistributionType,IncompleteDistribution}
