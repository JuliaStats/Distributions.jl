# finite mixture models

"""

  All subtypes of `AbstractMixtureModel` should implement the following methods:

  - ncomponents(d): the number of components

  - component(d, k):  return the k-th component

  - probs(d):       return a vector of prior probabilities over components.
"""
abstract type AbstractMixtureModel{VF<:VariateForm,VS<:ValueSupport,C<:Distribution} <: Distribution{VF, VS} end

"""
MixtureModel{VF<:VariateForm,VS<:ValueSupport,C<:Distribution,CT<:Real}
A mixture of distributions, parametrized on:
* `VF,VS` variate and support
* `C` distribution family of the mixture
* `CT` the type for probabilities of the prior
"""
struct MixtureModel{VF<:VariateForm,VS<:ValueSupport,C<:Distribution,CT<:Real} <: AbstractMixtureModel{VF,VS,C}
    components::Vector{C}
    prior::Categorical{CT}

    function MixtureModel{VF,VS,C}(cs::Vector{C}, pri::Categorical{CT}) where {VF,VS,C,CT}
        length(cs) == ncategories(pri) ||
            error("The number of components does not match the length of prior.")
        new{VF,VS,C,CT}(cs, pri)
    end
end

const UnivariateMixture{S<:ValueSupport,   C<:Distribution} = AbstractMixtureModel{Univariate,S,C}
const MultivariateMixture{S<:ValueSupport, C<:Distribution} = AbstractMixtureModel{Multivariate,S,C}
const MatrixvariateMixture{S<:ValueSupport,C<:Distribution} = AbstractMixtureModel{Matrixvariate,S,C}

# Interface

"""
    component_type(d::AbstractMixtureModel)

The type of the components of `d`.
"""
component_type(d::AbstractMixtureModel{VF,VS,C}) where {VF,VS,C} = C

"""
    components(d::AbstractMixtureModel)

Get a list of components of the mixture model `d`.
"""
components(d::AbstractMixtureModel) = [component(d, k) for k in 1:ncomponents(d)]

"""
    probs(d::AbstractMixtureModel)

Get the vector of prior probabilities of all components of `d`.
"""
probs(d::AbstractMixtureModel)

"""
    mean(d::Union{UnivariateMixture, MultivariateMixture})

Compute the overall mean (expectation).
"""
mean(d::AbstractMixtureModel)

"""
    insupport(d::MultivariateMixture, x)

Evaluate whether `x` is within the support of mixture distribution `d`.
"""
insupport(d::AbstractMixtureModel, x::AbstractVector)

"""
    pdf(d::Union{UnivariateMixture, MultivariateMixture}, x)

Evaluate the (mixed) probability density function over `x`. Here, `x` can be a single
sample or an array of multiple samples.
"""
pdf(d::AbstractMixtureModel, x::Any)

"""
    logpdf(d::Union{UnivariateMixture, MultivariateMixture}, x)

Evaluate the logarithm of the (mixed) probability density function over `x`.
Here, `x` can be a single sample or an array of multiple samples.
"""
logpdf(d::AbstractMixtureModel, x::Any)

"""
    rand(d::Union{UnivariateMixture, MultivariateMixture})

Draw a sample from the mixture model `d`.

    rand(d::Union{UnivariateMixture, MultivariateMixture}, n)

Draw `n` samples from `d`.
"""
rand(d::AbstractMixtureModel)

"""
    rand!(d::Union{UnivariateMixture, MultivariateMixture}, r::AbstactArray)

Draw multiple samples from `d` and write them to `r`.
"""
rand!(d::AbstractMixtureModel, r::AbstractArray)


#### Constructors

"""
    MixtureModel(components, [prior])

Construct a mixture model with a vector of `components` and a `prior` probability vector.
If no `prior` is provided then all components will have the same prior probabilities.
"""
MixtureModel(components::Vector{C}) where {C<:Distribution} =
    MixtureModel(components, Categorical(length(components)))

"""
    MixtureModel(C, params, [prior])

Construct a mixture model with component type ``C``, a vector of parameters for constructing
the components given by ``params``, and a prior probability vector.
If no `prior` is provided then all components will have the same prior probabilities.
"""
function MixtureModel(::Type{C}, params::AbstractArray) where C<:Distribution
    components = C[_construct_component(C, a) for a in params]
    MixtureModel(components)
end

function MixtureModel(components::Vector{C}, prior::Categorical) where C<:Distribution
    VF = variate_form(C)
    VS = value_support(C)
    MixtureModel{VF,VS,C}(components, prior)
end

MixtureModel(components::Vector{C}, p::VT) where {C<:Distribution,VT<:AbstractVector{<:Real}} =
    MixtureModel(components, Categorical(p))

_construct_component(::Type{C}, arg) where {C<:Distribution} = C(arg)
_construct_component(::Type{C}, args::Tuple) where {C<:Distribution} = C(args...)

function MixtureModel(::Type{C}, params::AbstractArray, p::Vector{T}) where {C<:Distribution,T<:Real}
    components = C[_construct_component(C, a) for a in params]
    MixtureModel(components, p)
end




#### Basic properties

"""
    length(d::MultivariateMixture)

The length of each sample (only for `Multivariate`).
"""
length(d::MultivariateMixture) = length(d.components[1])
size(d::MatrixvariateMixture) = size(d.components[1])

ncomponents(d::MixtureModel) = length(d.components)
components(d::MixtureModel) = d.components
component(d::MixtureModel, k::Int) = d.components[k]

probs(d::MixtureModel) = probs(d.prior)
params(d::MixtureModel) = ([params(c) for c in d.components], params(d.prior)[1])
partype(d::MixtureModel) = promote_type(partype(d.prior), map(partype, d.components)...)

minimum(d::MixtureModel) = minimum([minimum(dci) for dci in d.components])
maximum(d::MixtureModel) = maximum([maximum(dci) for dci in d.components])

function mean(d::UnivariateMixture)
    K = ncomponents(d)
    p = probs(d)
    m = 0.0
    for i = 1:K
        pi = p[i]
        if pi > 0.0
            c = component(d, i)
            m += mean(c) * pi
        end
    end
    return m
end

function mean(d::MultivariateMixture)
    K = ncomponents(d)
    p = probs(d)
    m = zeros(length(d))
    for i = 1:K
        pi = p[i]
        if pi > 0.0
            c = component(d, i)
            BLAS.axpy!(pi, mean(c), m)
        end
    end
    return m
end

"""
    var(d::UnivariateMixture)

Compute the overall variance (only for ``UnivariateMixture``).
"""
function var(d::UnivariateMixture)
    K = ncomponents(d)
    p = probs(d)
    means = Vector{Float64}(undef, K)
    m = 0.0
    v = 0.0
    for i = 1:K
        pi = p[i]
        if pi > 0.0
            ci = component(d, i)
            means[i] = mi = mean(ci)
            m += pi * mi
            v += pi * var(ci)
        end
    end
    for i = 1:K
        pi = p[i]
        if pi > 0.0
            v += pi * abs2(means[i] - m)
        end
    end
    return v
end

function cov(d::MultivariateMixture)
    K = ncomponents(d)
    p = probs(d)
    m = zeros(length(d))
    md = zeros(length(d))
    V = zeros(length(d),length(d))

    for i = 1:K
        pi = p[i]
        if pi > 0.0
            c = component(d, i)
            BLAS.axpy!(pi, mean(c), m)
            BLAS.axpy!(pi, cov(c), V)
        end
    end
    for i = 1:K
        pi = p[i]
        if pi > 0.0
            c = component(d, i)
            # todo: use more in-place operations
            md = mean(c) - m
            BLAS.axpy!(pi, md*md', V)
        end
    end
    return V
end



#### show

function show(io::IO, d::MixtureModel)
    K = ncomponents(d)
    pr = probs(d)
    println(io, "MixtureModel{$(component_type(d))}(K = $K)")
    Ks = min(K, 8)
    for i = 1:Ks
        @printf(io, "components[%d] (prior = %.4f): ", i, pr[i])
        println(io, component(d, i))
    end
    if Ks < K
        println(io, "The rest are omitted ...")
    end
end



#### Evaluation

function insupport(d::AbstractMixtureModel, x::AbstractVector)
    K = ncomponents(d)
    p = probs(d)
    @inbounds for i in eachindex(p)
        pi = p[i]
        if pi > 0.0 && insupport(component(d, i), x)
            return true
        end
    end
    return false
end

function _cdf(d::UnivariateMixture, x::Real)
    K = ncomponents(d)
    p = probs(d)
    r = 0.0
    @inbounds for i in eachindex(p)
        pi = p[i]
        if pi > 0.0
            c = component(d, i)
            r += pi * cdf(c, x)
        end
    end
    return r
end

cdf(d::UnivariateMixture{Continuous}, x::Real) = _cdf(d, x)
cdf(d::UnivariateMixture{Discrete}, x::Integer) = _cdf(d, x)

function _mixpdf1(d::AbstractMixtureModel, x)
    ps = probs(d)
    cs = components(d)
    return sum((ps[i] > 0) * (ps[i] * pdf(cs[i], x)) for i in eachindex(ps))
end

function _mixpdf!(r::AbstractArray, d::AbstractMixtureModel, x)
    K = ncomponents(d)
    p = probs(d)
    fill!(r, 0.0)
    t = Array{eltype(p)}(undef, size(r))
    @inbounds for i in eachindex(p)
        pi = p[i]
        if pi > 0.0
            if d isa UnivariateMixture
                t .= pdf.(component(d, i), x)
            else
                pdf!(t, component(d, i), x)
            end
            BLAS.axpy!(pi, t, r)
        end
    end
    return r
end

function _mixlogpdf1(d::AbstractMixtureModel, x)
    # using the formula below for numerical stability
    #
    # logpdf(d, x) = log(sum_i pri[i] * pdf(cs[i], x))
    #              = log(sum_i pri[i] * exp(logpdf(cs[i], x)))
    #              = log(sum_i exp(logpri[i] + logpdf(cs[i], x)))
    #              = m + log(sum_i exp(logpri[i] + logpdf(cs[i], x) - m))
    #
    #  m is chosen to be the maximum of logpri[i] + logpdf(cs[i], x)
    #  such that the argument of exp is in a reasonable range
    #

    K = ncomponents(d)
    p = probs(d)
    lp = Vector{eltype(p)}(undef, K)
    m = -Inf   # m <- the maximum of log(p(cs[i], x)) + log(pri[i])
    @inbounds for i in eachindex(p)
        pi = p[i]
        if pi > 0.0
            # lp[i] <- log(p(cs[i], x)) + log(pri[i])
            lp_i = logpdf(component(d, i), x) + log(pi)
            lp[i] = lp_i
            if lp_i > m
                m = lp_i
            end
        end
    end
    v = 0.0
    @inbounds for i = 1:K
        if p[i] > 0.0
            v += exp(lp[i] - m)
        end
    end
    return m + log(v)
end

function _mixlogpdf!(r::AbstractArray, d::AbstractMixtureModel, x)
    K = ncomponents(d)
    p = probs(d)
    n = length(r)
    Lp = Matrix{eltype(p)}(undef, n, K)
    m = fill(-Inf, n)
    @inbounds for i in eachindex(p)
        pi = p[i]
        if pi > 0.0
            lpri = log(pi)
            lp_i = view(Lp, :, i)
            # compute logpdf in batch and store
            if d isa UnivariateMixture
                lp_i .= logpdf.(component(d, i), x)
            else
                logpdf!(lp_i, component(d, i), x)
            end


            # in the mean time, add log(prior) to lp and
            # update the maximum for each sample
            for j = 1:n
                lp_i[j] += lpri
                if lp_i[j] > m[j]
                    m[j] = lp_i[j]
                end
            end
        end
    end

    fill!(r, 0.0)
    @inbounds for i = 1:K
        if p[i] > 0.0
            lp_i = view(Lp, :, i)
            for j = 1:n
                r[j] += exp(lp_i[j] - m[j])
            end
        end
    end

    @inbounds for j = 1:n
        r[j] = log(r[j]) + m[j]
    end
    return r
end

pdf(d::UnivariateMixture{Continuous}, x::Real) = _mixpdf1(d, x)
pdf(d::UnivariateMixture{Discrete}, x::Int) = _mixpdf1(d, x)
logpdf(d::UnivariateMixture{Continuous}, x::Real) = _mixlogpdf1(d, x)
logpdf(d::UnivariateMixture{Discrete}, x::Int) = _mixlogpdf1(d, x)

_pdf!(r::AbstractArray, d::UnivariateMixture{Discrete}, x::UnitRange) = _mixpdf!(r, d, x)
_pdf!(r::AbstractArray, d::UnivariateMixture, x::AbstractArray) = _mixpdf!(r, d, x)
_logpdf!(r::AbstractArray, d::UnivariateMixture, x::AbstractArray) = _mixlogpdf!(r, d, x)

_pdf(d::MultivariateMixture, x::AbstractVector) = _mixpdf1(d, x)
_logpdf(d::MultivariateMixture, x::AbstractVector) = _mixlogpdf1(d, x)
_pdf!(r::AbstractArray, d::MultivariateMixture, x::AbstractMatrix) = _mixpdf!(r, d, x)
_lodpdf!(r::AbstractArray, d::MultivariateMixture, x::AbstractMatrix) = _mixlogpdf!(r, d, x)


## component-wise pdf and logpdf

function _cwise_pdf1!(r::AbstractVector, d::AbstractMixtureModel, x)
    K = ncomponents(d)
    length(r) == K || error("The length of r should match the number of components.")
    for i = 1:K
        r[i] = pdf(component(d, i), x)
    end
    r
end

function _cwise_logpdf1!(r::AbstractVector, d::AbstractMixtureModel, x)
    K = ncomponents(d)
    length(r) == K || error("The length of r should match the number of components.")
    for i = 1:K
        r[i] = logpdf(component(d, i), x)
    end
    r
end

function _cwise_pdf!(r::AbstractMatrix, d::AbstractMixtureModel, X)
    K = ncomponents(d)
    n = size(X, ndims(X))
    size(r) == (n, K) || error("The size of r is incorrect.")
    for i = 1:K
        if d isa UnivariateMixture
            view(r,:,i) .= pdf.(Ref(component(d, i)), X)
        else
            pdf!(view(r,:,i),component(d, i), X)
        end
    end
    r
end

function _cwise_logpdf!(r::AbstractMatrix, d::AbstractMixtureModel, X)
    K = ncomponents(d)
    n = size(X, ndims(X))
    size(r) == (n, K) || error("The size of r is incorrect.")
    for i = 1:K
        if d isa UnivariateMixture
            view(r,:,i) .= logpdf.(Ref(component(d, i)), X)
        else
            logpdf!(view(r,:,i), component(d, i), X)
        end
    end
    r
end

componentwise_pdf!(r::AbstractVector, d::UnivariateMixture, x::Real) = _cwise_pdf1!(r, d, x)
componentwise_pdf!(r::AbstractVector, d::MultivariateMixture, x::AbstractVector) = _cwise_pdf1!(r, d, x)
componentwise_pdf!(r::AbstractMatrix, d::UnivariateMixture, x::AbstractVector) = _cwise_pdf!(r, d, x)
componentwise_pdf!(r::AbstractMatrix, d::MultivariateMixture, x::AbstractMatrix) = _cwise_pdf!(r, d, x)

componentwise_logpdf!(r::AbstractVector, d::UnivariateMixture, x::Real) = _cwise_logpdf1!(r, d, x)
componentwise_logpdf!(r::AbstractVector, d::MultivariateMixture, x::AbstractVector) = _cwise_logpdf1!(r, d, x)
componentwise_logpdf!(r::AbstractMatrix, d::UnivariateMixture, x::AbstractVector) = _cwise_logpdf!(r, d, x)
componentwise_logpdf!(r::AbstractMatrix, d::MultivariateMixture, x::AbstractMatrix) = _cwise_logpdf!(r, d, x)

componentwise_pdf(d::UnivariateMixture, x::Real) = componentwise_pdf!(Vector{eltype(x)}(undef, ncomponents(d)), d, x)
componentwise_pdf(d::UnivariateMixture, x::AbstractVector) = componentwise_pdf!(Matrix{eltype(x)}(undef, length(x), ncomponents(d)), d, x)
componentwise_pdf(d::MultivariateMixture, x::AbstractVector) = componentwise_pdf!(Vector{eltype(x)}(undef, ncomponents(d)), d, x)
componentwise_pdf(d::MultivariateMixture, x::AbstractMatrix) = componentwise_pdf!(Matrix{eltype(x)}(undef, size(x,2), ncomponents(d)), d, x)

componentwise_logpdf(d::UnivariateMixture, x::Real) = componentwise_logpdf!(Vector{eltype(x)}(undef, ncomponents(d)), d, x)
componentwise_logpdf(d::UnivariateMixture, x::AbstractVector) = componentwise_logpdf!(Matrix{eltype(x)}(undef, length(x), ncomponents(d)), d, x)
componentwise_logpdf(d::MultivariateMixture, x::AbstractVector) = componentwise_logpdf!(Vector{eltype(x)}(undef, ncomponents(d)), d, x)
componentwise_logpdf(d::MultivariateMixture, x::AbstractMatrix) = componentwise_logpdf!(Matrix{eltype(x)}(undef, size(x,2), ncomponents(d)), d, x)


## Sampling

struct MixtureSampler{VF,VS,Sampler} <: Sampleable{VF,VS}
    csamplers::Vector{Sampler}
    psampler::AliasTable
end

function MixtureSampler(d::MixtureModel{VF,VS}) where {VF,VS}
    csamplers = map(sampler, d.components)
    psampler = sampler(d.prior)
    MixtureSampler{VF,VS,eltype(csamplers)}(csamplers, psampler)
end

rand(rng::AbstractRNG, s::MixtureSampler{Univariate}) =
    rand(rng, s.csamplers[rand(rng, s.psampler)])
rand(rng::AbstractRNG, d::MixtureModel{Univariate}) =
    rand(rng, component(d, rand(rng, d.prior)))

# multivariate mixture sampler for a vector
_rand!(rng::AbstractRNG, s::MixtureSampler{Multivariate}, x::AbstractVector) =
    _rand!(rng, s.csamplers[rand(rng, s.psampler)], x)
_rand!(rng::AbstractRNG, s::MixtureModel{Multivariate}, x::AbstractVector) =
    _rand!(rng, sampler(s), x)

sampler(d::MixtureModel) = MixtureSampler(d)
