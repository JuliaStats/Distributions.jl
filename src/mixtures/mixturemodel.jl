# finite mixture models

####
#
#  All subtypes of AbstractMixtureModel should implement the following methods:
#
#  - ncomponents(d): the number of components
#
#  - component(d, k):  return the k-th component
#
#  - probs(d):       return a vector of prior probabilities over components.
#

abstract AbstractMixtureModel{VF<:VariateForm,VS<:ValueSupport,C<:Distribution} <: Distribution{VF, VS}

immutable MixtureModel{VF<:VariateForm,VS<:ValueSupport,C<:Distribution} <: AbstractMixtureModel{VF,VS,C}
    components::Vector{C}
    prior::Categorical

    function MixtureModel(cs::Vector{C}, pri::Categorical)
        length(cs) == ncategories(pri) ||
            error("The number of components does not match the length of prior.")
        new(cs, pri)
    end
end

typealias UnivariateMixture{S<:ValueSupport,   C<:Distribution} AbstractMixtureModel{Univariate,S,C}
typealias MultivariateMixture{S<:ValueSupport, C<:Distribution} AbstractMixtureModel{Multivariate,S,C}
typealias MatrixvariateMixture{S<:ValueSupport,C<:Distribution} AbstractMixtureModel{Matrixvariate,S,C}

component_type{VF,VS,C}(d::AbstractMixtureModel{VF,VS,C}) = C

#### Constructors

function MixtureModel{C<:Distribution}(components::Vector{C}, prior::Categorical)
    VF = variate_form(C)
    VS = value_support(C)
    MixtureModel{VF,VS,C}(components, prior)
end

MixtureModel{C<:Distribution}(components::Vector{C}, p::Vector{Float64}) =
    MixtureModel(components, Categorical(p))

# all components have the same prior probabilities
MixtureModel{C<:Distribution}(components::Vector{C}) =
    MixtureModel(components, Categorical(length(components)))

_construct_component{C<:Distribution}(::Type{C}, arg) = C(arg)
_construct_component{C<:Distribution}(::Type{C}, args::Tuple) = C(args...)

function MixtureModel{C<:Distribution}(::Type{C}, params::AbstractArray, p::Vector{Float64})
    components = C[_construct_component(C, a) for a in params]
    MixtureModel(components, p)
end

function MixtureModel{C<:Distribution}(::Type{C}, params::AbstractArray)
    components = C[_construct_component(C, a) for a in params]
    MixtureModel(components)
end


#### Basic properties

length(d::MultivariateMixture) = length(d.components[1])
size(d::MatrixvariateMixture) = size(d.components[1])

ncomponents(d::MixtureModel) = length(d.components)
components(d::MixtureModel) = d.components
component(d::MixtureModel, k::Int) = d.components[k]

probs(d::MixtureModel) = probs(d.prior)

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

function var(d::UnivariateMixture)
    K = ncomponents(d)
    p = probs(d)
    means = Array(Float64, K)
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

function _mixpdf1(d::AbstractMixtureModel, x)
    K = ncomponents(d)
    p = probs(d)
    @assert length(p) == K
    v = 0.0
    for i = 1:K
        @inbounds pi = p[i]
        if pi > 0.0
            c = component(d, i)
            v += pdf(c, x) * pi
        end
    end
    return v
end

function _mixpdf!(r::DenseArray, d::AbstractMixtureModel, x)
    K = ncomponents(d)
    p = probs(d)
    @assert length(p) == K
    fill!(r, 0.0)
    t = Array(Float64, size(r))
    for i = 1:K
        @inbounds pi = p[i]
        if pi > 0.0
            pdf!(t, component(d, i), x)
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
    @assert length(p) == K

    lp = Array(Float64, K)
    m = -Inf   # m <- the maximum of log(p(cs[i], x)) + log(pri[i])
    for i = 1:K
        @inbounds pi = p[i]
        if pi > 0.0
            # lp[i] <- log(p(cs[i], x)) + log(pri[i])
            lp_i = logpdf(component(d, i), x) + log(pi)
            @inbounds lp[i] = lp_i
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

function _mixlogpdf!(r::DenseArray, d::AbstractMixtureModel, x)
    K = ncomponents(d)
    p = probs(d)
    @assert length(p) == K
    n = length(r)
    Lp = Array(Float64, n, K)
    m = fill(-Inf, n)
    for i = 1:K
        @inbounds pi = p[i]
        if pi > 0.0
            lpri = log(pi)
            lp_i = view(Lp, :, i)
            # compute logpdf in batch and store
            logpdf!(lp_i, component(d, i), x)

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

_pdf!(r::AbstractArray, d::UnivariateMixture, x::DenseArray) = _mixpdf!(r, d, x)
_logpdf!(r::AbstractArray, d::UnivariateMixture, x::DenseArray) = _mixlogpdf!(r, d, x)

_pdf(d::MultivariateMixture, x::AbstractVector) = _mixpdf1(d, x)
_logpdf(d::MultivariateMixture, x::AbstractVector) = _mixlogpdf1(d, x)
_pdf!(r::AbstractArray, d::MultivariateMixture, x::DenseMatrix) = _mixpdf!(r, d, x)
_lodpdf!(r::AbstractArray, d::MultivariateMixture, x::DenseMatrix) = _mixlogpdf!(r, d, x)


## component-wise pdf and logpdf

function _cwise_pdf1!(r::StridedVector, d::AbstractMixtureModel, x)
    K = ncomponents(d)
    length(r) == K || error("The length of r should match the number of components.")
    for i = 1:K
        r[i] = pdf(component(d, i), x)
    end
    r
end

function _cwise_logpdf1!(r::StridedVector, d::AbstractMixtureModel, x)
    K = ncomponents(d)
    length(r) == K || error("The length of r should match the number of components.")
    for i = 1:K
        r[i] = logpdf(component(d, i), x)
    end
    r
end

function _cwise_pdf!(r::StridedMatrix, d::AbstractMixtureModel, X)
    K = ncomponents(d)
    n = size(X, ndims(X))
    size(r) == (n, K) || error("The size of r is incorrect.")
    for i = 1:K
        pdf!(view(r,:,i), component(d, i), X)
    end
    r
end

function _cwise_logpdf!(r::StridedMatrix, d::AbstractMixtureModel, X)
    K = ncomponents(d)
    n = size(X, ndims(X))
    size(r) == (n, K) || error("The size of r is incorrect.")
    for i = 1:K
        logpdf!(view(r,:,i), component(d, i), X)
    end
    r
end

componentwise_pdf!(r::StridedVector, d::UnivariateMixture, x::Real) = _cwise_pdf1!(r, d, x)
componentwise_pdf!(r::StridedVector, d::MultivariateMixture, x::AbstractVector) = _cwise_pdf1!(r, d, x)
componentwise_pdf!(r::StridedMatrix, d::UnivariateMixture, x::AbstractVector) = _cwise_pdf!(r, d, x)
componentwise_pdf!(r::StridedMatrix, d::MultivariateMixture, x::AbstractMatrix) = _cwise_pdf!(r, d, x)

componentwise_logpdf!(r::StridedVector, d::UnivariateMixture, x::Real) = _cwise_logpdf1!(r, d, x)
componentwise_logpdf!(r::StridedVector, d::MultivariateMixture, x::AbstractVector) = _cwise_logpdf1!(r, d, x)
componentwise_logpdf!(r::StridedMatrix, d::UnivariateMixture, x::AbstractVector) = _cwise_logpdf!(r, d, x)
componentwise_logpdf!(r::StridedMatrix, d::MultivariateMixture, x::AbstractMatrix) = _cwise_logpdf!(r, d, x)

componentwise_pdf(d::UnivariateMixture, x::Real) = componentwise_pdf!(Array(Float64, ncomponents(d)), d, x)
componentwise_pdf(d::UnivariateMixture, x::AbstractVector) = componentwise_pdf!(Array(Float64, length(x), ncomponents(d)), d, x)
componentwise_pdf(d::MultivariateMixture, x::AbstractVector) = componentwise_pdf!(Array(Float64, ncomponents(d)), d, x)
componentwise_pdf(d::MultivariateMixture, x::AbstractMatrix) = componentwise_pdf!(Array(Float64, size(x,2), ncomponents(d)), d, x)

componentwise_logpdf(d::UnivariateMixture, x::Real) = componentwise_logpdf!(Array(Float64, ncomponents(d)), d, x)
componentwise_logpdf(d::UnivariateMixture, x::AbstractVector) = componentwise_logpdf!(Array(Float64, length(x), ncomponents(d)), d, x)
componentwise_logpdf(d::MultivariateMixture, x::AbstractVector) = componentwise_logpdf!(Array(Float64, ncomponents(d)), d, x)
componentwise_logpdf(d::MultivariateMixture, x::AbstractMatrix) = componentwise_logpdf!(Array(Float64, size(x,2), ncomponents(d)), d, x)


## Sampling

immutable MixtureSampler{VF,VS,Sampler} <: Sampleable{VF,VS}
    csamplers::Vector{Sampler}
    psampler::AliasTable
end

function MixtureSampler{VF,VS}(d::MixtureModel{VF,VS})
    csamplers = map(sampler, d.components)
    psampler = sampler(d.prior)
    MixtureSampler{VF,VS,eltype(csamplers)}(csamplers, psampler)
end


rand(d::MixtureModel) = rand(component(d, rand(d.prior)))

rand(s::MixtureSampler) = rand(s.csamplers[rand(s.psampler)])
_rand!(s::MixtureSampler{Multivariate}, x::DenseVector) = _rand!(s.csamplers[rand(s.psampler)], x)

sampler(d::MixtureModel) = MixtureSampler(d)
