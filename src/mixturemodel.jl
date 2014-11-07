# finite mixture models

immutable MixtureModel{VF<:VariateForm, VS<:ValueSupport, Component<:Distribution} <: Distribution{VF,VS}
    components::Vector{Component}
    prior::Categorical
end

typealias UnivariateMixture    MixtureModel{Univariate} 
typealias MultivariateMixture  MixtureModel{Multivariate}
typealias MatrixvariateMixture MixtureModel{Matrixvariate}


#### Constructors

function MixtureModel{C<:Distribution}(components::Vector{C}, prior::Categorical)
    length(components) == ncategories(prior) ||
        error("Inconsistent sizes of components and prior.")
    VF = variate_form(C)
    VS = value_support(C)
    MixtureModel{VF,VS,C}(components, prior)
end

MixtureModel{C<:Distribution}(components::Vector{C}, p::Vector{Float64}) = 
    MixtureModel(components, Categorical(p))

# all components have the same prior probabilities
MixtureModel{C<:Distribution}(components::Vector{C}) = 
    MixtureModel(components, Categorical(length(components)))


#### Basic properties

length(d::MultivariateMixture) = length(d.components[1])
size(d::MatrixvariateMixture) = size(d.components[1])

components(d::MixtureModel) = d.components
priorprobs(d::MixtureModel) = d.prior.prob

function mean(d::UnivariateMixture)
    cs = components(d)
    p = priorprobs(d)
    m = 0.0
    for i = 1:length(cs)
        pi = p[i]
        if pi > 0.0
            m += mean(cs[i]) * pi
        end
    end
    return m
end

function mean(d::MultivariateMixture)
    cs = components(d)
    p = priorprobs(d)
    m = zeros(length(d))
    for i = 1:length(cs)
        pi = p[i]
        if pi > 0.0
            BLAS.axpy!(pi, mean(cs[i]), m)
        end
    end
    return m
end

function var(d::UnivariateMixture)
    cs = components(d)
    p = priorprobs(d)
    K = length(cs)
    means = Array(Float64, K)
    m = 0.0
    v = 0.0
    for i = 1:K
        pi = p[i]
        if pi > 0.0
            ci = cs[i]        
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


#### Evaluation

function _mixpdf1(d::MixtureModel, x)
    cs = components(d)
    p = priorprobs(d)
    v = 0.0
    for i = 1:length(cs)
        pi = p[i]
        if pi > 0.0
            v += pdf(cs[i], x) * pi
        end
    end
    return v
end

function _mixpdf!(r::DenseArray, d::MixtureModel, x)
    cs = components(d)
    p = priorprobs(d)
    fill!(r, 0.0)
    t = Array(Float64, size(r))
    for i = 1:length(cs)
        pi = p[i]
        if pi > 0.0
            # compute pdf in batch
            pdf!(t, cs[i], x)

            # accumulate pdf in batch
            BLAS.axpy!(pi, t, r)
        end
    end
    return r
end

function _mixlogpdf1(d::MixtureModel, x)
    # using the formula below for numerical stability
    #
    # logpdf(d, x) = log(sum_i pri[i] * pdf(cs[i], x))
    #              = log(sum_i pri[i] * exp(logpdf(cs[i], x)))
    #              = log(sum_i exp(logpri[i] + logpdf(cs[i], x)))
    #              = m + log(sum_i exp(logpri[i] + logpdf(cs[i], x) - m))
    #     
    #  m is chosen to be the maximum of logpri[i] + logpdf(cs[i], x) - m
    #  such that the argument of exp is in a reasonable range
    #

    cs = components(d)
    p = priorprobs(d)
    K = length(cs)
    lp = Array(Float64, K)
    m = -Inf   # m <- the maximum of log(p(cs[i], x)) + log(pri[i])
    for i = 1:K
        pi = p[i]
        if pi > 0.0
            # lp[i] <- log(p(cs[i], x)) + log(pri[i])
            lp[i] = lp_i = logpdf(cs[i], x) + log(pi)
            if lp_i > m
                m = lp_i
            end
        end
    end
    v = 0.0
    for i = 1:K
        if p[i] > 0.0
            v += exp(lp[i] - m)
        end
    end
    return m + log(v)
end

function _mixlogpdf!(r::DenseArray, d::MixtureModel, x)
    cs = components(d)
    p = priorprobs(d)
    K = length(cs)
    n = length(r)
    Lp = Array(Float64, n, K)
    m = fill(-Inf, n)
    for i = 1:K
        pi = p[i]
        if pi > 0.0
            lpri = log(pi)
            lp_i = view(Lp, :, i)
            # compute logpdf in batch and store
            logpdf!(lp_i, cs[i], x)

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
    for i = 1:K
        if p[i] > 0.0
            lp_i = view(Lp, :, i)
            for j = 1:n
                r[j] += exp(lp_i[j] - m[j])
            end
        end
    end
    for j = 1:n
        r[j] = log(r[j]) + m[j]
    end
    return r
end

pdf(d::UnivariateMixture, x::Real) = _mixpdf1(d, x)
logpdf(d::UnivariateMixture, x::Real) = _mixlogpdf1(d, x)
_pdf!(r::AbstractArray, d::UnivariateMixture, x::DenseArray) = _mixpdf!(r, d, x)
_logpdf!(r::AbstractArray, d::UnivariateMixture, x::DenseArray) = _mixlogpdf!(r, d, x)

_pdf(d::MultivariateMixture, x::AbstractVector) = _mixpdf1(d, x)
_logpdf(d::MultivariateMixture, x::AbstractVector) = _mixlogpdf1(d, x)
_pdf!(r::AbstractArray, d::MultivariateMixture, x::DenseMatrix) = _mixpdf!(r, d, x)
_lodpdf!(r::AbstractArray, d::MultivariateMixture, x::DenseMatrix) = _mixlogpdf!(r, d, x)


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


rand(d::MixtureModel) = rand(d.components[rand(d.prior)])

rand(s::MixtureSampler) = rand(s.csamplers[rand(s.psampler)])
_rand!(s::MixtureSampler{Multivariate}, x::DenseVector) = _rand!(s.csamplers[rand(s.psampler)], x)

sampler(d::MixtureModel) = MixtureSampler(d)

