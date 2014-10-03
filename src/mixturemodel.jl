immutable MixtureModel{VF,VS,Component<:Distribution} <: Distribution{VF,VS}
    components::Vector{Component}
    probs::Vector{Float64}
    aliastable::AliasTable
    function MixtureModel(c::Vector{Component}, p::Vector{Float64})
        if !(Component <: Distribution{VF,VS})
            throw(TypeError(:MixtureModel,
                            "Mixture components type mismatch",
                            Distribution{VF,VS},
                            Component))
        end
        if length(c) != length(p)
            throw(ArgumentError(string("components and probs must have",
                                       " the same number of elements")))
        end
        sizes = [size(component)::Tuple for component in c]
        if !all(sizes .== sizes[1])
            error("MixtureModel: mixture components have different dimensions")
        end
        sump = 0.0
        for i in 1:length(p)
            if p[i] < 0.0
                error("MixtureModel: probabilities must be non-negative")
            end
            sump += p[i]
        end
        table = AliasTable(p ./ sump)
        new(c, p ./ sump, table)
    end
end

function MixtureModel{C<:Distribution}(cs::Vector{C}, ps::Vector{Float64})
    VF = variate_form(C)
    VS = value_support(C)
    MixtureModel{VF,VS,C}(cs, ps)
end

dim(d::MixtureModel{Multivariate}) = dim(d.components[1])

function mean(d::MixtureModel)
    m = 0.0
    for i in 1:length(d.components)
        if d.probs[i] > 0.
            m += mean(d.components[i]) * d.probs[i]
        end
    end
    return m
end

function _mixpdf(d::MixtureModel, x)
    p = 0.0
    for i in 1:length(d.components)
        p += pdf(d.components[i], x) * d.probs[i]
    end
    return p
end

function _mixlogpdf(d::MixtureModel, x)
    l = [(logpdf(d.components[i],x)+log(d.probs[i]))::Float64
         for i in find(d.probs .> 0.)]
    m = maximum(l)
    log(sum(exp(l.-m))) + m
end

pdf(d::MixtureModel{Univariate}, x::Real) = _mixpdf(d, x)
logpdf(d::MixtureModel{Univariate}, x::Real) = _mixlogpdf(d, x)

_pdf(d::MixtureModel{Multivariate}, x::AbstractVector) = _mixpdf(d, x)
_logpdf(d::MixtureModel{Multivariate}, x::AbstractVector) = _mixpdf(d, x)

function rand(d::MixtureModel)
    i = rand(d.aliastable)
    return rand(d.components[i])
end

function var(d::MixtureModel)
    m = 0.0
    squared_mean_mixture = mean(d).^2
    for i in 1:length(d.components)
        if d.probs[i] > 0.
            m += (var(d.components[i]) .- squared_mean_mixture .+ mean(d.components[i]).^2) * d.probs[i]
        end
    end
    return m
end

size(d::MixtureModel) = size(d.components[1])
length(d::MixtureModel) = length(d.components[1])
