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
        table = AliasTable(p)
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
        m += mean(d.components[i]) * d.probs[i]
    end
    return m
end

function _pdf(d::MixtureModel, x)
    p = 0.0
    for i in 1:length(d.components)
        p += pdf(d.components[i], x) * d.probs[i]
    end
    return p
end

function _logpdf(d::MixtureModel, x)
    l = [(logpdf(d.components[i],x)+log(d.probs[i]))::Float64
         for i in find(d.probs .> 0.)]
    m = maximum(l)
    log(sum(exp(l.-m))) + m
end

# avoid dispatch ambiguity by defining sufficiently specific methods
pdf(d::MixtureModel{Univariate}, x::Real) = _pdf(d, x)
pdf(d::MixtureModel{Multivariate}, x::Vector) = _pdf(d, x)
pdf(d::MixtureModel{Matrixvariate}, x::Matrix) = _pdf(d, x)

logpdf(d::MixtureModel{Univariate}, x::Real) = _logpdf(d,x)
logpdf(d::MixtureModel{Multivariate}, x::Vector) = _logpdf(d,x)
logpdf(d::MixtureModel{Matrixvariate}, x::Matrix) = _logpdf(d,x)

function rand(d::MixtureModel)
    i = rand(d.aliastable)
    return rand(d.components[i])
end

# TODO: Correct this definition
function var(d::MixtureModel)
    m = 0.0
    for i in 1:length(d.components)
        m += var(d.components[i]) * d.probs[i]^2
    end
    return m
end

size(d::MixtureModel) = size(d.components[1])
length(d::MixtureModel) = length(d.components[1])
