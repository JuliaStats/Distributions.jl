immutable MixtureModel{VF<:VariateForm,VS<:ValueSupport} <: Distribution{VF,VS}
    components::Vector{Distribution{VF,VS}} # Vector should be able to contain any type of
                                            # distribution with comparable support
    probs::Vector{Float64}
    aliastable::AliasTable
    function MixtureModel(c::Vector, p::Vector{Float64})
        if length(c) != length(p)
            error("components and probs must have the same number of elements")
        end
        dims = map(dim, c)
        if !all(dims .== dims[1])
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

dim(d::MixtureModel) = dim(d.components[1])

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

# avoid dispatch ambiguity by defining sufficiently specific methods
pdf(d::MixtureModel{Univariate}, x::Real) = _pdf(d, x)
pdf(d::MixtureModel{Multivariate}, x::Vector) = _pdf(d, x)
pdf(d::MixtureModel{Matrixvariate}, x::Matrix) = _pdf(d, x)

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
