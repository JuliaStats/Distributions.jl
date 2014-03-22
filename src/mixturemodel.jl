immutable MixtureModel <: Distribution
    components::Vector # Vector should be able to contain any type of
                       # distribution with comparable support
    probs::Vector{Float64}
    aliastable::AliasTable
    function MixtureModel(c::Vector, p::Vector{Float64})
        if length(c) != length(p)
            error("components and probs must have the same number of elements")
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

function mean(d::MixtureModel)
    m = 0.0
    for i in 1:length(d.components)
        m += mean(d.components[i]) * d.probs[i]
    end
    return m
end

function pdf(d::MixtureModel, x::Any)
    p = 0.0
    for i in 1:length(d.components)
        p += pdf(d.components[i], x) * d.probs[i]
    end
    return p
end

function rand(d::MixtureModel)
    i = rand(d.aliastable)
    return rand(d.components[i])
end

function var(d::MixtureModel)
    m = 0.0
    squared_mean_mixture = mean(d).^2
    for i in 1:length(d.components)
        m += (var(d.components[i]) .- squared_mean_mixture .+ mean(d.components[i]).^2) * d.probs[i]
    end
    return m
end
