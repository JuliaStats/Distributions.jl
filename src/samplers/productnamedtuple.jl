struct ProductNamedTupleSampler{Tnames,Tsamplers,S<:ValueSupport} <:
       Sampleable{NamedTupleVariate{Tnames},S}
    samplers::NamedTuple{Tnames,Tsamplers}
end

function rand(rng::AbstractRNG, spl::ProductNamedTupleSampler{K}) where {K}
    return NamedTuple{K}(map(Base.Fix1(rand, rng), spl.samplers))
end

function rand!(
    rng::AbstractRNG, spl::ProductNamedTupleSampler, xs::AbstractArray{<:NamedTuple{K}}
) where {K}
    for i in eachindex(xs)
        xs[i] = NamedTuple{K}(rand(rng, spl))
    end
    return xs
end
