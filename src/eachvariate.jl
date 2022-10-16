## AbstractArray wrapper for collection of variates
## Similar to https://github.com/JuliaLang/julia/pull/32310 - replace with EachSlice?
struct EachVariate{V,P,A,T,N} <: AbstractArray{T,N}
    parent::P
    axes::A
end

function EachVariate{V}(x::AbstractArray{<:Real,M}) where {V,M}
    ax = ntuple(i -> axes(x, i + V), Val(M - V))
    T = typeof(view(x, ntuple(i -> i <= V ? Colon() : firstindex(x, i), Val(M))...))
    return EachVariate{V,typeof(x),typeof(ax),T,M-V}(x, ax)
end

function ChainRulesCore.rrule(::Type{EachVariate{V}}, x::AbstractArray{<:Real}) where {V}
    y = EachVariate{V}(x)
    size_x = size(x)
    function EachVariate_pullback(Δ)
        # TODO: Should we also handle `Tangent{<:EachVariate}`?
        Δ_out = reshape(mapreduce(vec, vcat, ChainRulesCore.unthunk(Δ)), size_x)
        return (ChainRulesCore.NoTangent(), Δ_out)
    end
    return y, EachVariate_pullback
end

Base.IteratorSize(::Type{EachVariate{V,P,A,T,N}}) where {V,P,A,T,N} = Base.HasShape{N}()

Base.axes(x::EachVariate) = x.axes

Base.size(x::EachVariate) = map(length, x.axes)
Base.size(x::EachVariate, d::Int) = 1 <= ndims(x) ? length(axes(x)[d]) : 1

# We don't need `setindex!` (currently), therefore only `getindex` is implemented
Base.@propagate_inbounds function Base.getindex(
    x::EachVariate{V,P,A,T,N}, I::Vararg{Int,N},
) where {V,P,A,T,N}
    return view(x.parent, ntuple(_ -> Colon(), Val(V))..., I...)
end

# optimization for univariate distributions
eachvariate(x::AbstractArray{<:Real}, ::Type{Univariate}) = x
function eachvariate(x::AbstractArray{<:Real}, ::Type{ArrayLikeVariate{N}}) where {N}
    return EachVariate{N}(x)
end
