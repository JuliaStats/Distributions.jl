## AbstractArray wrapper for collection of variates
## Similar to https://github.com/JuliaLang/julia/pull/32310 - replace with EachSlice?
struct EachVariate{V,P,A,T,N} <: AbstractArray{T,N}
    parent::P
    axes::A
end

function EachVariate{V}(x::AbstractArray{<:Real,M}) where {V,M}
    ax = ntuple(i -> axes(x, i + V), Val(M - V))
    T = Base.promote_op(view, typeof(x), ntuple(i -> i <= V ? Colon : eltype(axes(x, i)), Val(M))...)
    return EachVariate{V,typeof(x),typeof(ax),T,M-V}(x, ax)
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
