# Generic rand methods

function rand!(s::Sampleable{Univariate}, A::AbstractArray)
    for i in 1:length(A)
        @inbounds A[i] = rand(s)
    end
    return A
end

rand{S<:ValueSupport}(s::Sampleable{Univariate,S}, shp::Union(Int,(Int...))) = 
    rand!(s, Array(eltype(S), shp))

function rand!(s::Sampleable{Multivariate}, A::DenseMatrix)
    for i = 1:size(A,2)
        rand!(s, view(A,:,i))
    end
    return A
end

rand{S<:ValueSupport}(s::Sampleable{Multivariate,S}) = 
    rand!(s, Array(eltype(S), length(s)))

rand{S<:ValueSupport}(s::Sampleable{Multivariate,S}, n::Int) = 
    rand!(s, Array(eltype(S), length(s), n))

function rand!{M<:Matrix}(s::Sampleable{Matrixvariate}, X::Array{M})
    for i in 1:length(X)
        X[i] = rand(s)
    end
    return X
end

rand{S<:ValueSupport}(s::Sampleable{Matrixvariate,S}, n::Int) =
    rand!(s, Array(Matrix{eltype(S)}, n))
