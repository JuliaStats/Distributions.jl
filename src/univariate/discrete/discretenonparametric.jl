struct DiscreteNonParametric{T<:Real,P<:Real,S<:AbstractVector{T}} <: DiscreteUnivariateDistribution
    support::S
    p::Vector{P}

    DiscreteNonParametric{T,P,S}(vs::S, ps::Vector{P}, ::NoArgCheck) where {T<:Real,P<:Real,S<:AbstractVector{T}} =
        new{T,P,S}(vs, ps)

    function DiscreteNonParametric{T,P,S}(vs::S, ps::Vector{P}) where {T<:Real,P<:Real,S<:AbstractVector{T}}
        @check_args(DiscreteNonParametric, length(vs) == length(ps))
        @check_args(DiscreteNonParametric, isprobvec(ps))
        @check_args(DiscreteNonParametric, allunique(vs))
        sort_order = sortperm(vs)
        new{T,P,S}(vs[sort_order], ps[sort_order])
    end
end

DiscreteNonParametric(vs::S, ps::Vector{P}) where {T<:Real,P<:Real,S<:AbstractVector{T}} =
    DiscreteNonParametric{T,P,S}(vs, ps)

# Conversion
convert(::Type{DiscreteNonParametric{T,P,S}}, d::DiscreteNonParametric) where {T,P,S} =
    DiscreteNonParametric{T,P,S}(S(support(d)), Vector{P}(probs(d)), NoArgCheck())

# Accessors
params(d::DiscreteNonParametric) = (d.support, d.p)
support(d::DiscreteNonParametric) = d.support
probs(d::DiscreteNonParametric)  = d.p

# Sampling

function rand(d::DiscreteNonParametric{T,P}) where {T,P}
    x = support(d)
    p = probs(d)
    draw = rand(P)
    cp = zero(P)
    i = 0
    while cp < draw
        cp += p[i +=1]
    end
    x[i]
end

sampler(d::DiscreteNonParametric) =
    DiscreteNonParametricSampler(support(d), probs(d))

# Evaluation

pdf(d::DiscreteNonParametric) = copy(probs(d))

# Helper functions for pdf and cdf required to fix ambiguous method
# error involving [pc]df(::DisceteUnivariateDistribution, ::Int)
function _pdf(d::DiscreteNonParametric{T,P}, x::T) where {T,P}
    idx_range = searchsorted(support(d), x)
    if length(idx_range) > 0
        return probs(d)[first(idx_range)]
    else
        return zero(P)
    end
end
pdf(d::DiscreteNonParametric{T}, x::Int) where T  = _pdf(d, convert(T, x))
pdf(d::DiscreteNonParametric{T}, x::Real) where T = _pdf(d, convert(T, x))

_cdf(d::DiscreteNonParametric{T}, x::T) where T =
    sum(probs(d)[1:searchsortedlast(support(d), x)]) #TODO: Switch to single-pass
cdf(d::DiscreteNonParametric{T}, x::Int) where T = _cdf(d, convert(T, x))
cdf(d::DiscreteNonParametric{T}, x::Real) where T = _cdf(d, convert(T, x))

function quantile(d::DiscreteNonParametric, q::Real)
    0 <= q <= 1 || throw(DomainError())
    x = support(d)
    p = probs(d)
    k = length(x)
    i = 1
    cp = p[1]
    while cp < q && i < k #Note: is i < k necessary?
        i += 1
        @inbounds cp += p[i]
    end
    x[i]
end


minimum(d::DiscreteNonParametric) = support(d)[1]
maximum(d::DiscreteNonParametric) = support(d)[end]
insupport(d::DiscreteNonParametric, x::Real) =
    length(searchsorted(support(d), x)) > 0

mean(d::DiscreteNonParametric) = dot(probs(d), support(d))

function var(d::DiscreteNonParametric{T}) where T
    m = mean(d)
    x = support(d)
    p = probs(d)
    k = length(x)
    σ² = zero(T)
    for i in 1:k
        @inbounds σ² += abs2(x[i] - m) * p[i]
    end
    σ²
end

function skewness(d::DiscreteNonParametric{T}) where T
    m = mean(d)
    x = support(d)
    p = probs(d)
    k = length(x)
    μ₃ = zero(T)
    σ² = zero(T)
    @inbounds for i in 1:k
        d = x[i] - m
        d²w = abs2(d) * p[i]
        μ₃ += d * d²w
        σ² += d²w
    end
    μ₃ / (σ² * sqrt(σ²))
end

function kurtosis(d::DiscreteNonParametric{T}) where T
    m = mean(d)
    x = support(d)
    p = probs(d)
    k = length(x)
    μ₄ = zero(T)
    σ² = zero(T)
    @inbounds for i in 1:k
        d² = abs2(x[i] - m)
        d²w = d² * p[i]
        μ₄ += d² * d²w
        σ² += d²w
    end
    μ₄ / abs2(σ²) - 3
end

entropy(d::DiscreteNonParametric) = entropy(probs(d))
entropy(d::DiscreteNonParametric, b::Real) = entropy(probs(d), b)

mode(d::DiscreteNonParametric) = support(d)[argmax(probs(d))]
function modes(d::DiscreteNonParametric{T,P}) where {T,P}
    x = support(d)
    p = probs(d)
    k = length(x)
    mds = T[]
    max_p = zero(P)
    @inbounds for i in 1:k
        pi = p[i]
        xi = x[i]
        if pi > max_p
            max_p = pi
            mds = [xi]
        elseif pi == max_p
            push!(mds, xi)
        end
    end
    mds
end

function mgf(d::DiscreteNonParametric, t::Real)
    x, p = params(d)
    s = zero(Float64)
    for i in 1:length(x)
        s += p[i] * exp(t*x[i])
    end
    s
end

function cf(d::DiscreteNonParametric, t::Real)
    x, p = params(d)
    s = zero(Complex{Float64})
    for i in 1:length(x)
       s += p[i] * cis(t*x[i])
    end
    s
end

# Sufficient statistics

struct DiscreteNonParametricStats{T<:Real,P<:AbstractFloat,S<:AbstractVector{T}} <: SufficientStats
    support::S
    freq::Vector{P}
end

function suffstats(::Type{DiscreteNonParametric}, x::AbstractArray{T}) where {T<:Real}

    N = length(x)
    N == 0 && return DiscreteNonParametricStats(T[], Float64[])

    n = 1
    vs = Vector{T}(undef,N)
    ps = zeros(N)
    x = sort(vec(x))

    vs[1] = x[1]
    ps[1] += 1.

    xprev = x[1]
    @inbounds for i = 2:N
        xi = x[i]
        if xi != xprev
            n += 1
            vs[n] = xi
        end
        ps[n] += 1.
        xprev = xi
    end

    resize!(vs, n)
    resize!(ps, n)
    DiscreteNonParametricStats(vs, ps)

end

function suffstats(::Type{DiscreteNonParametric}, x::AbstractArray{T}, w::AbstractArray{P}) where {T<:Real,P<:AbstractFloat}

    @check_args(DiscreteNonParametric, length(x) == length(w))

    N = length(x)
    N == 0 && return DiscreteNonParametricStats(T[], P[])

    n = 1
    vs = Vector{T}(undef, N)
    ps = zeros(P, N)

    xorder = sortperm(vec(x))
    x = vec(x)[xorder]
    w = vec(w)[xorder]

    vs[1] = x[1]
    ps[1] += w[1]

    xprev = x[1]
    @inbounds for i = 2:N
        xi = x[i]
        wi = w[i]
        if xi != xprev
            n += 1
            vs[n] = xi
        end
        ps[n] += wi
        xprev = xi
    end

    resize!(vs, n)
    resize!(ps, n)
    DiscreteNonParametricStats(vs, ps)

end

# # Model fitting

fit_mle(::Type{DiscreteNonParametric}, ss::DiscreteNonParametricStats{T,P,S}) where {T,P,S} =
    DiscreteNonParametric{T,P,S}(ss.support, pnormalize!(copy(ss.freq)), NoArgCheck())
