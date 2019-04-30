"""
    HistogramDist(B,P)

The continuous distribution formed by sampling uniformly from a histogram with ``n`` bins described by ``n+1`` bin edges ``B`` and ``n`` bin probabilities ``P``. Its probability density function is

```math
f(x; B, P) = \\frac{P_i}{B_{i+1}-B_i} \\quad\\text{ where } B_i ≤ x < B_{i+1}\\text.
```

```julia
HistogramDist(B,P)   # histogram distribution with bins edges B and probabilities P

params(d)            # Get the parameters, i.e. (B,P)
minimum(d)           # Get the lower bound, i.e. B[1]
maximum(d)           # Get the upper bound, i.e. B[n]
probs(d)             # Get the bin probabilities, i.e. P
ncomponents(d)       # Get the number of bins, i.e. n
component(d,i)       # Get the ith component, i.e. Uniform(B[i],B[i+1])
```
"""
struct HistogramDist{Tp<:Real,Tb<:Real,TB<:AbstractVector{Tb},TP<:Categorical{Tp},TA} <: ContinuousUnivariateDistribution
	# params
	B :: TB
	P :: TP
	# cache
	_pdf :: Vector{Tp}
	_cdf :: Vector{Tp}
	_aliastable :: TA
	function HistogramDist{Tp,Tb,TB,TP}(B::TB, P::TP) where {Tp<:Real,Tb<:Real,TB<:AbstractVector{Tb},TP<:Categorical{Tp}}
		ps = probs(P)
		typeof(axes(ps)) <: Tuple{Base.OneTo} || error("P must have 1-up indexing")
		typeof(axes(B)) <: Tuple{Base.OneTo} || error("B must have 1-up indexing")
		n = length(ps)
		length(B)==n+1 || error("expect one more bin edge than bin probs")
		ws = Vector{Tp}(undef, n)
		ws .= view(B,2:n+1) .- view(B,1:n)
		all(ws .> 0) || error("bin edges must be sorted and distinct")
		pdf = Vector{Tp}(undef, n)
		pdf .= ps ./ ws
		cdf = zeros(Tp, n)
		cdf[2:end] .= cumsum(ps[1:end-1])
		aliastable = AliasTable(ps)
		new{Tp,Tb,TB,TP,typeof(aliastable)}(B, P, pdf, cdf, aliastable)
	end
end

#### Constructors
HistogramDist{Tp,Tb,TB,TP}(B, P) where {Tp,Tb,TB,TP} =
	HistogramDist{Tp,Tb,TB,TP}(convert(TB, B), convert(TP, P))

HistogramDist{Tp,Tb,TB}(B, P::TP) where {Tp,Tb,TB,TP<:Categorical{Tp}} =
	HistogramDist{Tp,Tb,TB,TP}(B, P)
HistogramDist{Tp,Tb,TB}(B, P::TP) where {Tp,Tb,TB,TP<:AbstractVector{Tp}} =
	HistogramDist{Tp,Tb,TB,Categorical{Tp,TP}}(B, P)
HistogramDist{Tp,Tb,TB}(B, P) where {Tp,Tb,TB} =
	HistogramDist{Tp,Tb,TB,Categorical{Tp,Vector{Tp}}}(B, P)

HistogramDist{Tp,Tb}(B::BT, P) where {Tp,Tb,BT<:AbstractVector{Tb}} =
	HistogramDist{Tp,Tb,BT}(B, P)
HistogramDist{Tp,Tb}(B, P) where {Tp,Tb} =
	HistogramDist{Tp,Tb,Vector{Tb}}(B, P)

HistogramDist{Tp}(B::TB, P) where {Tp,Tb,TB<:AbstractVector{Tb}} =
	HistogramDist{Tp,Tb,TB}(B, P)
HistogramDist{Tp}(B, P) where {Tp} =
	HistogramDist{Tp,Tp}(B, P)

HistogramDist(B, P::Categorical{Tp}) where {Tp} =
	HistogramDist{Tp}(B, P)
HistogramDist(B, P::AbstractVector{Tp}) where {Tp} =
	HistogramDist{Tp}(B, P)

@distr_support HistogramDist d.B[1] d.B[end]

#### Conversions
convert(::Type{TH}, B, P) where {TH<:HistogramDist} = TH(B, P)
convert(::Type{TH}, d::TH) where {TH<:HistogramDist} = d
convert(::Type{TH}, d::HistogramDist) where {TH<:HistogramDist} = TH(d.B, d.P)
function convert(::Type{TH}, h::StatsBase.Histogram{T,1}) where {TH<:HistogramDist,T}
	h2 = LinearAlgebra.normalize(h)
	TH(h2.edges[1], h2.weights)
end
convert(::Type{H}, d::HistogramDist) where {H<:StatsBase.Histogram} = H(d.B, probs(d), :left, true)

#### Parameters
params(d::HistogramDist) = (d.B, probs(d.P))
probs(d::HistogramDist) = probs(d.P)
ncomponents(d::HistogramDist) = length(probs(d))
component(d::HistogramDist, i::Integer) = (@boundscheck 1≤i<length(d.B) || throw(BoundsError(i)); @inbounds Uniform(d.B[i], d.B[i+1]))
components(d::HistogramDist) = [@inbounds component(d,i) for i in 1:ncomponents(d)]

#### show
function show(io::IO, d::HistogramDist)
	print(io, "HistogramDist(B=")
	show(io, d.B)
	print(io, ", P=")
	show(io, probs(d))
	print(io, ")")
end

#### Statistics
mean(d::HistogramDist) =
	@inbounds sum(p*mean(component(d,i)) for (i,p) in enumerate(probs(d)))
var(d::HistogramDist) = begin
	v1 = @inbounds sum(p*var(component(d,i)) for (i,p) in enumerate(probs(d)))
	m = mean(d)
	v2 = @inbounds sum(p*abs2(mean(component(d,i)) - m) for (i,p) in enumerate(probs(d)))
	v1 + v2
end

#### Evaluation
function pdf(d::HistogramDist{Tp}, x::Real) where {Tp}
	i = searchsortedlast(d.B, x)
	@inbounds(0 < i < length(d.B) ? d._pdf[i] : zero(Tp)) :: Tp
end
function cdf(d::HistogramDist{Tp}, x::Real) where {Tp}
	i = searchsortedlast(d.B, x)
	@inbounds(i ≤ 0 ? zero(Tp) : i ≥ length(d.B) ? one(Tp) : (x-d.B[i])*d._pdf[i] + d._cdf[i]) :: Tp
end
function quantile(d::HistogramDist, p::Real)
	i = searchsortedlast(d._cdf, p)
	(i ≤ 0 || p > 1) && throw(DomainError(p, "not a probability"))
	@inbounds convert(Float64, d.B[i] + (p - d._cdf[i]) / d._pdf[i])
end
mgf(d::HistogramDist, t::Real) =
	@inbounds sum(p*mgf(component(d,i),t) for (i,p) in enumerate(probs(d)))
cf(d::HistogramDist, t::Real) =
	@inbounds sum(p*sf(component(d,i),t) for (i,p) in enumerate(probs(d)))

#### Sampling
function rand(rng::AbstractRNG, d::HistogramDist)
	p = rand(rng)
	i = rand(rng, d._aliastable)
	@inbounds d.B[i] + p*(d.B[i+1] - d.B[i])
end

# function rand(rng::AbstractRNG, d::HistogramDist)
#     p = rand(rng)
#     i = searchsortedlast(d._cdf, p)
#     @inbounds convert(Float64, d.B[i] + (p - d._cdf[i]) / d._pdf[i])
# end

#### Fitting
function fit_mle(::Type{TD}, x::AbstractArray{<:Real}, bins::AbstractArray{<:Real}) where {TD<:HistogramDist}
	n = length(bins) - 1
	n ≥ 0 || error("require at least one bin edge")
	issorted(bins) || error("bins must be sorted")
	counts = zeros(Int, n)
	for pt in x
	    i = searchsortedlast(bins, pt)
	    if 1 ≤ i ≤ n
	    	@inbounds counts[i] += 1
	    else
	    	error("out of bounds: $pt")
	    end
	end
	TD(bins, counts./sum(counts))
end
