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
struct HistogramDist{Tp<:Real,Tb<:Real,TB<:AbstractVector{Tb},TP<:AbstractVector{Tp},TA} <: ContinuousUnivariateDistribution
	# params
	B :: TB
	P :: TP
	# cache
	_cdf :: Vector{Tp}
	_ccdf :: Vector{Tp}
	_aliastable :: TA
	function HistogramDist{Tp,Tb,TB,TP}(B::TB, P::TP) where {Tp<:Real,Tb<:Real,TB<:AbstractVector{Tb},TP<:AbstractVector{Tp}}
		@check_args(HistogramDist, typeof(axes(P)) <: Tuple{Base.OneTo})
		@check_args(HistogramDist, typeof(axes(B)) <: Tuple{Base.OneTo})
		@check_args(HistogramDist, isprobvec(P))
		n = length(P)
		@check_args(HistogramDist, length(B)==n+1)
		binwidths = Vector{Tp}(undef, n)
		binwidths .= view(B,2:n+1) .- view(B,1:n)
		@check_args(HistogramDist, all(binwidths .> 0))
		cdf = zeros(Tp, n+1)
		for i in 1:n
			@inbounds cdf[i+1] = cdf[i] + P[i]
		end
		cdf ./= cdf[end]
		@assert iszero(cdf[1]) && isone(cdf[end])
		ccdf = zeros(Tp, n+1)
		for i in n:-1:1
		    @inbounds ccdf[i] = ccdf[i+1] + P[i]
		end
		ccdf ./= ccdf[1]
		@assert iszero(ccdf[end]) && isone(ccdf[1])
		aliastable = AliasTable(P)
		new{Tp,Tb,TB,TP,typeof(aliastable)}(B, P, cdf, ccdf, aliastable)
	end
end

#### Constructors
HistogramDist{Tp,Tb,TB,TP}(B, P) where {Tp,Tb,TB,TP} =
	HistogramDist{Tp,Tb,TB,TP}(convert(TB, B), convert(TP, P))

HistogramDist{Tp,Tb,TB}(B, P::TP) where {Tp,Tb,TB,TP<:AbstractVector{Tp}} =
	HistogramDist{Tp,Tb,TB,TP}(B, P)
HistogramDist{Tp,Tb,TB}(B, P) where {Tp,Tb,TB} =
	HistogramDist{Tp,Tb,TB,Vector{Tp}}(B, P)

HistogramDist{Tp,Tb}(B::BT, P) where {Tp,Tb,BT<:AbstractVector{Tb}} =
	HistogramDist{Tp,Tb,BT}(B, P)
HistogramDist{Tp,Tb}(B, P) where {Tp,Tb} =
	HistogramDist{Tp,Tb,Vector{Tb}}(B, P)

HistogramDist{Tp}(B::TB, P) where {Tp,Tb,TB<:AbstractVector{Tb}} =
	HistogramDist{Tp,Tb,TB}(B, P)
HistogramDist{Tp}(B, P) where {Tp} =
	HistogramDist{Tp,Tp}(B, P)

HistogramDist(B, P::Categorical) =
	HistogramDist(B, probs(P))
HistogramDist(B, P::AbstractVector{Tp}) where {Tp} =
	HistogramDist(B, float(P))
HistogramDist(B, P::AbstractVector{Tp}) where {Tp<:AbstractFloat} =
	HistogramDist{Tp}(B, P)

@distr_support HistogramDist convert(eltype(d), @inbounds d.B[1]) convert(eltype(d), @inbounds d.B[end])

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
params(d::HistogramDist) = (d.B, d.P)
probs(d::HistogramDist) = d.P
ncomponents(d::HistogramDist) = length(d.P)
component(d::HistogramDist, i::Integer) = (@boundscheck 1≤i≤ncomponents(d) || throw(BoundsError(i)); @inbounds Uniform(d.B[i], d.B[i+1]))
components(d::HistogramDist) = [@inbounds component(d,i) for i in 1:ncomponents(d)]

#### show
function show(io::IO, d::HistogramDist)
	print(io, "HistogramDist(B=")
	show(io, d.B)
	print(io, ", P=")
	show(io, d.P)
	print(io, ")")
end

#### Compare
==(d1::HistogramDist, d2::HistogramDist) = ((d1.B == d2.B) || all(d1.B .== d2.B)) && ((d1.P == d2.P) || all(d1.P .== d2.P))
Base.isapprox(d1::HistogramDist, d2::HistogramDist) = ((d1.B ≈ d2.B) || all(d1.B .≈ d2.B)) && ((d1.P ≈ d2.P) || all(d1.P .≈ d2.P))
Base.hash(d::HistogramDist, h::UInt) = hash(d.P, hash(d.B, h))

#### Statistics
mean(d::HistogramDist) =
	@inbounds convert(eltype(d), sum(p*mean(component(d,i)) for (i,p) in enumerate(probs(d))))
var(d::HistogramDist) = begin
	v1 = @inbounds sum(p*var(component(d,i)) for (i,p) in enumerate(probs(d)))
	m = mean(d)
	v2 = @inbounds sum(p*abs2(mean(component(d,i)) - m) for (i,p) in enumerate(probs(d)))
	convert(eltype(d), v1 + v2)
end
mode(d::HistogramDist) = @inbounds d.B[argmax(d.P)]

#### Evaluation
interplinear(a::Real, b::Real, x::Real) = interplinear(promote(a,b,x)...)
interplinear(a::T, b::T, x::T) where {T<:Integer} = a + x*(b-a)
interplinear(a::T, b::T, x::T) where {T<:AbstractFloat} = ((x,a,b) = x<T(0.5) ? (x,a,b) : (1-x,b,a); S=b-a; s = abs(b)>abs(a) ? (S-b)+a : (S+a)-b; (a+x*S)-x*s)
function pdf(d::HistogramDist{Tp}, x::Real) where {Tp}
	i = searchsortedlast(d.B, x)
	@inbounds(0 < i < length(d.B) ? d.P[i] / (d.B[i+1] - d.B[i]) : zero(Tp)) :: Tp
end
function cdf(d::HistogramDist{Tp}, x::Real) where {Tp}
	i = searchsortedlast(d.B, x)
	@inbounds(i ≤ 0 ? zero(Tp) : i ≥ length(d.B) ? one(Tp) : interplinear(d._cdf[i], d._cdf[i+1], (x-d.B[i])/(d.B[i+1]-d.B[i]))) :: Tp
end
function ccdf(d::HistogramDist{Tp}, x::Real) where {Tp}
	i = searchsortedlast(d.B, x)
	@inbounds(i ≤ 0 ? one(Tp) : i ≥ length(d.B) ? zero(Tp) : interplinear(d._ccdf[i], d._ccdf[i+1], (x-d.B[i])/(d.B[i+1]-d.B[i]))) :: Tp
end
function quantile(d::HistogramDist, p::Real)
	0 ≤ p ≤ 1 || throw(DomainError(p, "not a probability"))
	i = searchsortedlast(d._cdf, p)
	@inbounds i==1 ? convert(eltype(d), d.B[1]) : i==length(d.B) ? convert(eltype(d), d.B[end]) : convert(eltype(d), interplinear(d.B[i], d.B[i+1], (p - d._cdf[i])/(d._cdf[i+1] - d._cdf[i])))
end
mgf(d::HistogramDist, t::Real) =
	@inbounds sum(p*mgf(component(d,i),t) for (i,p) in enumerate(probs(d)))
cf(d::HistogramDist, t::Real) =
	@inbounds sum(p*sf(component(d,i),t) for (i,p) in enumerate(probs(d)))

#### Sampling
function rand(rng::AbstractRNG, d::HistogramDist)
	p = rand(rng)
	i = rand(rng, d._aliastable)
	@inbounds convert(eltype(d), interplinear(d.B[i], d.B[i+1], p))
end

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
