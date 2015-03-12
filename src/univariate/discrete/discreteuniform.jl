immutable DiscreteUniform <: DiscreteUnivariateDistribution
    a::Int
    b::Int
    pv::Float64

    function DiscreteUniform(a::Int, b::Int)
        a <= b || error("a and b must satisfy a <= b")
        new(a, b, 1.0 / (b - a + 1))
    end

    @compat DiscreteUniform(a::Real, b::Real) = DiscreteUniform(round(Int, a), round(Int, b))
    DiscreteUniform(b::Real) = DiscreteUniform(0, round(Int, b))
    DiscreteUniform() = new(0, 1, 0.5)
end

@distr_support DiscreteUniform d.a d.b

### Parameters

span(d::DiscreteUniform) = d.b - d.a + 1
probval(d::DiscreteUniform) = d.pv
params(d::DiscreteUniform) = (d.a, d.b)


### Show

show(io::IO, d::DiscreteUniform) = show(io, d, (:a, :b))


### Statistics

mean(d::DiscreteUniform) = middle(d.a, d.b)

median(d::DiscreteUniform) = middle(d.a, d.b)

@compat var(d::DiscreteUniform) = (abs2(Float64(span(d))) - 1.0) / 12.0

skewness(d::DiscreteUniform) = 0.0

function kurtosis(d::DiscreteUniform)
    @compat n2 = abs2(Float64(span(d)))
    return -1.2 * (n2 + 1.0) / (n2 - 1.0)
end

@compat entropy(d::DiscreteUniform) = log(Float64(span(d)))

mode(d::DiscreteUniform) = d.a
modes(d::DiscreteUniform) = [d.a:d.b]


### Evaluation

cdf(d::DiscreteUniform, x::Int) = (x < d.a ? 0.0 :
                                   x > d.b ? 1.0 :
                                   (floor(Int,x) - d.a + 1.0) * d.pv)

pdf(d::DiscreteUniform, x::Int) = insupport(d, x) ? d.pv : 0.0

logpdf(d::DiscreteUniform, x::Int) = insupport(d, x) ? log(d.pv) : -Inf

pdf(d::DiscreteUniform) = fill(probval(d), span(d))

function _pdf!(r::AbstractArray, d::DiscreteUniform, rgn::UnitRange)
    vfirst = round(Int, first(rgn))
    vlast = round(Int, last(rgn))
    vl = max(vfirst, d.a)
    vr = min(vlast, d.b)
    if vl > vfirst
        for i = 1:(vl - vfirst)
            r[i] = 0.0
        end
    end
    fm1 = vfirst - 1
    if vl <= vr
        pv = d.pv
        for v = vl:vr
            r[v - fm1] = pv
        end
    end
    if vr < vlast
        for i = (vr-vfirst+2):length(rgn)
            r[i] = 0.0
        end
    end
    return r
end

function _logpdf!(r::AbstractArray, d::DiscreteUniform, x::AbstractArray)
    lpv = log(probval(d))
    for i = 1:length(x)
        @inbounds r[i] = insupport(d, x[i]) ? lpv : -Inf
    end
    return r
end

quantile(d::DiscreteUniform, p::Float64) = d.a + floor(Int,p * span(d))

function mgf(d::DiscreteUniform, t::Real)
    a, b = d.a, d.b
    u = b - a + 1
    t == 0 ? 1.0 : (exp(t*a) * expm1(t*u)) / (u*expm1(t))
end

function cf(d::DiscreteUniform, t::Real)
    a, b = d.a, d.b
    u = b - a + 1
    t == 0 ? complex(1.0) : (im*cos(t*(a+b)/2) + sin(t*(a-b-1)/2)) / (u*sin(t/2))
end


### Sampling

rand(d::DiscreteUniform) = randi(d.a, d.b)

# Fit model

function fit_mle{T <: Real}(::Type{DiscreteUniform}, x::Array{T})
    if isempty(x)
        throw(ArgumentError("x cannot be empty."))
    end

    xmin = xmax = x[1]
    for i = 2:length(x)
        @inbounds xi = x[i]
        if xi < xmin
            xmin = xi
        elseif xi > xmax
            xmax = xi
        end
    end

    DiscreteUniform(xmin, xmax)
end
