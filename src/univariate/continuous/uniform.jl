immutable Uniform <: ContinuousUnivariateDistribution
    a::Float64
    b::Float64

    function Uniform(a::Real, b::Real)
	   a < b || error("Uniform: a must be less than b")
	   @compat new(Float64(a), Float64(b))
    end

    Uniform() = new(0.0, 1.0)
end

@distr_support Uniform d.a d.b


#### Parameters

params(d::Uniform) = (d.a, d.b)

location(d::Uniform) = d.a
scale(d::Uniform) = d.b - d.a


#### Statistics

mean(d::Uniform) = middle(d.a, d.b)
median(d::Uniform) = mean(d)
mode(d::Uniform) = mean(d)
modes(d::Uniform) = Float64[]

var(d::Uniform) = (w = d.b - d.a; w^2 / 12.0)

skewness(d::Uniform) = 0.0
kurtosis(d::Uniform) = -1.2

entropy(d::Uniform) = log(d.b - d.a)


#### Evaluation

pdf(d::Uniform, x::Float64) = insupport(d, x) ? 1.0 / (d.b - d.a) : 0.0
logpdf(d::Uniform, x::Float64) = insupport(d, x) ? -log(d.b - d.a) : -Inf 

function cdf(d::Uniform, x::Float64) 
    (a, b) = params(d)
    x <= a ? 0.0 :
    x >= d.b ? 1.0 : (x - a) / (b - a)
end

function ccdf(d::Uniform, x::Float64) 
    (a, b) = params(d)
    x <= a ? 1.0 :
    x >= d.b ? 0.0 : (b - x) / (b - a)
end

quantile(d::Uniform, p::Float64) = d.a + p * (d.b - d.a)
cquantile(d::Uniform, p::Float64) = d.b + p * (d.a - d.b)


function mgf(d::Uniform, t::Real)
    (a, b) = params(d)
    u = 0.5 * (b - a) * t
    u == zero(u) && return one(u)
    v = 0.5 * (a + b) *t
    exp(v) * (sinh(u) / u)
end

function cf(d::Uniform, t::Real)
    (a, b) = params(d)
    u = 0.5 * (b - a) * t
    u == zero(u) && return complex(one(u))
    v = 0.5 * (a + b) * t
    cis(v) * (sin(u) / u)
end


#### Evaluation

rand(d::Uniform) = d.a + (d.b - d.a) * rand()


#### Fitting

function fit_mle{T <: Real}(::Type{Uniform}, x::Vector{T})
    if isempty(x)
        throw(ArgumentError("x cannot be empty."))
    end

    xmin = xmax = x[1]
    for i = 2:length(x)
        xi = x[i]
        if xi < xmin
            xmin = xi
        elseif xi > xmax
            xmax = xi
        end
    end

    Uniform(xmin, xmax)
end

