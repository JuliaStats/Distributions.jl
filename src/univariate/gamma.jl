immutable Gamma <: ContinuousUnivariateDistribution
    shape::Float64
    scale::Float64
    function Gamma(sh::Real, sc::Real)
        if sh > 0.0 && sc > 0.0
            new(float64(sh), float64(sc))
        else
            error("Both shape and scale must be positive")
        end
    end
end

Gamma(sh::Real) = Gamma(sh, 1.0)
Gamma() = Gamma(1.0, 1.0) # Standard exponential distribution

@_jl_dist_2p Gamma gamma

function entropy(d::Gamma)
    x = (1.0 - d.shape) * digamma(d.shape)
    x += lgamma(d.shape) + log(d.scale) + d.shape
    return x
end

insupport(d::Gamma, x::Number) = isreal(x) && isfinite(x) && 0.0 <= x

kurtosis(d::Gamma) = 6.0 / d.shape

mean(d::Gamma) = d.shape * d.scale

function mgf(d::Gamma, t::Real)
    k, theta = d.shape, d.scale
    return (1.0 - t * theta)^(-k)
end

function cf(d::Gamma, t::Real)
    k, theta = d.shape, d.scale
    return (1.0 - im * t * theta)^(-k)
end

function modes(d::Gamma)
    if d.shape >= 1.0
        d.scale * (d.shape - 1.0)
    end
end

# rand()
#
#  A simple method for generating gamma variables - Marsaglia and Tsang (2000)
#  http://www.cparity.com/projects/AcmClassification/samples/358414.pdf
#  Page 369
#  basic simulation loop for pre-computed d and c
function randg2(d::Float64, c::Float64) 
    while true
        x = v = 0.0
        while v <= 0.0
            x = randn()
            v = 1.0 + c * x
        end
        v = v^3
        U = rand()
        x2 = x^2
        if U < 1.0 - 0.331 * x2^2 ||
           log(U) < 0.5 * x2 + d * (1.0 - v + log(v))
            return d * v
        end
    end
end

function rand(d::Gamma)
    dpar = (d.shape <= 1.0 ? d.shape + 1.0 : d.shape) - 1.0 / 3.0
    return d.scale *
           randg2(dpar, 1.0 / sqrt(9.0 * dpar)) *
           (d.shape > 1.0 ? 1.0 : rand()^(1.0 / d.shape))
end

function rand!(d::Gamma, A::Array{Float64})
    dpar = (d.shape <= 1.0 ? d.shape + 1.0 : d.shape) - 1.0 / 3.0
    cpar = 1.0 / sqrt(9.0 * dpar)
    for i in 1:length(A)
        A[i] = randg2(dpar, cpar)
    end
    if d.shape <= 1.0
        ainv = 1.0 / d.shape
        for i in 1:length(A)
            A[i] *= rand()^ainv
        end
    end
    return d.scale * A
end

skewness(d::Gamma) = 2.0 / sqrt(d.shape)

var(d::Gamma) = d.shape * d.scale * d.scale

function fit(::Type{Gamma}, x::Array)
    a_old = 0.5 / (log(mean(x)) - mean(log(x)))
    a_new = 0.5 / (log(mean(x)) - mean(log(x)))

    z = 1.0 / a_old + (mean(log(x)) - log(mean(x)) + log(a_old) - digamma(a_old)) / (a_old^2 * (1.0 / a_old - trigamma(a_old)))
    a_new, a_old = 1.0 / z, a_new

    iterations = 1

    while abs(a_old - a_new) > 1e-16 && iterations <= 10_000
        z = 1.0 / a_old + (mean(log(x)) - log(mean(x)) + log(a_old) - digamma(a_old)) / (a_old^2 * (1.0 / a_old - trigamma(a_old)))
        a_new, a_old = 1.0 / z, a_new
    end

    if iterations >= 10_000
        error("Parameter estimation failed to converge in 10,000 iterations")
    end

    Gamma(a_new, mean(x) / a_new)
end
