# Von-Mises Fisher: a multivariate distribution useful in directional statistics

# Useful notes:
# http://www.mitsuba-renderer.org/~wenzel/vmf.pdf
# Some of the code adapted from http://www.unc.edu/~sungkyu/manifolds/randvonMisesFisherm.m
# as well as the movMF R package.

immutable VonMisesFisher <: ContinuousMultivariateDistribution
    mu::Vector{Float64}
    kappa::Float64

    function VonMisesFisher{T <: Real}(mu::Vector{T}, kappa::Float64)
        mu = mu ./ norm(mu)
        if kappa < 0
            throw(ArgumentError("kappa must be a nonnegative real number."))
        end
        new(float64(mu), kappa)
    end
end

dim(d::VonMisesFisher) = length(d.mu)

mean(d::VonMisesFisher) = d.mu

scale(d::VonMisesFisher) = d.kappa

insupport{T <: Real}(d::VonMisesFisher, x::Vector{T}) = abs(sum(x) - 1.) < 1e-8

function rand(d::VonMisesFisher, n::Int)
    randvonMisesFisher(n, d.kappa, d.mu)
end

function logpdf(d::VonMisesFisher, x::Vector{Float64}; stable=true)
    if abs(d.kappa - 0.0) < eps() return 1/4/pi; end
    if stable
        # As suggested by Wenzel Jakob: http://www.mitsuba-renderer.org/~wenzel/vmf.pdf
        return d.kappa * dot(d.mu, x) - d.kappa + log(d.kappa) - log(2*pi) - log(1-exp(-2*d.kappa))
    else 
        # As described on Wikipedia
        p = dim(d)
        logCpk = 0.0
        if p == 3
            logCpk = log(d.kappa) - log(2 * pi * (exp(kappa) - exp(-kappa)))
        else
            logCpk = (p/2 - 1) * log(d.kappa) - (p/2) * log(2*pi) - log(besselj(p/2-1, d.kappa))
        end
        return d.kappa * dot(d.mu, x) + logCpk
    end
end

# Helper functions

# Sample n vectors x ~ VonMisesFisher(mu, kappa)
function randvonMisesFisher(n, kappa, mu)
    m = length(mu)
    w = rW(n, kappa, m)
    v = rand(MvNormal(zeros(m-1), eye(m-1)), n)
    v = normalize(v',2,2)
    r = sqrt(1.0 .- w .^ 2)
    for j = 1:size(v,2) v[:,j] = v[:,j] .* r; end  
    x = hcat(v, w)
    mu = mu / norm(mu)
    return rotMat(mu)'*x'
end

# Randomly sample W
function rW(n, kappa, m)
    y = zeros(n)
    l = kappa;
    d = m - 1;
    b = (- 2. * l + sqrt(4. * l * l + d * d)) / d;
    x = (1. - b) / (1. + b);
    c = l * x + d * log(1. - x * x);
    w = 0
    for i=1:n
        done = false
        while !done
            z = rand(Beta(d / 2., d / 2.))
            w = (1. - (1. + b) * z) / (1. - (1. - b) * z);
            u = rand()
            if l * w + d * log(1. - x * w) - c >= log(u)
                done = true
            end
        end
        y[i] = w
    end
    return y
end

# Rotation helper function
function rotMat(b)
    d = length(b)
    b= b/norm(b)
    a = [zeros(d-1,1); 1]
    alpha = acos(a'*b)[1]
    c = b - a * (a'*b); c = c / norm(c)
    A = a*c' - c*a'
    return eye(d) + sin(alpha)*A + (cos(alpha) - 1)*(a*a' +c*c')
end

# Each row of x assumed to be ~ VonMisesFisher(mu, kappa)
# MLE notes from: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.186.1887&rep=rep1&type=pdf

function fit_mle(::Type{VonMisesFisher}, x::Matrix{Float64})
    (n,p) = size(x)
    sx = sum(x, 1)
    mu = sx[:] / norm(sx)
    rbar = norm(sx) / n
    kappa0 = rbar * (p-rbar^2) / (1-rbar^2)  # Eqn. 4
    # TODO: Include a few Newton steps to get a better approximation.
    # A(p,kappa) = besselj(p/2, kappa) / besselj(p/2-1, kappa)
    # apk0 = A(p,kappa0)
    # kappa1 = kappa0 + (apk0 - rbar) / (1 - apk0^2 - (p-1)*apk0/kappa0)
    # apk1 = A(p,kappa1)
    # kappa2 = kappa1 + (apk1 - rbar) / (1 - apk1^2 - (p-1)*apk1/kappa1)
    return VonMisesFisher(mu, kappa0)#, kappa1, kappa2)
end
