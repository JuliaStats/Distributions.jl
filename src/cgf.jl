"""
    cgf(d::UnivariateDistribution, t)

Evaluate the [cumulant-generating-function](https://en.wikipedia.org/wiki/Cumulant) of `distribution` at `t`.
Mathematically the cumulant-generating-function is the logarithm of the [moment-generating-function](https://en.wikipedia.org/wiki/Moment-generating_function):
`cgf = log ∘ mgf`. In practice, however, the right hand side may have overflow issues.

See also [`mgf`](@ref)
"""
function cgf end

function cgf(d::Dirac, t)
    t*d.value
end

function cgf_Bernoulli(p,t)
    # log(1-p+p*exp(t))
    logaddexp(log1p(-p), t+log(p))
end
function cgf(d::Bernoulli, t)
    p, = params(d)
    cgf_Bernoulli(p,t)
end
function cgf(d::Binomial, t)
    n,p = params(d)
    n*cgf_Bernoulli(p,t)
end
function cgf_Geometric(p, t)
    # log(p / (1 - (1-p) * exp(t)))
    log(p) - logsubexp(0, t + log1p(-p))
end
function cgf(d::Geometric, t)
    p, = params(d)
    cgf_Geometric(p,t)
end
function cgf(d::NegativeBinomial, t)
    r,p = params(d)
    r*cgf_Geometric(p,t)
end
cgf(d::Poisson, t) = mean(d) * expm1(t)

function expfd0_taylor(x)
    # taylor series of (exp(x) - 1) / x
    evalpoly(x, (1, 1/2, 1/6, 1/24, 1/120))
end

function cgf(d::Uniform, t)
    a,b = params(d)
    # log((exp(t*b) - exp(t*a))/ (t*(b-a)))
    x = t*(b-a)
    if abs(x) < sqrt(eps(float(one(x))))
        t*a + log(expfd0_taylor(x))
    else
        logsubexp(t*b, t*a) - log(abs(x))
    end
end
function cgf(d::Normal, t)
    μ,σ = params(d)
    t*μ + (σ*t)^2/2
end
function cgf(d::Exponential, t)
    μ = mean(d)
    return - log1p(- t * μ)
end

function cgf(d::Gamma, t)
    α, θ = params(d)
    return α * cgf(Exponential{typeof(θ)}(θ), t)
end
function cgf(d::Laplace, t)
    μ,θ = params(d)
    t*μ - log1p(-(θ*t)^2)
end
function cgf_Chisq(ν,t)
    -ν/2*log1p(-2*t)
end
function cgf(d::Chisq, t)
    ν, = params(d)
    cgf_Chisq(ν,t)
end
function cgf(d::NoncentralChisq, t)
    ν, λ = params(d)
    λ*t/(1-2t) + cgf_Chisq(ν,t)
end
