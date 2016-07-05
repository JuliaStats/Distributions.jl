# Truncated normal distribution

TruncatedNormal(mu::Float64, sigma::Float64, a::Float64, b::Float64) =
    Truncated(Normal(mu, sigma), a, b)

TruncatedNormal(mu::Real, sigma::Real, a::Real, b::Real) =
    TruncatedNormal(Float64(mu), Float64(sigma), Float64(a), Float64(b))

### statistics

minimum{T <: Real}(d::Truncated{Normal{T},Continuous}) = d.lower
maximum{T <: Real}(d::Truncated{Normal{T},Continuous}) = d.upper


function mode{T <: Real}(d::Truncated{Normal{T},Continuous})
    μ = mean(d.untruncated)
    d.upper < μ ? d.upper :
    d.lower > μ ? d.lower : μ
end

modes{T <: Real}(d::Truncated{Normal{T},Continuous}) = [mode(d)]


function mean{T <: Real}(d::Truncated{Normal{T},Continuous})
    d0 = d.untruncated
    μ = mean(d0)
    σ = std(d0)
    a = (d.lower - μ) / σ
    b = (d.upper - μ) / σ
    μ + ((normpdf(a) - normpdf(b)) / d.tp) * σ
end

function var{T <: Real}(d::Truncated{Normal{T},Continuous})
    d0 = d.untruncated
    μ = mean(d0)
    σ = std(d0)
    a = (d.lower - μ) / σ
    b = (d.upper - μ) / σ
    z = d.tp
    φa = normpdf(a)
    φb = normpdf(b)
    aφa = isinf(a) ? 0.0 : a * φa
    bφb = isinf(b) ? 0.0 : b * φb
    t1 = (aφa - bφb) / z
    t2 = abs2((φa - φb) / z)
    abs2(σ) * (1 + t1 - t2)
end

function entropy{T <: Real}(d::Truncated{Normal{T},Continuous})
    d0 = d.untruncated
    z = d.tp
    μ = mean(d0)
    σ = std(d0)
    a = (d.lower - μ) / σ
    b = (d.upper - μ) / σ
    aφa = isinf(a) ? 0.0 : a * normpdf(a)
    bφb = isinf(b) ? 0.0 : b * normpdf(b)
    0.5 * (log2π + 1.) + log(σ * z) + (aφa - bφb) / (2.0 * z)
end


### sampling

## Use specialized sampler, as quantile-based method is inaccurate in
## tail regions of the Normal, issue #343

function rand{T <: Real}(d::Truncated{Normal{T},Continuous})
    d0 = d.untruncated
    μ = mean(d0)
    σ = std(d0)
    a = (d.lower - μ) / σ
    b = (d.upper - μ) / σ
    z = randnt(a, b, d.tp)
    return μ + σ * z
end

# Rejection sampler based on algorithm from Robert (1995)
#
#  - Available at http://arxiv.org/abs/0907.4010

function randnt(lb::Float64, ub::Float64, tp::Float64)
    r::Float64
    if tp > 0.3   # has considerable chance of falling in [lb, ub]
        r = randn()
        while r < lb || r > ub
            r = randn()
        end
        return r

    else 
        span = ub - lb
        if lb > 0 && span > 2.0 / (lb + sqrt(lb^2 + 4.0)) * exp((lb^2 - lb * sqrt(lb^2 + 4.0)) / 4.0)
            a = (lb + sqrt(lb^2 + 4.0))/2.0
            while true
                r = rand(Exponential(1.0 / a)) + lb
                u = rand()
                if u < exp(-0.5 * (r - a)^2) && r < ub
                    return r
                end
            end
        elseif ub < 0 && ub - lb > 2.0 / (-ub + sqrt(ub^2 + 4.0)) * exp((ub^2 + ub * sqrt(ub^2 + 4.0)) / 4.0)
            a = (-ub + sqrt(ub^2 + 4.0)) / 2.0
            while true
                r = rand(Exponential(1.0 / a)) - ub
                u = rand()
                if u < exp(-0.5 * (r - a)^2) && r < -lb
                    return -r
                end
            end
        else 
            while true
                r = lb + rand() * (ub - lb)
                u = rand()
                if lb > 0
                    rho = exp((lb^2 - r^2) * 0.5)
                elseif ub < 0
                    rho = exp((ub^2 - r^2) * 0.5)
                else
                    rho = exp(-r^2 * 0.5)
                end
                if u < rho
                    return r
                end
            end
        end
    end
end
