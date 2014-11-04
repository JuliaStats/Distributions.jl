# Truncated normal distribution

### statistics

minimum(d::Truncated{Normal}) = d.lower
maximum(d::Truncated{Normal}) = d.upper


function mode(d::Truncated{Normal})
    μ = mean(d.untruncated)
    d.upper < mu ? d.upper :
    d.lower > mu ? d.lower : μ
end

modes(d::Truncated{Normal}) = [mode(d)]


function mean(d::Truncated{Normal})
    d0 = d.untruncated
    μ = mean(d0)
    σ = std(d0)
    a = (d.lower - μ) / σ
    b = (d.upper - μ) / σ
    μ + ((φ(a) - φ(b)) / d.tp) * σ
end

function var(d::Truncated{Normal})
    d0 = d.untruncated
    μ = mean(d0)
    σ = std(d0)
    a = (d.lower - μ) / σ
    b = (d.upper - μ) / σ
    z = d.tp
    φa = φ(a)
    φb = φ(b)
    aφa = isinf(a) ? 0.0 : a * φa
    bφb = isinf(b) ? 0.0 : b * φb
    t1 = (aφa - bφb) / z
    t2 = abs2((φa - φb) / z)
    abs2(σ) * (1 + t1 - t2)
end

function entropy(d::Truncated{Normal})
    d0 = d.untruncated
    z = d.tp
    μ = mean(d0)
    σ = std(d0)
    a = (d.lower - μ) / σ
    b = (d.upper - μ) / σ
    aφa = isinf(a) ? 0.0 : a * φ(a)
    bφb = isinf(b) ? 0.0 : b * φ(b)
    0.5 * (log2π + 1.) + log(σ * z) + (aφa - bφb) / (2.0 * z)
end


### sampling

function rand(d::Truncated{Normal})
    mu = mean(d.untruncated)
    sigma = std(d.untruncated)
    z = randnt((d.lower - mu) / sigma, (d.upper - mu) / sigma)
    return mu + sigma * z
end

# Rejection sampler based on algorithm from Robert (1992)
#  - Available at http://arxiv.org/abs/0907.4010
function randnt(lower::Real, upper::Real)
    if (lower <= 0 && upper == Inf) ||
       (upper >= 0 && lower == Inf) ||
       (lower <= 0 && upper >= 0 && upper - lower > sqrt2π)
        while true
            r = randn()
            if r > lower && r < upper
                return r
            end
        end
    elseif lower > 0 && upper - lower > 2.0 / (lower + sqrt(lower^2 + 4.0)) * exp((lower^2 - lower * sqrt(lower^2 + 4.0)) / 4.0)
        a = (lower + sqrt(lower^2 + 4.0))/2.0
        while true
            r = rand(Exponential(1.0 / a)) + lower
            u = rand()
            if u < exp(-0.5 * (r - a)^2) && r < upper
                return r
            end
        end    
    elseif upper < 0 && upper - lower > 2.0 / (-upper + sqrt(upper^2 + 4.0)) * exp((upper^2 + upper * sqrt(upper^2 + 4.0)) / 4.0)
        a = (-upper + sqrt(upper^2 + 4.0)) / 2.0
        while true
            r = rand(Exponential(1.0 / a)) - upper
            u = rand()
            if u < exp(-0.5 * (r - a)^2) && r < -lower
                return -r
            end
        end
    else
        while true
            r = lower + rand() * (upper - lower)
            u = rand()
            if lower > 0
                rho = exp((lower^2 - r^2) * 0.5)
            elseif upper < 0
                rho = exp((upper^2 - r^2) * 0.5)
            else
                rho = exp(-r^2 * 0.5)
            end
            if u < rho
                return r
            end
        end
    end
    return 0.0
end
