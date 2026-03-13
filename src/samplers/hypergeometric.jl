"""
### Hypergeometric Sampler

***Sampling implementation reference:*** V. Kachitvichyanukul & B. Schmeiser
  "Computer generation of hypergeometric random variates"
  Journal of Statistical Computation and Simulation, 22(2):127-145
  doi:10.1080/00949658508810839
"""
struct HypergeometricSampler{D} <: Sampleable{Univariate,Discrete}
    d::D
    useHIN::Bool
    ns::Int
    nf::Int
    n::Int
    M::Int
    A::Float64
    xL::Float64
    xR::Float64
    λL::Float64
    λR::Float64
    p1::Float64
    p2::Float64
    p3::Float64
end

function HypergeometricSampler(d::Hypergeometric)
    (;ns, nf, n) = d
    ntotal = ns+nf
    ns, nf = minmax(ns, nf)
    n = min(n, ntotal - n)
    M = fld((n + 1) * (ns + 1), ntotal + 2)
    if M - max(0, n - nf) < 10
        return HypergeometricSampler(d, true, ns, nf, n, M, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    else
        A = loggamma(M + 1) +
            loggamma(ns - M + 1) +
            loggamma(n - M + 1) +
            loggamma(nf - n + M + 1)
        D = 1.5 * sqrt((ntotal - n) * n * ns * nf / ((ntotal - 1) * ntotal^2)) + 0.5
        xL = M - D + 0.5
        xR = M + D + 0.5
        nL = exp(
            A -
            loggamma(xL + 1) -
            loggamma(ns - xL + 1) -
            loggamma(n - xL + 1) -
            loggamma(nf - n + xL + 1)
        )
        nR = exp(
            A -
            loggamma(xR) -
            loggamma(ns - xR + 2) -
            loggamma(n - xR + 2) -
            loggamma(nf - n + xR)
        )
        λL = -log(xL * (nf - n + xL) / ((ns - xL + 1) * (n - xL + 1)))
        λR = -log((ns - xR + 1) * (n - xR + 1) / (xR * (nf - n + xR)))
        p1 = 2D
        p2 = p1 + nL / λL
        p3 = p2 + nR / λR
        
        return HypergeometricSampler(d, false, ns, nf, n, M, A, xL, xR, λL, λR, p1, p2, p3)
    end
end

function hyper_fixup(y::Int, spl::HypergeometricSampler)
    d = spl.d
    if 2*d.n < d.ns + d.nf
        if d.ns <= d.nf
            return y
        else
            return  d.n - y
        end
    elseif d.ns <= d.nf
        return d.ns - y
    else
        return d.n - d.nf + y
    end
end

function Random.rand(rng::AbstractRNG, spl::HypergeometricSampler)
    (; ns, nf, n, useHIN) = spl
    ntotal = ns+nf
    if useHIN
        if n < nf
            p = gamma(nf + 1) * gamma(ntotal - n + 1) / (gamma(ntotal + 1) * gamma(nf - n + 1))
            y = 0
        else
            p = gamma(ns + 1) * gamma(n + 1) / (gamma(ntotal + 1) * gamma(n - nf + 1))
            y = n - nf
        end
        u = rand(rng)
        while u > p
            u -= p
            p *= (ns - y) * (n - y) / ((y + 1) * (nf - n + y + 1))
            y += 1
        end
        return hyper_fixup(y, spl)
    end
    (; M, A, xL, xR, λL, λR, p1, p2, p3) = spl
    while true
        # Step 1:
        # Begin logic to generate hypergeometric variate y.
        # Generate u = U(0, p) for selecting the region
        # v = U(0,1) for the accept reject decision. 
        u = p3 * rand(rng)
        v = rand(rng)
        local y::Int
        if u <= p1
            # Region 1: is selected, generate a
            # uniform variate between xL, and xR.
            y = floor(Int, xL + u)
        elseif u <= p2
            # Region 2: left exponential tail
            y = floor(Int, xL + log(v) / λL)
            y < max(0, n - nf) && continue
            v *= (u - p1) * λL
        else
            # Region 3. Right exponential tail
            y = floor(Int, xR - log(v) / λR)
            y > min(ns, n) && continue
            v *= (u - p2) * λR
        end
        # Step 4: acceptance/rejection comparison
        # TODO: as an optimization, if M<100 or y<50
        # Evaluate f (y) via recursive relationship
        # f(y+1) = f(y)(ns-y)(n-y)/((y+1)(nf-y+1))
        
        # 4.2 Squeezing: Check the value of ln(v) against upper and lower bound of ln(f(y))
        logv = log(v)
        yn = ns - y + 1
        yk = n - y + 1
        nk = nf - n + y + 1
        RSTE = (y - M) ./ (- y - 1, yn, yk, -nk)
        G = yn * yk / muladd(y, nk, nk) - 1

        coefs = (0.0, 1.0, -1/2, 1/3)
        GU  = evalpoly(G, coefs)
        # use (G^2)^2 instead of G^4 since it is faster and we don't care about the tiny inaccuracy introduced
        GL = GU - 0.25 * (G^2)^2 / (1 + max(0, G))
        
        XMSTE = (M, ns - M, n - M, nf - n + M) .+ 0.5
        Ub = sum(XMSTE .* evalpoly.(RSTE, Ref(coefs))) + y * GU - M * GL + 0.0034
        logv > Ub && continue
        
        DRSTE = XMSTE .* (RSTE.^2).^2 ./ (1 .+ min.(RSTE, 0.0))
        if logv < Ub - 0.25 * sum(DRSTE) + (y + M) * (GL - GU) - 0.0078
            return hyper_fixup(y, spl)
        end
        # 4.3 final rejection step
        if logv > A - loggamma(y + 1) - loggamma(ns - y + 1) - loggamma(n - y + 1) - loggamma(nf - n + y + 1)
            continue
        end
        return hyper_fixup(y, spl)
    end
end
