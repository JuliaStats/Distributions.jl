"""
    Hypergeometric(s, f, n)

A *Hypergeometric distribution* describes the number of successes in `n` draws without replacement from a finite population containing `s` successes and `f` failures.

```math
P(X = k) = {{{s \\choose k} {f \\choose {n-k}}}\\over {s+f \\choose n}}, \\quad \\text{for } k = \\max(0, n - f), \\ldots, \\min(n, s).
```

```julia
Hypergeometric(s, f, n)  # Hypergeometric distribution for a population with
                         # s successes and f failures, and a sequence of n trials.

params(d)       # Get the parameters, i.e. (s, f, n)
```

External links

* [Hypergeometric distribution on Wikipedia](http://en.wikipedia.org/wiki/Hypergeometric_distribution)

"""
struct Hypergeometric <: DiscreteUnivariateDistribution
    ns::Int     # number of successes in population
    nf::Int     # number of failures in population
    n::Int      # sample size

    function Hypergeometric(ns::Real, nf::Real, n::Real; check_args::Bool=true)
        @check_args(
            Hypergeometric,
            (ns, ns >= zero(ns)),
            (nf, nf >= zero(nf)),
            zero(n) <= n <= ns + nf,
        )
        new(ns, nf, n)
    end
end


@distr_support Hypergeometric max(d.n - d.nf, 0) min(d.ns, d.n)

partype(::Hypergeometric) = Int

### Parameters

params(d::Hypergeometric) = (d.ns, d.nf, d.n)


### Statistics

mean(d::Hypergeometric) = d.n * d.ns / (d.ns + d.nf)

function var(d::Hypergeometric)
    N = d.ns + d.nf
    p = d.ns / N
    d.n * p * (1.0 - p) * (N - d.n) / (N - 1.0)
end
mode(d::Hypergeometric) = floor(Int, (d.n + 1) * (d.ns + 1) / (d.ns + d.nf + 2))

function modes(d::Hypergeometric)
    if (d.ns == d.nf) && mod(d.n, 2) == 1
        [(d.n-1)/2, (d.n+1)/2]
    else
        [mode(d)]
    end
end

skewness(d::Hypergeometric) = (d.nf-d.ns)*sqrt(d.ns+d.nf-1)*(d.ns+d.nf-2*d.n)/sqrt(d.n*d.ns*d.nf*(d.ns+d.nf-d.n))/(d.ns+d.nf-2)

function kurtosis(d::Hypergeometric)
    ns = float(d.ns)
    nf = float(d.nf)
    n = float(d.n)
    N = ns + nf
    a = (N-1) * N^2 * (N * (N+1) - 6*ns * (N-ns) - 6*n*(N-n)) + 6*n*ns*(nf)*(N-n)*(5*N-6)
    b = (n*ns*(N-ns) * (N-n)*(N-2)*(N-3))
    a/b
end

entropy(d::Hypergeometric) = entropy(map(Base.Fix1(pdf, d), support(d)))

### Evaluation & Sampling

@_delegate_statsfuns Hypergeometric hyper ns nf n

## sampling

# Implements
#   V. Kachitvichyanukul & B. Schmeiser
#   "Computer generation of hypergeometric random variates"
#   Journal of Statistical Computation and Simulation, 22(2):127-145
#   doi:10.1080/00949658508810839
@rand_rdist(Hypergeometric)
function rand(d::Hypergeometric)
    # 0.0
    n1s, n2s, ks = d.ns, d.nf, d.n
    n1, n2, k = d.ns, d.nf, d.n
    n = n1+n2
    n1, n2 = minmax(n1, n2)
    k = min(k, n-k)
    y = _hyper(n, n1, n2, k)
    if 2ks < n
        if n1s <= n2s
            return y
        else
            return  ks - y
        end
    elseif n1s <= n2s
        return n1s - y
    else
        return ks - n2s + y
    end
end

function _hyper(n, n1, n2, k)
    # 0.1 Use HIN algorithm if fast
    M = fld((k+1)*(n1+1), n+2)
    if M - max(0, k-n2) < 10
        if k < n2
            p = gamma(n2+1)*gamma(n1+n2-k+1)/(gamma(n1+n2+1)*gamma(n2-k+1))
            x = 0
        else
            p = gamma(n1+1)*gamma(k+1)/(gamma(n1+n2+1)*gamma(k-n2+1))
            x = k-n2
        end
        u = rand()
        while u > p
            u -=p
            p *= (n1-x)*(k-x)/((x+1)*(n2-k+1+x))
            x += 1
        end
        return x
    end
    A = loggamma(M+1) + loggamma(n1-M+1) + loggamma(k-M+1) + loggamma(n2-k+M+1)
    D = 1.5*sqrt((n-k)*k*n1*n2/((n-1)*n^2))+0.5
    xL, xR = M-D+0.5, M+D+0.5
    kL = exp(A - loggamma(xL+1) - loggamma(n1-xL+1) - loggamma(k-xL+1) - loggamma(n2-k+xL+1))
    kR = exp(A - loggamma(xR) - loggamma(n1-xR+2) - loggamma(k-xR+2) - loggamma(n2-k+xR))
    λL = -log(xL*(n2-k+xL)/((n1-xL+1)*(k-xL+1)))
    λR = -log((n1-xR+1)*(k-xR+1)/(xR*(n2-k+xR)))
    p1 = 2D
    p2 = p1 + kL/λL
    p3 = p2 + kR/λR
    while true
        # Step 1:
        # Begin logic to generate one hypergeometric variate y.
        # Generate u U(0, p,) for selecting the region, v .v U(0,l) for
        # the acceptlreject decision. If region 1 is selected, generate a
        # uniform variate between xL, and xR.
        u, v = p3*rand(), rand()
        local y::Int
        if u <= p1
            y = floor(Int, xL + u)
        elseif u <= p2
                # Step 2: left exponential tail
                y = floor(Int, xL + log(v)/λL)
                y < max(0, k-n2) && continue
                v *= (u-p1)*λL
        else
            # Step 3. Right exponential tail
            y = floor(Int, xR - log(v)/λR)
            y > min(n1, k) && continue
            v *= (u-p2)*λR
        end
        # Step 4: acceptance/rejection comparison
        if false && (M < 100 || y <= 50) #this optimization has a typo somewhere in the paper I think
            # Evaluate f (y) via recursive relationship
            # f(y+1) = f(y)(n1-y)(k-y)/((y+1)(n2-y+1))
            f = if M < y
                prod((n1-i+1)*(k-i+1)/(i*(n2-k+i)) for i in M+1:y)
            elseif y < M
                prod((i+1)*(n2-k+i)/((n1-i)*(k-i)) for i in y:M)
            else
                1.0
            end
            v > f && continue
        else
            # 4.2 Squeezing: Check the value of ln(v) against upper and lower bound of ln(f(y))
            y1 = y+1
            yM = y-M
            yn = n1-y+1
            yk = k-y+1
            nk = n2-k+y1
            R=-yM/y1
            S=yM/yn
            T=yM/yk
            E=-yM/nk
            G=yn*yk/(y1*nk)-1
            DG=1
            if G < 0
                DG = 1+G
            end
            coefs = (0.0, 1.0, -1/2, 1/3)
            GU  = evalpoly(G, coefs)
            GL = GU-G^4/(4*DG)
            XM = M+.5
            Xn = n1-M+.5
            Xk = k-M+.5
            nM=n2-k+XM
            Ub = XM*evalpoly(R, coefs) + Xn*evalpoly(S, coefs) + Xk*evalpoly(T, coefs) + nM*evalpoly(E, coefs) + y*GU-M*GL+0.0034
            Av=log(v)
            Av > Ub && continue
            DR = XM*R^4
            if R < 0
                DR = DR/(1+R)
            end
            DS = Xn*S^4
            if S < 0
                DS = DS/(1+S)
            end
            DT = Xk*T^4
            if T < 0
                DT = DT/(1+T)
            end
            DE = nM*E^4
            if E < 0
                DE = DE/(1+E)
            end
            if Av < Ub - 0.25*(DR+DS+DT+DE) + (y+M)*(GL-GU)-0.0078
                return y
            end
            # 4.3 final rejection step
            if Av > A - loggamma(y+1) - loggamma(n1-y+1) - loggamma(k-y+1) - loggamma(n2 - k + y + 1)
                continue
            end
        end
        return y
    end
end
