using LinearAlgebra

export STLSQ, LASSO

abstract type Optimizer end

"""
Optimizer is a functor, which when called, tries to regress the library
of candidate terms onto the derivative

Arguments:
- Θ is our data transformed by our selected library of candidate functions
- Ẋ is the derivative in our data
"""
(::Optimizer)(θ, Ẋ; kwargs...) = nothing

"""
Fields:
- τ determines, how small can a value be before being zeroed
"""
Base.@kwdef struct STLSQ <: Optimizer
    τ = 0.025
end

function (optimizer::STLSQ)(Θ, Ẋ; max_iter=10, kwargs...)
    vars = variables(Ẋ; kwargs...)
    # Compute LSQ as the initial guess of our system
    Ξ = Θ \ Ẋ

    for _ in 1:max_iter
        small_indices = @. abs(Ξ) < optimizer.τ
        Ξ[small_indices] .= 0

        # For each variable, do the thresholded LSQ
        for variable in 1:vars
            big_indices_in = @. !small_indices[:, variable]
            Ξ[big_indices_in, variable] .= Θ[:, big_indices_in] \ Ẋ[:, variable]
        end
    end

    return Ξ
end

Base.@kwdef struct LASSO <: Optimizer
    μ = 500
    ρ = 1e3
    τ = 1e-2
end

"""
Soft thresholding operator
"""
S(z, η) = max(z - η, 0) - max(-z - η, 0)

"""
LASSO using ADMM algorithm

Implemnted per [the lecture](https://juliateachingctu.github.io/Julia-for-Optimization-and-Learning/stable/lecture_12/sparse/) and [this article](https://www.stat.cmu.edu/~ryantibs/convexopt-F18/lectures/admm.pdf)
"""
function (optimizer::LASSO)(Θ, Ẋ; max_iter=100, kwargs...)
    λ, Q = eigen(Θ' * Θ)
    Q_inv = Matrix(Q') # Q is orthonormal
    ρ, μ = optimizer.ρ, optimizer.μ

    Ξ = zeros(variables(Θ), variables(Ẋ))

    for variable in 1:variables(Ẋ)
        ω, u, z = [zeros(variables(Θ)) for _ in 1:3]

        for _ in 1:max_iter
            ω = Q * ((Diagonal(1 ./ (λ .+ ρ)) * (Q_inv * (Θ' * Ẋ[:, variable] + ρ * (z - u)))))
            z = S.(ω + u, μ / ρ)
            u = u + ω - z
        end

        Ξ[:, variable] .= ω
    end

    small_indices = @. abs(Ξ) < optimizer.τ
    Ξ[small_indices] .= 0

    return Ξ
end