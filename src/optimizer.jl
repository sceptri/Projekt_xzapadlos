using LinearAlgebra, Flux, Plots, SciMLSensitivity

export STLSQ, LASSO, NeuralODE

abstract type Optimizer end

abstract type RegressionOptimizer <: Optimizer end

"""
Optimizer is a functor, which when called, tries to regress the library
of candidate terms onto the derivative

Arguments:
- Θ is our data transformed by our selected library of candidate functions
- Ẋ is the derivative in our data
"""
(::RegressionOptimizer)(θ, Ẋ; kwargs...) = nothing

"""
Fields:
- τ determines, how small can a value be before being zeroed
"""
Base.@kwdef struct STLSQ <: RegressionOptimizer
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

"""
LASSO using ADMM algorithm

Implemented per [the lecture](https://juliateachingctu.github.io/Julia-for-Optimization-and-Learning/stable/lecture_12/sparse/) and [this article](https://www.stat.cmu.edu/~ryantibs/convexopt-F18/lectures/admm.pdf)
"""
Base.@kwdef struct LASSO <: RegressionOptimizer
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

Implemented per [the lecture](https://juliateachingctu.github.io/Julia-for-Optimization-and-Learning/stable/lecture_12/sparse/) and [this article](https://www.stat.cmu.edu/~ryantibs/convexopt-F18/lectures/admm.pdf)
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

abstract type DiffEqOptimizer <: Optimizer end

Base.@kwdef struct NeuralODE <: DiffEqOptimizer
    hidden = 5
    η = 0.1
end

function (optimizer::NeuralODE)(Θ_problem::T, X, saveat; max_iter=150, kwargs...) where {T<:ODEProblem}
    parametric_Θ = function (p)
        p_problem = remake(Θ_problem, p=p[1])
        return solve(p_problem, Tsit5(), saveat=saveat)
    end

    Θ_params = Flux.params(Θ_problem.p)

    predict() = parametric_Θ(Θ_params)
    loss = function ()
        Y = vcat(reshape.(predict().u, 1, variables(X))...)
        Y_length = size(Y, 1)
        return sum(abs2, X[1:Y_length, :] - Y)# + sum(abs, Θ_params[1])
		# return sum(abs2, 1 .- Y) + sum(abs,Θ_params[1])
    end

    # we don't really need any input data, as the data is included in the loss function
    data = Iterators.repeated((), max_iter)
    optimization_alg = ADAM(optimizer.η)

    cb = function () #callback function to observe training
        display(loss())
        # using `remake` to re-create our `prob` with current parameters `p`
        display(plot(solve(remake(Θ_problem, p=Θ_params[1]), Tsit5(), saveat=saveat); kwargs...))
    end

    # Display the ODE with the initial parameter values.
    cb()

    Flux.train!(loss, Θ_params, data, optimization_alg, cb=cb)

    return reshape(Θ_params[1], length(Θ_params[1]) ÷ Int(variables(X)), Int(variables(X)))
end