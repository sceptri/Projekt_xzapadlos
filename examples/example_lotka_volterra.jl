using Revise
using SINDyAndBeyond
using ModelingToolkit, DifferentialEquations
using Plots, Random

# ------- Generating data ---------

@variables t x(t) y(t)
@parameters α, β, δ, γ

D = Differential(t)

# Equations of the Lorenz System
ẋ(x, y, α, β) = α * x - β * x * y
ẏ(x, y, δ, γ) = -δ * y + γ * x * y

# Define parameters, starting values and time range for our data
params = [
    α => 1.5,
    β => 1.0,
    δ => 3.0,
    γ => 1.0
]
starting_values = [1.0, 1.0]
time_range = (0.0, 10.0)

@named ode_system = ODESystem([
    D(x) ~ ẋ(x, y, α, β),
    D(y) ~ ẏ(x, y, δ, γ),
])
ode_problem = ODEProblem(ode_system, starting_values, time_range, params)
solution = solve(ode_problem, Tsit5(); saveat=0.1)

# Compute the derivatives from the data
derivatives = [[
    ẋ(x, y, Dict(params)[α], Dict(params)[β]),
    ẏ(x, y, Dict(params)[δ], Dict(params)[γ]),
] for (x, y) in solution.u]

# transform vectors of states and derivatives into a matrices
X = vcat(reshape.(solution.u, 1, 2)...)
Ẋ = vcat(reshape.(derivatives, 1, 2)...)

# ------- Learning from data ---------

basis = PolynomialLibrary(1, 2)

# Regression optimizer
optimizer = LASSO(τ=1e-1, μ=500, ρ=ρ)
Ξ = discover(X, Ẋ, basis, optimizer; max_iter=100)
prettyprint(Ξ, basis)

# Diff Eq Optimizer
# TODO: Clean this up
optimizer = NeuralODE(η=1)
Random.seed!(666)
Ξ = discover(X, basis, optimizer;
    timespan=time_range,
    u₀=starting_values,
    max_iter=1000,
    ylims=(0, 6)
)
prettyprint(Ξ, basis)