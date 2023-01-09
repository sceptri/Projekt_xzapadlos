using Revise
using SINDyAndBeyond
using ModelingToolkit, DifferentialEquations

# ------- Generating data ---------

@variables t x(t) y(t) z(t)
@parameters σ, ρ, β

D = Differential(t)

# Equations of the Lorenz System
ẋ(x, y, z, σ) = σ * (y - x)
ẏ(x, y, z, ρ) = x * (ρ - z) - y
ż(x, y, z, β) = x * y - β * z

# Define parameters, starting values and time range for our data
params = [
    σ => 10,
    ρ => 28,
    β => 8 / 3
]
starting_values = [1, 1, 1]
time_range = (0.0, 30)

@named ode_system = ODESystem([
    D(x) ~ ẋ(x, y, z, σ),
    D(y) ~ ẏ(x, y, z, ρ),
    D(z) ~ ż(x, y, z, β)
])
ode_problem = ODEProblem(ode_system, starting_values, time_range, params)
solution = solve(ode_problem, Tsit5(); saveat=0.01)

# Compute the derivatives from the data
derivatives = [[
    ẋ(x, y, z, Dict(params)[σ]),
    ẏ(x, y, z, Dict(params)[ρ]),
    ż(x, y, z, Dict(params)[β])
] for (x, y, z) in solution.u]

# transform vectors of states and derivatives into a matrices
X = vcat(reshape.(solution.u, 1, 3)...)
Ẋ = vcat(reshape.(derivatives, 1, 3)...)

# ------- Learning from data ---------

basis = PolynomialLibrary(2, 3)
optimizer = LASSO(τ=1e-1, μ=500, ρ=ρ)
Ξ = discover(X, Ẋ, basis, optimizer; max_iter=100)
prettyprint(Ξ, basis)


# Comparison of different ρs
ρ_range = 0.01:0.05:1
L = zeros(3, length(ρ_range))

for (index, ρ) in enumerate(ρ_range)
    optimizer = LASSO(τ=1e-1, μ=500, ρ=ρ)
    Ξ = discover(X, Ẋ, basis, optimizer; max_iter=100)

    L[:, index] .= L₂(Ẋ', (basis(X) * Ξ)')
end

using Plots
plot(L')