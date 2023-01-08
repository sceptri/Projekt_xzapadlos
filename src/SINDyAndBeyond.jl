module SINDyAndBeyond

using Printf

include("library.jl")

export sparsifyDynamics, discover, prettyprint

variables(X; iterate_by=eachrow) = length(iterate_by(X'))

# TODO: Add comments and docstring
"""
Arguments:
- Θ is our data transformed by our selected library of candidate functions
- Ẋ is the derivative in our data
- λ is the *sparsification knob*
"""
function sparsifydynamics(Θ, Ẋ; λ, kwargs...)
    # n may be passed via keyword arguments
    n = !(@isdefined n) ? variables(Ẋ; kwargs...) : n
    Ξ = Θ \ Ẋ

    # TODO: Rework as general optimizer concept
    for _ in 1:10
        small_indices = @. abs(Ξ) < λ
        Ξ[small_indices] .= 0

        for index in 1:n
            big_indices_in = @. !small_indices[:, index]
            Ξ[big_indices_in, index] .= Θ[:, big_indices_in] \ Ẋ[:, index]
        end
    end

    return Ξ
end

# TODO: return a custom struct
function discover(X, Ẋ, library::T; kwargs...) where {T<:Library}
    Θ = library(X)
    Ξ = sparsifydynamics(Θ, Ẋ; kwargs...)
    return Ξ
end

# TODO: move pretty printing away from the main file
function prettyprint_singular(coefficient, power_vector, variables_names)
    coef = @sprintf "%.2f" coefficient
    if all(power_vector .== 0)
        return coef
    end

    terms = ["$(variables_names[index])" * (power > 1 ? "^$(power)" : "")
             for (index, power) in enumerate(power_vector)
             if power > 0
    ]

    return join([coef, terms...], "⋅")
end

function prettyprint(Ξ, library::PredefinedLibrary, variables_names=["x", "y", "z", "w"])
    used_variables = variables(Ξ)
    if length(variables_names) < used_variables
        error("Not enough variables' names supplied")
    end

    println("Found equations for the system are:")
    for var in 1:used_variables
        indices = findall(Ξ[:, var] .!= 0)
        right_side_terms = [
            prettyprint_singular(Ξ[index, var], deconstruct_state(library, index), variables_names)
            for index in indices
        ]
        right_side = join(right_side_terms, " + ")


        # There is a dot (\dot) after variable name
        println("\t$(variables_names[var])̇ = $(right_side)")
    end
end

end