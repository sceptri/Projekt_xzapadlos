using Printf

export prettyprint

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