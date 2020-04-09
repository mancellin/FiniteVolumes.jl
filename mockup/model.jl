abstract type AbstractModel end

nb_vars(s::AbstractModel) = nb_vars(typeof(s))
nb_vars_supp(s::AbstractModel) = nb_vars_supp(typeof(s))
w_names(s::AbstractModel) = w_names(typeof(s))
wsupp_names(s::AbstractModel) = wsupp_names(typeof(s))

struct LocalState{ModelType, T}
    main::SVector{nb_vars(ModelType), T}
    supp::SVector{nb_vars_supp(ModelType), T}
end

function getproperty(w::LocalState{ModelType, T} where {ModelType, T}, name::Symbol)
    i_name = find(==(name), w_names(ModelType))
    if !(isnothing(i_name))
        return w.main[i_name]
    end
    i_name = find(==(name), wsupp_names(ModelType))
    if !(isnothing(i_name))
        return w.main[i_name]
    end
    return error("No field $name found.")
end

propertynames(w::LocalState{M, T} where {M, T}) = [w_names(M)..., wsupp_names(M)...]
