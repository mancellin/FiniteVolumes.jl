abstract type AbstractModel end

# Functions to be implemented
function nb_dims end
function nb_vars end
function nb_vars_supp end
function w_names end

function flux end
function eigenvalues end
function left_eigenvectors end
function right_eigenvectors end

# Optional function
function directional_splitting end

# Default behavior: no supplementary variables
nb_vars_supp(m::AbstractModel) = 0
wsupp_names(m::AbstractModel) = Tuple([])
compute_wsupp(m::AbstractModel, w::Union{<:Number, <:SVector}) = nothing
compute_wsupp(m::AbstractModel, w::Vector) = foreach(wi -> compute_wsupp(m, wi), w)

# Default behavior: conservative variables are the main variables
compute_v(m::AbstractModel, w) = w
invert_v(m::AbstractModel, v) = v

# Default behavior: invariant by rotation
rotate_model(m::AbstractModel, rotation_matrix) = m
rotate_state(w, m::AbstractModel, rotation_matrix) = w
rotate_flux(F, m::AbstractModel, rotation_matrix) = F

function compute_w_int(m::AbstractModel, w_L, w_R)
    w_mean = (w_L + w_R)/2
    compute_wsupp(m, w_mean)
    return w_mean
end

