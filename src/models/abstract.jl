abstract type AbstractModel end

# Functions to be implemented
function nb_dims end
function nb_vars end

function flux end
function eigenvalues end
function left_eigenvectors end
function right_eigenvectors end

# Optional function
function directional_splitting end

# Default behavior: conservative variables are the main variables
compute_v(m::AbstractModel, w) = w
consvartype(m::AbstractModel, w) = eltype(w)
invert_v(m::AbstractModel, v) = v

# Default behavior: invariant by rotation
rotate_model(m::AbstractModel, rotation_matrix, x=nothing) = m
rotate_state(w, m::AbstractModel, rotation_matrix) = w
rotate_flux(F, m::AbstractModel, rotation_matrix) = F

function compute_w_int(m::AbstractModel, w_L, w_R)
    w_mean = (w_L + w_R)/2
    return w_mean
end

