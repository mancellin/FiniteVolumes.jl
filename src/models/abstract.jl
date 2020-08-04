abstract type AbstractModel end

import Base.eltype

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

# Fallback behavior: automatic differentiation of the flux
using ForwardDiff
jacobian(m::AbstractModel, w) = ForwardDiff.jacobian(v -> normal_flux(m, invert_v(m, v)), compute_v(m, w))

# Fallback behavior: numerical computation of the eigenstructure
# TODO: improve performance by not recomputing several time
eigenvalues(m::AbstractModel, w) = jacobian(m, w) |> Array |> eigvals
right_eigenvectors(m::AbstractModel, w) = jacobian(m, w) |> Array |> eigen |> e -> e.vectors
left_eigenvectors(m::AbstractModel, w) = right_eigenvectors(m, w) |> inv
