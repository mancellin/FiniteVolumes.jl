# Anonymous = only defined by a flux function

struct AnonymousModel{N, D, T} <: AbstractModel
    flux::Function
end

nb_vars(m::AnonymousModel{N, D, T}) where {N, D, T} = N
Base.eltype(m::AnonymousModel{N, D, T}) where {N, D, T} = T
nb_dims(m::AnonymousModel{N, D, T}) where {N, D, T} = D

w_names(m::AnonymousModel{1, D, T}) where {D, T} = (:u,)
w_names(m::AnonymousModel{N, D, T}) where {N, D, T} = Tuple(Symbol("u_$i") for i in 1:N)

function rotate_model(m::AnonymousModel{N, D, T}, rotation_matrix, position=nothing) where {N, D, T}
    if D == 1
        fx = u -> rotation_matrix[1, 1] .* m.flux(u)
    elseif D > 1
        fx = u -> (rotation_matrix' * m.flux(u))[1]
    end
    AnonymousModel{N, 1, T}(fx)
end

normal_flux(m::AnonymousModel{N, 1, T}, w) where {N, T} = m.flux(w)

using ForwardDiff
using LinearAlgebra

jacobian(m::AnonymousModel{N, 1, T}, w) where {N, T} = ForwardDiff.jacobian(w -> normal_flux(m, w), w)
eigenvalues(m::AnonymousModel{N, 1, T}, w) where {N, T} = jacobian(m, w) |> Array |> eigvals
right_eigenvectors(m::AnonymousModel{N, 1, T}, w) where {N, T} = jacobian(m, w) |> Array |> eigen |> e -> e.vectors
left_eigenvectors(m::AnonymousModel{N, 1, T}, w) where {N, T} = right_eigenvectors(m, w) |> inv

# Support for Float instead of single-element SVector
jacobian(m::AnonymousModel{1, 1, T}, w::Number) where {T} = ForwardDiff.derivative(w -> normal_flux(m, w), w)
eigenvalues(m::AnonymousModel{1, 1, T}, w::Number) where {T} = jacobian(m, w)
left_eigenvectors(m::AnonymousModel{1, 1, T}, w::Number) where {T} = 1.0
right_eigenvectors(m::AnonymousModel{1, 1, T}, w::Number) where {T} = 1.0
