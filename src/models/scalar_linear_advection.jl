struct ScalarLinearAdvection{T, Dim} <: AbstractModel
    velocity::SVector{Dim, T}
end

ScalarLinearAdvection(v) = ScalarLinearAdvection(SVector{length(v), eltype(v)}(v...))

Base.eltype(m::ScalarLinearAdvection{T, D}) where {T, D} = T
nb_dims(m::ScalarLinearAdvection{T, D}) where {T, D} = D
nb_vars(m::ScalarLinearAdvection{T, D}) where {T, D} = 1
w_names(m::ScalarLinearAdvection{T, D}) where {T, D} = (:α,)

rotate_model(m::ScalarLinearAdvection, rotation_matrix) = ScalarLinearAdvection(rotation_matrix * m.velocity)

function normal_flux(m::ScalarLinearAdvection{T, D}, w, wsupp) where {T, D}
    return typeof(w)(w[1] * m.velocity[1])
    # Only the first coordinate in the frame of the interface, i.e. the normal vector
end

eigenvalues(m::ScalarLinearAdvection{T, D}, w, wsupp) where {T, D} = SVector{1, T}(m.velocity[1])
left_eigenvectors(m::ScalarLinearAdvection{T, D}, w, wsupp) where {T, D} = SMatrix{1, 1, T}(1.0)
right_eigenvectors(m::ScalarLinearAdvection{T, D}, w, wsupp) where {T, D} = SMatrix{1, 1, T}(1.0)
