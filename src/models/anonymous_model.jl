# Anonymous = only defined by a flux function

struct AnonymousModel{N, D, T, S} <: AbstractModel
    flux::Function
end
AnonymousModel{N, D, T}(f) where {N, D, T} = AnonymousModel{N, D, T, false}(f)
AnonymousModel{N, D}(f) where {N, D} = AnonymousModel{N, D, Float64}(f)

nb_vars(m::AnonymousModel{N}) where N = N
nb_dims(m::AnonymousModel{N, D}) where {N, D} = D
Base.eltype(m::AnonymousModel{N, D, T}) where {N, D, T} = T
is_space_dependant(m::AnonymousModel{N, D, T, S}) where {N, D, T, S} = S

w_names(m::AnonymousModel{1}) = (:u,)
w_names(m::AnonymousModel{N}) where N = Tuple(Symbol("u_$i") for i in 1:N)

function rotate_model(m::AnonymousModel{N, D, T, S}, rotation_matrix, position=nothing) where {N, D, T, S}
    fx_at_face = u -> is_space_dependant(m) ? m.flux(u, position) : m.flux(u)
    if D == 1
        fx = u -> rotation_matrix[1, 1] .* fx_at_face(u)
    elseif D > 1
        fx = u -> (rotation_matrix' * fx_at_face(u))[1]
    end
    AnonymousModel{N, 1, T, S}(fx)
end

normal_flux(m::AnonymousModel{N, 1, T}, w) where {N, T} = m.flux(w)

# Support for Float instead of single-element SVector
jacobian(m::AnonymousModel{1, 1, T}, w::Number) where {T} = ForwardDiff.derivative(w -> normal_flux(m, w), w)
eigenvalues(m::AnonymousModel{1, 1, T}, w::Number) where {T} = jacobian(m, w)
left_eigenvectors(m::AnonymousModel{1, 1, T}, w::Number) where {T} = 1.0
right_eigenvectors(m::AnonymousModel{1, 1, T}, w::Number) where {T} = 1.0
