# Anonymous = only defined by a flux function

struct AnonymousModel{T, D, S} <: AbstractModel
    flux::Function # si S: (T, SVector{D}) → T, sinon: T → T
end
AnonymousModel{T, D}(f) where {T, D} = AnonymousModel{T, D, false}(f)

Base.eltype(::AnonymousModel{T}) where {T} = T
nb_dims(::AnonymousModel{T, D}) where {T, D} = D
is_space_dependant(::AnonymousModel{T, D, S}) where {T, D, S} = S

function directional_splitting(m::AnonymousModel{T, 2, S}) where {T, S}
    if !is_space_dependant(m)
        horizontal_flux = u -> [m.flux(u)[1], zero(T)]
        vertical_flux = u -> [zero(T), m.flux(u)[2]]
    else
        horizontal_flux = (u, x) -> [m.flux(u, x)[1], zero(T)]
        vertical_flux = (u, x) -> [zero(T), m.flux(u, x)[2]]
    end
    return [AnonymousModel{T, 2, S}(horizontal_flux),
            AnonymousModel{T, 2, S}(vertical_flux)]
end

function rotate_model(m::AnonymousModel{T, D, S}, rotation_matrix, position=nothing) where {T, D, S}
    if !is_space_dependant(m)
        fx_at_face = u -> m.flux(u)
    else
        fx_at_face = u -> m.flux(u, position)
    end
    if D == 1
        # Need a special case so that the 2-scalars model u -> [u[1], u[2]]
        # is not interpreted as a 2D flux in 1D.
        local_flux = u -> rotation_matrix[1, 1] * fx_at_face(u)
    else
        local_flux = u -> (rotation_matrix' * fx_at_face(u))[1]
    end
    AnonymousModel{T, 1, S}(local_flux)
end

normal_flux(m::AnonymousModel{T, 1}, w) where {T} = m.flux(w)
