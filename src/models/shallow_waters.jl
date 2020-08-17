
struct ShallowWater{D, T} <: AbstractModel
    g::T
end


Base.eltype(m::ShallowWater{D, T}) where {D, T} = T
nb_dims(m::ShallowWater{D, T}) where {D, T} = D
nb_vars(m::ShallowWater{D, T}) where {D, T} = 1 + D
consvartype(m::ShallowWater{D, T}, w) where {D, T} = SVector{1 + D, T}

########################################

w_names(m::ShallowWater{1, T}) where T = (:h, :u)
w_names(m::ShallowWater{2, T}) where T = (:h, :ux, :uy)

function rotate_state(w, m::ShallowWater{1, T}, rotation_matrix) where T
    return (h=w.h, u=w.u * rotation_matrix[1, 1])
end

function rotate_state(w, m::ShallowWater{2, T}, rotation_matrix) where T
    return (h=w.h, 
            ux=rotation_matrix[1, 1] * w.ux + rotation_matrix[1, 2] * w.uy,
            uy=rotation_matrix[2, 1] * w.ux + rotation_matrix[2, 2] * w.uy,
           )
end

function compute_v(m::ShallowWater{1, T}, w::NamedTuple) where T
    h, u = w.h, w.u
    return SVector{2, eltype(m)}(h, h*u)
end

function compute_v(m::ShallowWater{2, T}, w::NamedTuple) where T
    return SVector{3, eltype(m)}(w.h, w.h*w.ux, w.h*w.uy)
end

invert_v(m::ShallowWater{1, T}, v) where T = (h=v[1], u=v[2]/v[1])

invert_v(m::ShallowWater{2, T}, v) where T = (h=v[1], ux=v[2]/v[1], uy=v[3]/v[1])

function normal_flux(m::ShallowWater{1, T}, w::NamedTuple) where T
    h, u = w.h, w.u
    return SVector(h*u, h*u^2 + h^2*m.g/2)
end

function normal_flux(m::ShallowWater{2, T}, w::NamedTuple) where T
    h, ux, uy = w.h, w.ux, w.uy
    return SVector(h*ux, h*ux*ux + h^2*m.g/2, h*ux*uy)
end

compute_v(m::ShallowWater{1, T}, w::AbstractArray) where T = compute_v(m, (h=w[1], u=w[2]))
normal_flux(m::ShallowWater{1, T}, w::AbstractArray) where T = normal_flux(m, (h=w[1], u=w[2]))

compute_v(m::ShallowWater{2, T}, w::AbstractArray) where T = compute_v(m, (h=w[1], ux=w[2], uy=w[3]))
normal_flux(m::ShallowWater{2, T}, w::AbstractArray) where T = normal_flux(m, (h=w[1], ux=w[2], uy=w[3]))

function rotate_flux(F, m::ShallowWater{1, T}, rotation_matrix) where T
    SVector{2, T}(F[1], F[2] * rotation_matrix[1, 1])
end

function rotate_flux(F, m::ShallowWater{2, T}, rotation_matrix) where T
    SVector{3, T}(F[1], 
                  rotation_matrix[1, 1] * F[2] + rotation_matrix[1, 2] * F[3],
                  rotation_matrix[2, 1] * F[2] + rotation_matrix[2, 2] * F[3],
                 )
end
