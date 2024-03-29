# const Scalar = Union{Number, StaticArrays.Scalar, SVector{1}}

jacobian(f, w::Number, n) = ForwardDiff.derivative(v -> f(v, n), w)
jacobian(f, w::AbstractVector, n) = ForwardDiff.jacobian(v -> f(v, n), w)

eigvals(f, w, n) = real.(LinearAlgebra.eigen(jacobian(f, w, n)).values)
function eigen(f, w, n)
    λc, Rc = LinearAlgebra.eigen(jacobian(f, w, n))
    return real.(λc), real.(Rc)
end

##############################

struct LinearAdvectionFlux{V}
    velocity::V
end
LinearAdvectionFlux(v::AbstractVector) = LinearAdvectionFlux{SVector{length(v), eltype(v)}}(SVector(v...))

(f::LinearAdvectionFlux)(w, n) = w * (f.velocity' * n)
jacobian(f::LinearAdvectionFlux, w, n) = f.velocity' * n
eigvals(f::LinearAdvectionFlux, w, n) = f.velocity' * n

##############################

struct AdvectionFlux{D, VF}
    velocity_at_face::VF
end

AdvectionFlux{D}(vf) where D = AdvectionFlux{D, typeof(vf)}(vf)

function numerical_flux(flux::AdvectionFlux, mesh, w, scheme, i_face, dt)
    numerical_flux(LinearAdvectionFlux(flux.velocity_at_face(mesh, i_face)), mesh, w, scheme, i_face, dt)
end

function RotationFlux(center)
    function rotation_velocity(mesh, i_face)
        x = face_center(mesh, i_face) .- center
        return SVector{2, Float64}(-x[2], x[1])
    end
    return AdvectionFlux{2}(rotation_velocity)
end

##############################

struct ShallowWater{T}
    g::T
end
ShallowWater(; g=9.81) = ShallowWater{typeof(g)}(g)

function (f::ShallowWater)(v::SVector{2}, n::Union{Number, Scalar, SVector{1}})
    h, hu = v[1], v[2]
    SVector(hu * n, hu^2/h + h^2 * f.g/2) * n
end

function (f::ShallowWater)(v::SVector{3}, n::SVector{2})
    h, ux, uy = v[1], v[2]/v[1], v[3]/v[1]
    un = ux*n[1] + uy*n[2]
    return SVector(h*un, h*ux*un + h^2*n[1]*f.g/2, h*uy*un + h^2*n[2]*f.g/2)
end

function jacobian(f::ShallowWater, v::AbstractVector, n::SVector{2})
    h, ux, uy = v[1], v[2]/v[1], v[3]/v[1]
    un = ux * n[1] + uy * n[2]
    g = f.g
    @SMatrix [0                 n[1]        n[2];
              -ux*un+g*h*n[1]   un+ux*n[1]  ux*n[2];
              -uy*un+g*h*n[2]   uy*n[1]     un+uy*n[2]]
end

function eigvals(f::ShallowWater, v, n::SVector{2})
    h, ux, uy = v[1], v[2]/v[1], v[3]/v[1]
    un = ux * n[1] + uy * n[2]
    g = f.g
    c = sqrt(g*h)
    return SVector(un - c, un, un + c)
end

function eigen(f::ShallowWater, v, n::SVector{2})
    h, ux, uy = v[1], v[2]/v[1], v[3]/v[1]
    un = ux * n[1] + uy * n[2]
    ut = -ux * n[2] + uy * n[1]
    g = f.g
    c = sqrt(g*h)
    vals = SVector(un - c, un, un + c)
    vects = 0.5 .* @SMatrix [1.0                    0.0   1.0;
                             n[1]*(un-c) + n[2]*ut  n[2]  n[1]*(c+un) + n[2]*ut;
                             n[1]*ut + n[2]*(un-c)  n[1]  n[1]*ut + n[2]*(c+un)]
    return vals, vects 
end

