using ProgressMeter: Progress, update!


# NUMERICAL FLUX
#
abstract type NumericalFlux end

struct Upwind <: NumericalFlux end

function in_local_coordinates(f, model, mesh, w, i_face)
    i_cell_1, i_cell_2 = cells_next_to_inner_face(mesh, i_face)
    w₁ = rotate_state(w[i_cell_1], model, rotation_matrix(mesh, i_face))
    w₂ = rotate_state(w[i_cell_2], model, rotation_matrix(mesh, i_face))
    local_model = rotate_model(model, rotation_matrix(mesh, i_face))
    ϕ = f(local_model, w₁, w₂)
    return rotate_flux(ϕ, model, transpose(rotation_matrix(mesh, i_face)))
end

function (::Upwind)(model::ScalarLinearAdvection, mesh, w, i_face; kwargs...)
    in_local_coordinates(model, mesh, w, i_face) do local_model, w₁, w₂
        λ = local_model.velocity[1]
        return normal_flux(local_model, λ > 0 ? w₁ : w₂)
    end
end

function (::Upwind)(model, mesh, w, i_face; kwargs...)  # l-upwind FVCF
    in_local_coordinates(model, mesh, w, i_face) do local_model, w₁, w₂
        flux₁ = normal_flux(model, w₁)
        flux₂ = normal_flux(model, w₂)

        w_int = compute_w_int(model, w₁, w₂)
        λ = eigenvalues(model, w_int)

        L₁ = left_eigenvectors(model, w₁)
        L₂ = left_eigenvectors(model, w₂)

        L_upwind = ifelse.(λ .> 0.0, L₁, L₂)
        L_flux_upwind = ifelse.(λ .> 0.0, L₁ * flux₁, L₂ * flux₂)

        ϕ = L_upwind \ L_flux_upwind
        return ϕ
    end
end


# BOUNDARY FLUX
function in_local_coordinates_at_boundary(f, model, mesh, w, i_face)
    i_cell_1 = cell_next_to_boundary_face(mesh, i_face)
    w₁ = rotate_state(w[i_cell_1], model, rotation_matrix(mesh, i_face))
    local_model = rotate_model(model, rotation_matrix(mesh, i_face))
    ϕ = f(local_model, w₁)
    return rotate_flux(ϕ, model, transpose(rotation_matrix(mesh, i_face)))
end

function neumann_bc(args...; kwargs...)
    in_local_coordinates_at_boundary(args...) do local_model, w₁
        normal_flux(local_model, w₁)
    end
end


# BALANCE
function div(model, mesh, w; numerical_flux=Upwind(), boundary_flux=neumann_bc, dt=nothing)
    Δv = zeros(consvartype(model, w), nb_cells(mesh))

    @inbounds for i_face in inner_faces(mesh)
        ϕ = numerical_flux(model, mesh, w, i_face; dt=dt)
        i_cell_1, i_cell_2 = cells_next_to_inner_face(mesh, i_face)
        Δv[i_cell_1] += ϕ * face_area(mesh, i_face) / cell_volume(mesh, i_cell_1)
        Δv[i_cell_2] -= ϕ * face_area(mesh, i_face) / cell_volume(mesh, i_cell_2)
    end

    @inbounds for i_face in boundary_faces(mesh)
        ϕ = boundary_flux(model, mesh, w, i_face; dt=dt)
        i_cell = cell_next_to_boundary_face(mesh, i_face)
        Δv[i_cell] += ϕ * face_area(mesh, i_face) / cell_volume(mesh, i_cell)
    end
    return Δv
end

function div(model, mesh; kwargs...)
    function divF(w)
        return div(model, mesh, w; kwargs...)
    end
end

function using_conservative_variables!(up!, model, w)
    v = zeros(consvartype(model, w), length(w))
    @inbounds for i_cell in 1:length(w)
        v[i_cell] = compute_v(model, w[i_cell])
    end
    up!(v)
    @inbounds for i_cell in 1:length(w)
        w[i_cell] = invert_v(model, v[i_cell])
    end
end


# COURANT

function courant(Δt, model::AbstractModel, mesh, w)
    courant = 0.0
    for i_face in inner_faces(mesh)
        i_cell_1, i_cell_2 = cells_next_to_inner_face(mesh, i_face)
        w₁ = rotate_state(w[i_cell_1], model, rotation_matrix(mesh, i_face))
        w₂ = rotate_state(w[i_cell_2], model, rotation_matrix(mesh, i_face))
        local_model = rotate_model(model, rotation_matrix(mesh, i_face))
        w_int = compute_w_int(local_model, w₁, w₂)
        λ = eigenvalues(local_model, w_int)
        maxλ = maximum(abs.(λ))
        courant = max(courant, maxλ * Δt * face_area(mesh, i_face) / max(cell_volume(mesh, i_cell_1), cell_volume(mesh, i_cell_2)))
    end
    return courant
end

function courant(Δt, model::ScalarLinearAdvection, mesh::RegularMesh1D, w)
    return abs(model.velocity[1]) * Δt / dx(mesh)
end

function courant(Δt, model::ScalarLinearAdvection, mesh::PeriodicRegularMesh2D, w)
    vx, vy = model.velocity
    return max(abs(vx) * Δt / dx(mesh), abs(vy) * Δt / dy(mesh))
end


# RUN

run!(model::AbstractModel, args...; kwargs...) = run!([model], args...; kwargs...)

function run!(models, mesh, w, t; nb_time_steps, dt=nothing, cfl=nothing, verbose=true, kwargs...)

	if verbose
		p = Progress(nb_time_steps, dt=0.1)
	end

    for i_time_step in 1:nb_time_steps

        if isnothing(dt)
            if !isnothing(cfl)
                dt = minimum(cfl/courant(1.0, m, mesh, w) for m in models)
            else
                error("No time step nor Courant number has been provided :(")
            end
        end

        for m in models
            using_conservative_variables!(m, w) do v
                v .-= dt * div(m, mesh, w; dt=dt, kwargs...)
            end
        end

		if verbose
			update!(p, i_time_step)
		end

        t += dt
    end

    return t
end

function run(model, mesh, w₀; kwargs...)
	w = deepcopy(w₀)
    t = 0.0
    run!(model, mesh, w, t; kwargs...)
	return t, w
end

