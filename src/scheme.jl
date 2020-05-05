using ProgressMeter: Progress, update!


# NUMERICAL FLUX
#
abstract type NumericalFlux end

struct Upwind <: NumericalFlux end

function in_local_coordinates(f, grid, model, w, wsupp, i_face)
    i_cell_1, i_cell_2 = cells_next_to_inner_face(grid, i_face)
    w₁, wsupp₁ = rotate_state(w[i_cell_1], wsupp[i_cell_1], model, rotation_matrix(grid, i_face))
    w₂, wsupp₂ = rotate_state(w[i_cell_2], wsupp[i_cell_2], model, rotation_matrix(grid, i_face))
    local_model = rotate_model(model, rotation_matrix(grid, i_face))
    ϕ = f(local_model, w₁, wsupp₁, w₂, wsupp₂)
    return rotate_flux(ϕ, model, transpose(rotation_matrix(grid, i_face)))
end

function (::Upwind)(mesh, model::ScalarLinearAdvection, w, wsupp, i_face)
    in_local_coordinates(mesh, model, w, wsupp, i_face) do local_model, w₁, wsupp₁, w₂, wsupp₂
        λ = local_model.velocity[1]
        return normal_flux(local_model, λ > 0 ? w₁ : w₂, wsupp₁)
    end
end

function (::Upwind)(mesh, model, w, wsupp, i_face)  # l-upwind FVCF
    in_local_coordinates(mesh, model, w, wsupp, i_face) do local_model, w₁, wsupp₁, w₂, wsupp₂
        flux₁ = normal_flux(model, w₁, wsupp₁)
        flux₂ = normal_flux(model, w₂, wsupp₂)

        w_int, wsupp_int = compute_w_int(model, w₁, wsupp₁, w₂, wsupp₂)
        λ = eigenvalues(model, w_int, wsupp_int)

        L₁ = left_eigenvectors(model, w₁, wsupp₁)
        L₂ = left_eigenvectors(model, w₂, wsupp₂)

        L_upwind = ifelse.(λ .> 0.0, L₁, L₂)
        L_flux_upwind = ifelse.(λ .> 0.0, L₁ * flux₁, L₂ * flux₂)

        ϕ = L_upwind \ L_flux_upwind
        return ϕ
    end
end


# BOUNDARY FLUX
function in_local_coordinates_at_boundary(f, grid, model, w, wsupp, i_face)
    i_cell_1 = cell_next_to_boundary_face(grid, i_face)
    w₁, wsupp₁ = rotate_state(w[i_cell_1], wsupp[i_cell_1], model, rotation_matrix(grid, i_face))
    local_model = rotate_model(model, rotation_matrix(grid, i_face))
    ϕ = f(local_model, w₁, wsupp₁)
    return rotate_flux(ϕ, model, transpose(rotation_matrix(grid, i_face)))
end

function neumann_bc(args...)
    in_local_coordinates_at_boundary(args...) do local_model, w₁, wsupp₁
        normal_flux(local_model, w₁, wsupp₁)
    end
end


# BALANCE
function div(model, mesh, w, wsupp; numerical_flux=Upwind(), boundary_flux=neumann_bc)
    Δv = zeros(eltype(w), nb_cells(mesh))

    @inbounds for i_face in inner_faces(mesh)
        ϕ = numerical_flux(mesh, model, w, wsupp, i_face)
        i_cell_1, i_cell_2 = cells_next_to_inner_face(mesh, i_face)
        Δv[i_cell_1] += ϕ * face_area(mesh, i_face) / cell_volume(mesh, i_cell_1)
        Δv[i_cell_2] -= ϕ * face_area(mesh, i_face) / cell_volume(mesh, i_cell_2)
    end

    @inbounds for i_face in boundary_faces(mesh)
        ϕ = boundary_flux(mesh, model, w, wsupp, i_face)
        i_cell = cell_next_to_boundary_face(mesh, i_face)
        Δv[i_cell] += ϕ * face_area(mesh, i_face) / cell_volume(mesh, i_cell)
    end
    return Δv
end

function div(model, mesh; kwargs...)
    function divF(w, wsupp=nothing)
        if wsupp == nothing
            wsupp = [compute_wsupp(model, wi) for wi in w]
        end
        return div(model, mesh, w, wsupp; kwargs...)
    end
end

function using_conservative_variables!(up!, model, w, wsupp)
    v = zeros(eltype(w), length(w))
    @inbounds for i_cell in 1:length(w)
        v[i_cell] = compute_v(model, w[i_cell], wsupp[i_cell])
    end
    up!(v)
    @inbounds for i_cell in 1:length(w)
        w[i_cell] = invert_v(model, v[i_cell])
        wsupp[i_cell] = compute_wsupp(model, w[i_cell])
    end
end


# COURANT

function courant(Δt, mesh, model::AbstractModel, w, wsupp)
    courant = 0.0
    for i_face in inner_faces(mesh)
        i_cell_1, i_cell_2 = cells_next_to_inner_face(mesh, i_face)
        w₁, wsupp₁ = rotate_state(w[i_cell_1], wsupp[i_cell_1], model, rotation_matrix(mesh, i_face))
        w₂, wsupp₂ = rotate_state(w[i_cell_2], wsupp[i_cell_2], model, rotation_matrix(mesh, i_face))
        local_model = rotate_model(model, rotation_matrix(mesh, i_face))
        w_int, wsupp_int = compute_w_int(local_model, w₁, wsupp₁, w₂, wsupp₂)
        λ = eigenvalues(local_model, w_int, wsupp_int)
        maxλ = maximum(abs.(λ))
        courant = max(courant, maxλ * Δt * face_area(mesh, i_face) / max(cell_volume(mesh, i_cell_1), cell_volume(mesh, i_cell_2)))
    end
    return courant
end

function courant(Δt, mesh::RegularMesh1D, model::ScalarLinearAdvection, w, wsupp)
    return abs(model.velocity[1]) * Δt / dx(mesh)
end

function courant(Δt, mesh::PeriodicRegularMesh2D, model::ScalarLinearAdvection, w, wsupp)
    vx, vy = model.velocity
    return max(abs(vx) * Δt / dx(mesh), abs(vy) * Δt / dy(mesh))
end


# RUN

function run!(models, mesh, w, t; nb_time_steps, dt=nothing, cfl=nothing, verbose=true, kwargs...)
    wsupp = map(wi -> compute_wsupp(models[1], wi), w)

	if verbose
		p = Progress(nb_time_steps, dt=0.1)
	end
    for i_time_step in 1:nb_time_steps

        if isnothing(dt)
            if !isnothing(cfl)
                dt = cfl/courant(1.0, mesh, models[1], w, wsupp)
            else
                error("No time step nor Courant number has been provided :(")
            end
        end

        for m in models
            using_conservative_variables!(m, w, wsupp) do v
                v .-= dt * div(m, mesh, w, wsupp; kwargs...)
            end
        end
		if verbose
			update!(p, i_time_step)
		end

        t += dt
    end
	if verbose
		update!(p, nb_time_steps)
	end
	
    return t
end

function run(model, grid, w₀; kwargs...)
	w = deepcopy(w₀)
    t = 0.0
    if model isa AbstractModel
        model = [model]
    end
    run!(model, grid, w, t; kwargs...)
	return t, w
end

