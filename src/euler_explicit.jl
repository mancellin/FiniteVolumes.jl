import ProgressMeter

struct FixedCourant{T}
    fixed_courant::T
end

run!(flux, args...; kwargs...) = run!((flux,), args...; kwargs...)

function run!(fluxes::Tuple, mesh, w, t; nb_time_steps, time_step, verbose=true, callback=nothing, numerical_flux=Upwind(), boundary_flux=NeumannBC(), kwargs...)

	if verbose
		p = ProgressMeter.Progress(nb_time_steps, dt=0.1)
	end

    Δw = zeros(Δw_type(fluxes[1], mesh, w, numerical_flux, t), size(w))

    for i_time_step in 1:nb_time_steps

        if time_step isa FixedCourant
            dt = minimum(time_step.fixed_courant/courant(one(eltype(t)), f, mesh, w) for f in fluxes)
        else
            dt = time_step
        end

        for f in fluxes
            # using_conservative_variables!(m, w) do v
            #     v .-= dt * div(m, mesh, w; dt=dt, kwargs...)
            # end
            # v(w) = compute_v(m, w)
            # v⁻¹(v) = invert_v(m, v)
            # Δv = div(m, mesh, w; dt=dt, kwargs...)
            # w .= v⁻¹.(v.(w) .- dt.*Δv)
            Δw .= zeros(eltype(Δw), size(Δw))
            div!(Δw, f, mesh, w, numerical_flux, dt)
            div!(Δw, f, mesh, w, boundary_flux, dt)
            w .-= dt .* Δw
        end

		if verbose
			ProgressMeter.update!(p, i_time_step)
		end

        t += dt

        if callback != nothing
            callback(i_time_step, t, w)
        end
    end

    return t
end

function run(flux, mesh, w₀; t=0.0, kwargs...)
	w = deepcopy(w₀)
    t = run!(flux, mesh, w, t; kwargs...)
	return t, w
end

