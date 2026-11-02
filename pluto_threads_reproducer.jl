### A Pluto.jl notebook ###
# v1.0.4

using Markdown
using InteractiveUtils

# ╔═╡ 47cd4600-5f13-4e50-b688-a7162a2a7a1b
md"""
After Pluto evaluates the notebook, rerun the initialization cell and then
rerun the `@fill` cell. The latter errors with `NotInitializedError`.
"""

# ╔═╡ a583c4ee-891b-4354-ae5b-134144e1e39a
using ParallelStencil

# ╔═╡ 5e8dd40b-50b2-4b34-8c51-85c6a85c073c
@init_parallel_stencil(Threads, Float64, 3)

# ╔═╡ 72d0257e-a276-4a0a-9cf4-6a5e2ef9ac66
@fill(2, 100, 100)

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
ParallelStencil = "94395366-693c-11ea-3b26-d9b7aac5d958"
"""

# ╔═╡ Cell order:
# ╟─47cd4600-5f13-4e50-b688-a7162a2a7a1b
# ╠═a583c4ee-891b-4354-ae5b-134144e1e39a
# ╠═5e8dd40b-50b2-4b34-8c51-85c6a85c073c
# ╠═72d0257e-a276-4a0a-9cf4-6a5e2ef9ac66
# ╟─00000000-0000-0000-0000-000000000001
