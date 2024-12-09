### A Pluto.jl notebook ###
# v0.19.46          

using Markdown
using InteractiveUtils

# ╔═╡ e6c64c80-773b-11ef-2379-bf6609137e69
md"""
# Mathematik für Informatiker 2b

### Wintersemester 2024/2025
### Johannes Gutenberg-Universität Mainz
### Prof. Dr. Hendrik Ranocha
"""

# ╔═╡ 3bf1c95f-4a14-4a23-ba93-cf9bb26cb41e
let
	repo = "2024_MfI2b"
	url = "https://ranocha.de/" * repo * "/"
	
	notebooks = String[]
	for name in readdir(@__DIR__)
		full_name = joinpath(@__DIR__, name)
		if isfile(full_name) && endswith(name, ".jl") && 
								startswith(name, r"\d")
			push!(notebooks, name)
		end
	end

	text = """Hier finden Sie eine Liste der statischen Notebooks zur Vorlesung.
			  Um die Notebooks dynamisch verwenden zu können müssen Sie Julia
			  lokal installieren wie in der README.md des 
			  [Repositories](https://github.com/ranocha/$(repo)) beschrieben."""
	for name in notebooks
		file = read(name, String)
		pattern = r"md\"\"\"\n# (\d+\.\d+ [^\n]+)"
		m = match(pattern, file)
		title = m.captures[1]
		
		text = text * "\n- [`" * title * "`](" *
				url * name[begin:end-3] * ".html)"
	end

	Markdown.parse(text)
end

# ╔═╡ Cell order:
# ╟─e6c64c80-773b-11ef-2379-bf6609137e69
# ╟─3bf1c95f-4a14-4a23-ba93-cf9bb26cb41e
