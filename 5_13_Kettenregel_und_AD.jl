### A Pluto.jl notebook ###
# v0.19.46

using Markdown
using InteractiveUtils

# ╔═╡ 437a2d3f-7f19-4813-af1b-babd8b883310
using LinearAlgebra

# ╔═╡ f05a5972-58b1-4788-a0a8-24966d6714da
begin
	using PlutoUI
	using PlutoUI: Slider
end

# ╔═╡ e21f7893-67e3-42ba-82e8-1297502cc1ea
begin
	using CairoMakie
	set_theme!(theme_latexfonts();
			   fontsize = 16,
			   Lines = (linewidth = 2,),
			   markersize = 16)
end

# ╔═╡ b0d18f0a-7ae7-4c9e-9e29-2f190aaae1c2
using LaTeXStrings

# ╔═╡ 02ed8724-fbe6-4cdd-bab6-9f7ccfed8380
using BenchmarkTools

# ╔═╡ 72f1a9ed-3047-4ab9-b038-10c76984c540
using Enzyme

# ╔═╡ fe0a3bf7-3105-437a-888b-94424ff94608
using SymPyPythonCall

# ╔═╡ e6c64c80-773b-11ef-2379-bf6609137e69
md"""
# 5.13 Kettenregel im Mehrdimensionalen und AD

Algorithmisches/automatisches Differenzieren (AD) ist essentieller
Bestandteil vieler Anwendungen im maschinellen Lernen (deep learning).
Hier erklären wir kurz die grundlegenden Ideen, die beiden
Grund-Varianten *forward-mode* und *reverse-mode* AD und den Zusammenhang
mit der Kettenregel im Mehrdimensionalen.
"""

# ╔═╡ cd57529b-64a3-43cd-8dbf-445583c8edcc
md"""
## Forward-mode AD für Skalare

Es gibt das Sprichwort

> Ableiten ist ein Handwerk, Integrieren eine Kunst

Wir betrachten hier zum Glück nur automatisches/algorithmisches Differenzieren.
Also müssen wir nur die grundlegenden Rechenregel der Differentialrechnung
implementieren - die Produkt- und Kettenregel etc. Vorher betrachten wir jedoch
ein Beispiel.
"""

# ╔═╡ a7d5feb5-c3c5-4d55-b721-f9a838a22e78
f(x) = log(x^2 + exp(sin(x)))

# ╔═╡ ee241f76-b1d9-4c00-9b53-020a1ba012dd
md"""
Wir können die Ableitung mithilfe der Kettenregel berechnen.
"""

# ╔═╡ 127944b9-b3eb-4df8-acc3-36d9958218ca
md"""
Wir können die Funktion auch als einen Graphen (*computational graph*) betrachten,
indem wir die Ausführung in einzelne Schritte zerlegen.
"""

# ╔═╡ a89b42c6-913e-4970-bd9e-8f163a4b96d8
md"""
Um die Ableitung zu berechnen können wir jetzt nacheinander die Kettenregel anwenden.
"""

# ╔═╡ 9777384f-dc51-4d6b-b55f-87b5fd2c5be7
md"""
Das würden wir jetzt gerne automatisieren! Dazu verwenden wir sogenannte
[duale Zahlen](https://en.wikipedia.org/wiki/Dual_number#Differentiation).
Diese haben einen Wert (`value`) und eine Ableitung (*derivative*, `deriv` --
der ε von oben). Formal schreiben wir eine duale Zahl als

$$x + \varepsilon y, \qquad x, y \in \mathbb{R},$$

ähnlich wie eine komplexe Zahl

$$z = x + \mathrm{i} y, \qquad x, y \in \mathbb{R}.$$

Allerdings erfüllt das neue Basis-Element $\varepsilon$

$$\varepsilon^2 = 0$$

statt $\mathrm{i}^2 = -1$. Daher haben duale Zahlen die Struktur einer
*Algebra* statt eines Körpers wie die komplexen Zahlen $\mathbb{C}$.

In unserer Anwendung wird der $\varepsilon$ die Ableitungen enthalten.
In der Tat ist mit $\varepsilon^2 = 0$

$$(a + \varepsilon b) (c + \varepsilon d) = a c + \varepsilon (a d + b c),$$

was nichts Anderes als die Produktregel ist. Das können wir wie folgt
implementieren.
"""

# ╔═╡ a7195ecb-5e70-4abc-880c-063533944e91
begin
	struct MyDual{T <: Real} <: Number
		value::T
		deriv::T
	end
	MyDual(x::Real, y::Real) = MyDual(promote(x, y)...)
end

# ╔═╡ be74e9c0-cd17-47c5-9656-8109db96897a
md"Jetzt können wir duale Zahlen erstellen."

# ╔═╡ dde0b734-1c54-4a54-b112-b256506fa0a6
MyDual(5, 2.0)

# ╔═╡ 927cbfbd-daa8-47ff-b100-6c0d81b5ea82
md"Als nächstes implementieren wir das Interface für Zahlen in Julia."

# ╔═╡ 1437a1cc-dddd-4d4f-be18-81ab0f92950b
Base.:+(x::MyDual, y::MyDual) = MyDual(x.value + y.value, x.deriv + y.deriv)

# ╔═╡ af3ea3f6-0b4e-4cf1-9aa7-d619fa4e846c
MyDual(1, 2) + MyDual(2.0, 3)

# ╔═╡ 7c60e778-b116-4504-a60a-72c52d91fd9d
Base.:-(x::MyDual, y::MyDual) = MyDual(x.value - y.value, x.deriv - y.deriv)

# ╔═╡ 1f97ab58-580d-4069-992f-f611312bc71f
MyDual(1, 2) - MyDual(2.0, 3)

# ╔═╡ 84a6bc63-8e55-45ed-a6b8-150a0bf3f22a
function Base.:*(x::MyDual, y::MyDual)
	MyDual(x.value * y.value, x.value * y.deriv + x.deriv * y.value)
end

# ╔═╡ a7cb831e-49bf-4a16-bc5e-118d31a069bd
MyDual(1, 2) * MyDual(2.0, 3)

# ╔═╡ 48b60e62-3879-4693-9540-c35a2a5d09c7
function Base.:/(x::MyDual, y::MyDual)
	MyDual(x.value / y.value, (x.deriv * y.value - x.value * y.deriv) / y.value^2)
end

# ╔═╡ d2014f18-ca17-4a6c-8bcc-efeecc7f29d2
MyDual(1, 2) / MyDual(2.0, 3)

# ╔═╡ fd1c6e36-1309-4c85-b0be-6e7a215a4f54
md"Als nächstes müssen wir noch implementieren, wie übliche und duale Zahlen konvertiert werden sollen."

# ╔═╡ 2efd0312-54b4-4c64-a505-11be21ab5c18
Base.convert(::Type{MyDual{T}}, x::Real) where {T <: Real} = MyDual(x, zero(T))

# ╔═╡ a43b5324-fc81-4247-b6b1-4e8f5c367cbe
Base.promote_rule(::Type{MyDual{T}}, ::Type{<:Real}) where {T <: Real} = MyDual{T}

# ╔═╡ 79544ce2-6bf0-4cd6-a4f2-ab2122d9bc49
MyDual(1, 2) + 3.0

# ╔═╡ fbc63b55-5d04-4b70-8dd4-f5b3eff6d99f
md"Jetzt implementieren wir die Ableitungen spezieller Funktionen, die wir im Beispiel brauchen."

# ╔═╡ d4ad273b-b548-43c3-9e69-807b47bace27
function Base.sin(x::MyDual)
	si, co = sincos(x.value)
	return MyDual(si, co * x.deriv)
end

# ╔═╡ bb22987a-c3b8-4242-8a3e-91f773841b40
sin(MyDual(π, 1))

# ╔═╡ 7421edea-ec1c-47cb-9b19-77d9203d6857
function Base.cos(x::MyDual)
	si, co = sincos(x.value)
	return MyDual(co, -si * x.deriv)
end

# ╔═╡ 7e76a3bf-a750-4f8c-a28a-d44505d3526f
cos(MyDual(π, 1))

# ╔═╡ 26659be3-07ea-4844-9f91-0490faa5a082
Base.log(x::MyDual) = MyDual(log(x.value), x.deriv / x.value)

# ╔═╡ 5695da28-2297-418b-92db-4c2271edbefd
log(MyDual(1.0, 1))

# ╔═╡ e9e02a31-9b3c-412c-84a9-8490a664715b
function Base.exp(x::MyDual)
	e = exp(x.value)
	return MyDual(e, e * x.deriv)
end

# ╔═╡ b8b74818-30f8-4219-b293-657025589a44
f′(x) = 1 / (x^2 + exp(sin(x))) * (2 * x + exp(sin(x)) * cos(x))

# ╔═╡ 48768b89-edf4-4c69-99ad-d304d2700bb4
function f_graph(x)
	c1 = x^2
	c2 = sin(x)
	c3 = exp(c2)
	c4 = c1 + c3
	c5 = log(c4)
	return c5
end

# ╔═╡ 2558317b-14e6-4871-9bd3-5b26be391019
function f_graph_derivative(x)
	c1 = x^2
	c1_ε = 2 * x
	
	c2 = sin(x)
	c2_ε = cos(x)
	
	c3 = exp(c2)
	c3_ε = exp(c2) * c2_ε
	
	c4 = c1 + c3
	c4_ε = c1_ε + c3_ε
	
	c5 = log(c4)
	c5_ε = c4_ε / c4
	return c5, c5_ε
end

# ╔═╡ c660ccf1-5bff-44f1-92b5-10465011748e
f_graph_derivative(1.0)

# ╔═╡ 4e4ad771-736b-4789-8ed0-43d2e1799b40
exp(MyDual(1.0, 1))

# ╔═╡ bc2c6be4-10a9-4cb9-be8e-aa64745c32ba
md"Damit können wir die Funktion `f` aus dem Beispiel differenzieren!"

# ╔═╡ 625d13ec-ce64-4330-9842-2e1596827079
md"""
Das funktioniert, weil der Compiler im Wesentlichen die Transformation 
`f` $\to$ `f_graph_derivative` für uns übernimmt. Wir können dies auch
Wesentlichen, indem wir die verschiedenen Schritte in der Compiler-Pipeline
von Julia betrachten.
"""

# ╔═╡ 1664e082-9ae2-4deb-9268-c03861aa49b2
@code_typed f_graph_derivative(1.0)

# ╔═╡ 0f210d38-f8be-470c-ab1e-1baeae951482
md"Weil der Compiler alle Schritte sieht, kann effizienter Code generiert werden."

# ╔═╡ f3a85308-8448-4458-b224-e2bd1ee48077
@benchmark f_graph_derivative($(Ref(1.0))[])

# ╔═╡ 74ebd661-8242-431d-8163-d1bbfcee52da
md"Damit können wir jetzt Ableitungen von Funktionen einer Variablen berechnen."

# ╔═╡ 63c9cd60-64b6-4755-b3b2-96174882f618
derivative(f, x::Real) = f(MyDual(x, one(x))).deriv

# ╔═╡ acf3dc94-3a21-4bce-862e-93ccb4d44b3a
md"Wir können auch die Ableitung als Funktion erhalten."

# ╔═╡ 252caf68-b944-4f30-98ae-dc40482ef1a6
derivative(f) = x -> derivative(f, x)

# ╔═╡ af152a4f-46f9-4bc1-8b87-ff7b3fd908d5
derivative(x -> 3 * x^2 + 4 * x + 5, 2)

# ╔═╡ e128836c-1fac-46ba-9db5-3803fa090759
derivative(3) do x
	sin(x) * log(x)
end

# ╔═╡ 33aae6b2-ddb7-4a09-b672-0aa426977971
md"""
## Heron-Verfahren

Das Heron-Verfahren (Newton-Verfahren) zur Berechnung der Quadratwurzel einer
Zahl ist die Iteration

$$t \leftarrow \frac{t}{2} + \frac{x}{2 t},$$

deren Konvergenz wir in Kapitel 1 betrachtet haben. Hier betrachten wir der Einfachheit halber 10 Iterationen.
"""

# ╔═╡ 88dbd750-7ca4-42ee-8790-e198131e9a0f
function heron(x; N = 10) 
    t = (1 + x) / 2 # initial guess t = 1
    for i in 2:N
        t = (t + x / t) / 2
    end    
    return t
end 

# ╔═╡ a748d342-8b42-4394-bc8f-905387b4f16a
heron(100.0)

# ╔═╡ a2238c71-0467-4273-be7d-f4188cf4efb3
heron(MyDual(100.0, 1.0))

# ╔═╡ bc49478e-523a-49a9-813b-36924c9a9814
let 
	@syms x
	diff(heron(x, N = 4), x) |> simplify
end

# ╔═╡ fb02594f-e6fb-40c0-908c-14f2db04bfe6
let 
	@syms x
	diff(heron(x, N = 5), x) |> simplify
end

# ╔═╡ ddf7d1b0-e4fb-4e0b-ab94-455b5751323c
let 
	@syms x
	diff(heron(x, N = 6), x)
end

# ╔═╡ f9c7430c-e383-4611-974d-472735df3bbc
let 
	@syms x
	diff(heron(x, N = 10), x)(100.0)
end

# ╔═╡ 484cad0d-f0b7-4349-ba5a-8bbc89a75c16
md"""
## Mehrere Variablen

Bei Funktionen von mehreren Variablen können wir die partiellen Ableitungen
(und damit Gradienten, Jacobi-Matrizen) berechnen, indem wir den skalaren
Code entsprechend anwenden.
"""

# ╔═╡ dd01a246-d8f0-45bb-bd65-ceba082b8996
g1(x) = x[1]^2 * x[2]

# ╔═╡ 56cf0311-7e97-432a-8671-2e44fb89e272
function gradient_scalar(g, x0)
	grad = zeros(typeof(g(x0)), length(x0))
	x = similar(x0, MyDual{eltype(x0)})
	copyto!(x, x0)
	for i in eachindex(x)
		x[i] = MyDual(x0[i], 1)
		grad[i] = g(x).deriv
		x[i] = x0[i]
	end
	return grad
end

# ╔═╡ 28df9335-f5e3-4afe-8d8a-3ccde842682c
gradient_scalar(g1, [1.0, 2.0])

# ╔═╡ b179851e-b248-4d14-ae88-451c59380125
md"""
Wenn wir das auf allgemeine Funktionen

$$f\colon \mathbb{R}^n \to \mathbb{R}^m$$

anwenden, füllen wir die Jacobi-Matrix

$$f'(x) = J_f(x) = 
    \begin{pmatrix}
      \partial_1 f_1(x) & \dots & \partial_n f_1(x) \\
      \vdots & \ddots & \vdots \\
      \partial_1 f_m(x) & \dots & \partial_n f_m(x)
    \end{pmatrix}$$

somit Spalte für Spalte.

Manchmal sind wir aber nicht an der vollen
Jacobi-Matrix interessiert, sondern nur an Richtungsableitungen, d.h.,
an Jacobi-Matrix-Vektor-Produkten. Die können wir effizient durch einen
einzelnen Aufruf von forward-mode AD berechen, indem wir die Koeffizienten
der Ableitungen entsprechend wählen.

$$\begin{aligned}
	f'(x) v &= J_f(x) v = 
    \begin{pmatrix}
      \partial_1 f_1(x) & \dots & \partial_n f_1(x) \\
      \vdots & \ddots & \vdots \\
      \partial_1 f_m(x) & \dots & \partial_n f_m(x)
    \end{pmatrix}
	\begin{pmatrix}
		v_1 \\
		\vdots \\
		v_n
	\end{pmatrix}
	\\
	&=
	\begin{pmatrix}
      v_1 \partial_1 f_1(x) + \dots + v_n \partial_n f_1(x) \\
      \vdots \\
      v_1 \partial_1 f_m(x) + \dots + v_n \partial_n f_m(x)
    \end{pmatrix}.
\end{aligned}$$

Im Rahmen von AD ist dies als *Jacobian vector product* bekannt und ist die
zentrale Operation von forward-mode AD.
"""

# ╔═╡ ee92845a-634b-4b93-9032-6cde78c7dc8b
md"""
## Forward- vs. reverse-mode AD

Oben haben wir die grundlegende Idee von forward-mode AD sowie eine einfache
Implementierung vorgestellt. Dieses Vorgehen heißt forward-mode AD, weil die
Informationen der ABleitungen in der gleichen Reihenfolge wie die übliche
Berechnung des Funktionswerts ausgeführt wird.

Als nächstes betrachten wir ein typisches Optimierungs-Problem, etwa

$$\min_x \| A x - b \|_2^2,$$

wobeio $A$ und $b$ gegeben sind. Um ein Optimierungsverfahren wie beispielsweise
den (stochastischen) Gradientenabstieg zu verwenden, müssen wir die Ableitungen
bzgl. $x$ ausrechnen, d.h.,

$$\nabla_x \| A x - b \|_2^2 = 2 (A x - b)^T A.$$

Dies testen wir in einem Beispiel.
"""

# ╔═╡ b1954b3c-3dbc-411a-a6ac-29fa39615b3d
let
	A = [1.0 2.0; 3.0 4.0]
	b = [5.0, 6.0]
	f = x -> begin
		y = A * x - b
		return y[1]^2 + y[2]^2
	end

	x = randn(2)
	result_ad = gradient_scalar(f, x)
	result = 2 * (A * x - b)' * A
	
	abs2(result_ad[1] - result[1]) + abs2(result_ad[2] - result[2])
end

# ╔═╡ 9481863b-5fa7-4f2d-af03-dd3ca7aa433c
md"""
Jetzt betrachten wir eine nicht-quadratische Matrix $A$ und nehmen
$b = 0$ als Vereinfachung. Dann haben wir

$$\nabla_x \| A x \|_2^2 = 2 (A x)^T A = 2 x^T A^T A.$$

Wenn man den Gradienten per Hand ausrechnen möchte, hat man zwei Möglichkeiten:

- erst $A^T A$ berechnen und dann $x^T (A^T A)$
- erst $x^T A^T$ berechnen und dann $(x^T A^T) A$

Welche Möglichkeit soll man wählen?
"""

# ╔═╡ f3d17d53-2fa9-4cad-a632-2cecac4e875a
order_1(A, x) = x' * (A' * A)

# ╔═╡ 14d32872-7a1e-44c4-a60f-5a9a809e7298
order_2(A, x) = (x' * A') * A

# ╔═╡ d4f127d2-7f4a-47e0-bd53-ada2066e2585
md"### $4 \times 4$ Matrix"

# ╔═╡ cadc4aab-f447-4662-b92d-6f8bf90340c1
A1 = randn(4, 4)

# ╔═╡ e508256a-4b90-43c2-bb32-3cbb52974ab6
x1 = randn(size(A1, 2))

# ╔═╡ b183eea7-7939-4f7d-875c-8d7ac8dd15f5
@benchmark order_1($A1, $x1)

# ╔═╡ 61cf5a89-6f00-4599-81f1-5c9ae90f1d37
@benchmark order_2($A1, $x1)

# ╔═╡ 4c4a9e9f-73b6-44e7-accf-3300bbd353d9
md"### $2 \times 8$ Matrix"

# ╔═╡ 51f58054-50a7-4d3d-89ec-2f867007cd54
A3 = randn(2, 8)

# ╔═╡ c3f4cfc5-c90b-404e-a18f-13c59f0b8600
x3 = randn(size(A3, 2))

# ╔═╡ e9c6b5d2-5d4a-489b-883c-9ba58b2539f2
@benchmark order_1($A3, $x3)

# ╔═╡ 62bf4cc7-4804-4b30-b0bf-863e4b527cf2
@benchmark order_2($A3, $x3)

# ╔═╡ b0de5259-7237-442e-a058-2f4da305ce50
md"""
### Allgemeine Einführung

Der Unterschied der beiden Reihenfolgen ist im Wesentlichen die grundlegende
Idee von forward- und reverse-mode AD. Für ein grobes gedankliches Modell
betrachten wir die Kettenregel angewandt auf die Funktion

$$x \mapsto f\Bigl( g\bigl( h(x) \bigr) \Bigr),$$

d.h.,

$$\biggl( f\Bigl( g\bigl( h(x) ) \bigr) \Bigr)\biggr)' = f'\Bigl( g\bigl( h(x) \bigr) \Bigr) \cdot g'\bigl( h(x) \bigr) \cdot h'(x).$$

Forward-mode AD berechnet die Ableitung von rechts nach links -- genau wie die
übliche Berechnung des Wertes von $(f \circ g \circ h)(x)$. Im Gegensatz dazu
berechnet reverse-mode AD die Ableitung von links nach rechts, d.h., in der
umgekehrten Reihenfolge. Um dies machen zu können, müssen Zwischenwerte 
$h(x)$ und $g\bigl( h(x) \bigr)$ gespeichert werden.
"""

# ╔═╡ 8fcf4771-5a72-4ef7-b0ae-4d4805160c6d
function forward(f, f′, g, g′, h, h′, x)
	h_x = h(x)
	h′_x = h′(x)

	gh_x = g(h_x)
	gh′_x = g′(h_x) * h′_x

	fgh_x = f(gh_x)
	fgh′_x = f′(gh_x) * gh′_x

	return fgh_x, fgh′_x
end

# ╔═╡ 2f53b546-b02f-42cd-ab03-ec92e702868b
function reverse(f, f′, g, g′, h, h′, x)
	h_x = h(x)
	gh_x = g(h_x)
	fgh_x = f(gh_x)

	f′_ghx = f′(gh_x)
	fg′_hx = f′_ghx * g′(h_x)
	fgh′_x = fg′_hx * h′(x)

	return fgh_x, fgh′_x
end

# ╔═╡ 7a1846df-7c5b-43af-86d9-d067d14d2752
const A = randn(10, 10^2)

# ╔═╡ 0967c13e-13ef-4a35-8045-36712f2feeeb
const b = randn(size(A, 1))

# ╔═╡ 9c20e8b7-7d90-4c68-b28c-f79feb0370ef
h(x::AbstractVector) = A * x

# ╔═╡ 9d838386-e87c-438d-90a6-56a443860249
h′(x::AbstractVector) = A

# ╔═╡ c651abd9-af17-4a75-821c-24eac406daf9
g(Ax::AbstractVector) = Ax - b

# ╔═╡ 505bad11-94e6-4e08-99c1-1537df969aee
g′(Ax::AbstractVector) = I

# ╔═╡ 7392a184-40df-4735-8560-d82ff2e4bf70
f(Ax_b::AbstractVector) = sum(abs2, Ax_b)

# ╔═╡ 43ef27fd-a67d-4cf8-a8e6-3d11880b5eac
f(1.0) ≈ f_graph(1.0)

# ╔═╡ ff0adc67-3dd9-4402-b033-63a843dc8790
let x = 1.0, h = sqrt(eps())
	(f(x + h) - f(x)) / h
end

# ╔═╡ c9b56c8c-38af-4069-b2fe-4c4cf0753f6c
let
	f_dual = f(MyDual(1.0, 1.0))
	(f_dual.value, f_dual.deriv) .- f_graph_derivative(1.0)
end

# ╔═╡ 26238234-602d-4117-ab56-120dbecb1130
@code_typed f(MyDual(1.0, 1.0))

# ╔═╡ 17421bbd-9e8d-475e-be20-b46da8cc6449
@benchmark f(MyDual($(Ref(1.0))[], 1.0))

# ╔═╡ 2f5bc87b-7675-42d0-bd86-6503403a3404
derivative(f, 1.0)

# ╔═╡ 05c9713c-917e-4b9b-a53b-38cde05be415
f′(Ax_b::AbstractVector) = 2 * Ax_b'

# ╔═╡ 56f7e83d-6d8f-413c-a2bb-fd89bebd28b5
(f(1.0), f′(1.0))

# ╔═╡ d45ce0d3-d23d-41ce-af4d-4946c3a37253
let
	f_dual = f(MyDual(1.0, 1.0))
	(f_dual.value, f_dual.deriv) .- (f(1.0), f′(1.0))
end

# ╔═╡ c6230689-5f2a-4f57-ad03-8d87423fa5a2
let df = derivative(f)
	x = range(0.1, 10.0, length = 10)
	df.(x) - f′.(x)
end

# ╔═╡ e569b49b-365e-4cee-88bf-33d583fd724c
let
	x = randn(size(A, 2))
	fwd = forward(f, f′, g, g′, h, h′, x)
	rev = reverse(f, f′, g, g′, h, h′, x)
	fwd[1] - rev[1], norm(fwd[2] - rev[2])
end

# ╔═╡ 46d1fab5-4cab-4b3f-a4bc-b81878619503
let
	x = randn(size(A, 2))
	@benchmark forward($f, $f′, $g, $g′, $h, $h′, $x)
end

# ╔═╡ 54d8fb0e-c42e-44c8-9357-28f8d30ac367
let
	x = randn(size(A, 2))
	@benchmark reverse($f, $f′, $g, $g′, $h, $h′, $x)
end

# ╔═╡ a74e1db9-0f6f-48d9-8061-ea5b8d3b2f08
md"""
Dies ist jedoch nur ein einfaches Modell -- und ein bekanntes Sprichwort ist

> Im Prinzip sind alle Modelle falsch, aber manche sind nützlich

Dies ist hier genauso. AD berechnet in der Regel nicht die volle Jacobi-Matrix mit der
Kettenregel in einem Durchlauf und materialisiert daher auch nicht die
Jacobi-Matrizen der einzelnen Funktionen der Zwischenschritte. Stattdessen
werden (in der Regel) die einzelnen Spalten (forward-mode AD) oder 
Zeilen (reverse-mode AD)
der Jacobi-Matrix nacheinander berechnet. Trotzdem liefert das hier vorgestellte
Modell die Erklärung, in welchen Situationen welche Variante von AD typischerweise
zu bevorzugen ist:

- Für $f\colon \mathbb{R}^n \to \mathbb{R}$ hat reverse-mode AD eine geringere Komplexität, weil der Gradient in einem einzelnen Aufruf berechnet werden kann -- unabhängig von der Anzahl der Variablen. Im Gegensatz dazu hat forward-mode AD eine Komplexität, die mit der Anzahl der Variablen skaliert.
- Für $f\colon \mathbb{R}^n \to \mathbb{R}^n$ ist forward-mode AD typischerweise besser, weil im Gegensatz zu reverse-mode AD keine Zwischenwerte gespeichert werden müssen (was einen gewissen Overhead verursacht).
- Für Kurven $f\colon \mathbb{R} \to \mathbb{R}^n$ ist forward-mode AD besser.
"""

# ╔═╡ 698c4bd2-29f9-4830-918f-3b55395d861c
md"""
## Weiterführende Quellen

Es gibt viel Material über AD (in Julia), zum Beispiel

- [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl)
- [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl)
- [Lecture notes "Advanced Topics from Scientific Computing" by Jürgen Fuhrmann](https://www.wias-berlin.de/people/fuhrmann/AdSciComp-WS2324/)
- [https://dj4earth.github.io/MPE24](https://dj4earth.github.io/MPE24/)
- [A JuliaLabs workshop](https://github.com/JuliaLabs/Workshop-OIST/blob/master/Lecture%203b%20--%20AD%20in%2010%20minutes.ipynb)
"""

# ╔═╡ 4340e86a-e0fe-4cfe-9d1a-9bb686cbb2fd
md"""
# Appendix

You can find code and utility material in this appendix.
"""

# ╔═╡ 42fa44f5-06df-41a1-9b33-71386a0cb6d2
space = html"<br><br><br>";

# ╔═╡ 96351793-9bcc-4376-9c95-b6b42f061ad8
space

# ╔═╡ bc148aac-1ef7-4611-b187-72f1255ff05f
space

# ╔═╡ 92377a23-ac4f-4d5f-9d57-a0a03693307c
space

# ╔═╡ e771a1f9-6813-4383-b34d-83530de4aa2e
md"""
#### Installing packages

_First, we will install (and compile) some packages. This can take a few minutes when  running this notebook for the first time._
"""


# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
CairoMakie = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0"
Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
SymPyPythonCall = "bc8888f7-b21e-4b7c-a06a-5d9c9496438c"

[compat]
BenchmarkTools = "~1.5.0"
CairoMakie = "~0.12.16"
Enzyme = "~0.13.19"
LaTeXStrings = "~1.4.0"
PlutoUI = "~0.7.60"
SymPyPythonCall = "~0.3.0"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.10.7"
manifest_format = "2.0"
project_hash = "60bf03ed26729b81016a69af0157b613852ddde3"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "Test"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.AbstractTrees]]
git-tree-sha1 = "2d9c9a55f9c93e8887ad391fbae72f8ef55e1177"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.4.5"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "50c3c56a52972d78e8be9fd135bfb91c9574c140"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.1.1"
weakdeps = ["StaticArrays"]

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

[[deps.AdaptivePredicates]]
git-tree-sha1 = "7e651ea8d262d2d74ce75fdf47c4d63c07dba7a6"
uuid = "35492f91-a3bd-45ad-95db-fcad7dcfedb7"
version = "1.2.0"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.Animations]]
deps = ["Colors"]
git-tree-sha1 = "e092fa223bf66a3c41f9c022bd074d916dc303e7"
uuid = "27a7e980-b3e6-11e9-2bcd-0b925532e340"
version = "0.4.2"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Automa]]
deps = ["PrecompileTools", "SIMD", "TranscodingStreams"]
git-tree-sha1 = "a8f503e8e1a5f583fbef15a8440c8c7e32185df2"
uuid = "67c07d97-cdcb-5c2c-af73-a7f9c32a568b"
version = "1.1.0"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "01b8ccb13d68535d73d2b0c23e39bd23155fb712"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.1.0"

[[deps.AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "16351be62963a67ac4083f748fdb3cca58bfd52f"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.7"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "f1dff6729bc61f4d49e140da1af55dcd1ac97b2f"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.5.0"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "8873e196c2eb87962a2048b3b8e08946535864a1"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+2"

[[deps.CEnum]]
git-tree-sha1 = "389ad5c84de1ae7cf0e28e381131c98ea87d54fc"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.5.0"

[[deps.CRC32c]]
uuid = "8bf52ea8-c179-5cab-976a-9e18b702a9bc"

[[deps.CRlibm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e329286945d0cfc04456972ea732551869af1cfc"
uuid = "4e9b3aee-d8a1-5a3d-ad8b-7d824db253f0"
version = "1.0.1+0"

[[deps.Cairo]]
deps = ["Cairo_jll", "Colors", "Glib_jll", "Graphics", "Libdl", "Pango_jll"]
git-tree-sha1 = "71aa551c5c33f1a4415867fe06b7844faadb0ae9"
uuid = "159f3aea-2a34-519c-b102-8c37f9878175"
version = "1.1.1"

[[deps.CairoMakie]]
deps = ["CRC32c", "Cairo", "Cairo_jll", "Colors", "FileIO", "FreeType", "GeometryBasics", "LinearAlgebra", "Makie", "PrecompileTools"]
git-tree-sha1 = "c3161fbfe99d9d7ee121cf2017d49966b152857c"
uuid = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0"
version = "0.12.16"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "009060c9a6168704143100f36ab08f06c2af4642"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.2+1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "3e4b134270b372f2ed4d4d0e936aabaefc1802bc"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.25.0"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.ColorBrewer]]
deps = ["Colors", "JSON", "Test"]
git-tree-sha1 = "61c5334f33d91e570e1d0c3eb5465835242582c4"
uuid = "a2cac450-b92f-5266-8821-25eda20663c8"
version = "0.4.0"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "c785dfb1b3bfddd1da557e861b919819b82bbe5b"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.27.1"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "b10d0b65641d57b8b4d5e234446582de5047050d"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.5"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "a1f44953f2382ebb937d60dafbe2deea4bd23249"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.10.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "362a287c3aa50601b0bc359053d5c2468f0e7ce0"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.11"

[[deps.CommonEq]]
git-tree-sha1 = "6b0f0354b8eb954cdba708fb262ef00ee7274468"
uuid = "3709ef60-1bee-4518-9f2f-acd86f176c50"
version = "0.2.1"

[[deps.CommonSolve]]
git-tree-sha1 = "0eee5eb66b1cf62cd6ad1b460238e60e4b09400c"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.4"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "8ae8d32e09f0dcf42a36b90d4e17f5dd2e4c4215"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.16.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.CondaPkg]]
deps = ["JSON3", "Markdown", "MicroMamba", "Pidfile", "Pkg", "Preferences", "TOML"]
git-tree-sha1 = "8f7faef2ca039ee068cd971a80ccd710d23fb2eb"
uuid = "992eb4ea-22a4-4c89-a5bb-47a3300528ab"
version = "0.2.23"

[[deps.ConstructionBase]]
git-tree-sha1 = "76219f1ed5771adbb096743bff43fb5fdd4c1157"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.8"
weakdeps = ["IntervalSets", "LinearAlgebra", "StaticArrays"]

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "1d0a14036acb104d9e89698bd408f63ab58cdc82"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.20"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelaunayTriangulation]]
deps = ["AdaptivePredicates", "EnumX", "ExactPredicates", "Random"]
git-tree-sha1 = "e1371a23fd9816080c828d0ce04373857fe73d33"
uuid = "927a84f5-c5f4-47a5-9785-b46e178433df"
version = "1.6.3"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["AliasTables", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "3101c32aab536e7a27b1763c0797dba151b899ad"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.113"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsTestExt = "Test"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e3290f2d49e661fbd94046d7e3726ffcb2d41053"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.4+0"

[[deps.EnumX]]
git-tree-sha1 = "bdb1942cd4c45e3c678fd11569d5cccd80976237"
uuid = "4e289a0a-7415-4d19-859d-a7e5c4648b56"
version = "1.0.4"

[[deps.Enzyme]]
deps = ["CEnum", "EnzymeCore", "Enzyme_jll", "GPUCompiler", "LLVM", "Libdl", "LinearAlgebra", "ObjectFile", "PrecompileTools", "Preferences", "Printf", "Random", "SparseArrays"]
git-tree-sha1 = "4143c72ad44b037b328fb1c89be03b0785f3f275"
uuid = "7da242da-08ed-463a-9acd-ee780be4f1d9"
version = "0.13.19"

    [deps.Enzyme.extensions]
    EnzymeBFloat16sExt = "BFloat16s"
    EnzymeChainRulesCoreExt = "ChainRulesCore"
    EnzymeLogExpFunctionsExt = "LogExpFunctions"
    EnzymeSpecialFunctionsExt = "SpecialFunctions"
    EnzymeStaticArraysExt = "StaticArrays"

    [deps.Enzyme.weakdeps]
    BFloat16s = "ab4f0b2a-ad5b-11e8-123f-65d77653426b"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    LogExpFunctions = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.EnzymeCore]]
git-tree-sha1 = "0cdb7af5c39e92d78a0ee8d0a447d32f7593137e"
uuid = "f151be2c-9106-41f4-ab19-57ee4f262869"
version = "0.8.8"
weakdeps = ["Adapt"]

    [deps.EnzymeCore.extensions]
    AdaptExt = "Adapt"

[[deps.Enzyme_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "9d829428a2fe57eac24daf342b9a85efd520e9fa"
uuid = "7cc45869-7501-5eee-bdea-0790c847d4ef"
version = "0.0.167+0"

[[deps.ExactPredicates]]
deps = ["IntervalArithmetic", "Random", "StaticArrays"]
git-tree-sha1 = "b3f2ff58735b5f024c392fde763f29b057e4b025"
uuid = "429591f6-91af-11e9-00e2-59fbe8cec110"
version = "2.2.8"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e51db81749b0777b2147fbe7b783ee79045b8e99"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.6.4+1"

[[deps.ExprTools]]
git-tree-sha1 = "27415f162e6028e81c72b82ef756bf321213b6ec"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.10"

[[deps.Extents]]
git-tree-sha1 = "81023caa0021a41712685887db1fc03db26f41f5"
uuid = "411431e0-e8b7-467b-b5e0-f676ba4f2910"
version = "0.1.4"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "8cc47f299902e13f90405ddb5bf87e5d474c0d38"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "6.1.2+0"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "4820348781ae578893311153d69049a93d05f39d"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.8.0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4d81ed14783ec49ce9f2e168208a12ce1815aa25"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+1"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "2dd20384bf8c6d411b5c7370865b1e9b26cb2ea3"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.16.6"

    [deps.FileIO.extensions]
    HTTPExt = "HTTP"

    [deps.FileIO.weakdeps]
    HTTP = "cd3eb016-35fb-5094-929b-558a96fad6f3"

[[deps.FilePaths]]
deps = ["FilePathsBase", "MacroTools", "Reexport", "Requires"]
git-tree-sha1 = "919d9412dbf53a2e6fe74af62a73ceed0bce0629"
uuid = "8fc22ac5-c921-52a6-82fd-178b2807b824"
version = "0.8.3"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates"]
git-tree-sha1 = "7878ff7172a8e6beedd1dea14bd27c3c6340d361"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.22"
weakdeps = ["Mmap", "Test"]

    [deps.FilePathsBase.extensions]
    FilePathsBaseMmapExt = "Mmap"
    FilePathsBaseTestExt = "Test"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "6a70198746448456524cb442b8af316927ff3e1a"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.13.0"
weakdeps = ["PDMats", "SparseArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "db16beca600632c95fc8aca29890d83788dd8b23"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.96+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.FreeType]]
deps = ["CEnum", "FreeType2_jll"]
git-tree-sha1 = "907369da0f8e80728ab49c1c7e09327bf0d6d999"
uuid = "b38be410-82b0-50bf-ab77-7b57e271db43"
version = "4.1.1"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "786e968a8d2fb167f2e4880baba62e0e26bd8e4e"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.3+1"

[[deps.FreeTypeAbstraction]]
deps = ["ColorVectorSpace", "Colors", "FreeType", "GeometryBasics"]
git-tree-sha1 = "d52e255138ac21be31fa633200b65e4e71d26802"
uuid = "663a7486-cb36-511b-a19d-713bb74d65c9"
version = "0.10.6"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1ed150b39aebcc805c26b93a8d0122c940f64ce2"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.14+0"

[[deps.GPUCompiler]]
deps = ["ExprTools", "InteractiveUtils", "LLVM", "Libdl", "Logging", "PrecompileTools", "Preferences", "Scratch", "Serialization", "TOML", "TimerOutputs", "UUIDs"]
git-tree-sha1 = "72408a76694e87e735f515a499ba1e069d232d39"
uuid = "61eb1bfa-7361-4325-ad38-22787b887f55"
version = "1.0.1"

[[deps.GeoFormatTypes]]
git-tree-sha1 = "59107c179a586f0fe667024c5eb7033e81333271"
uuid = "68eda718-8dee-11e9-39e7-89f7f65f511f"
version = "0.4.2"

[[deps.GeoInterface]]
deps = ["Extents", "GeoFormatTypes"]
git-tree-sha1 = "826b4fd69438d9ce4d2b19de6bc2f970f45f0f88"
uuid = "cf35fbd7-0cd7-5166-be24-54bfbe79505f"
version = "1.3.8"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "Extents", "GeoInterface", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "b62f2b2d76cee0d61a2ef2b3118cd2a3215d3134"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.11"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Giflib_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "0224cce99284d997f6880a42ef715a37c99338d1"
uuid = "59f7168a-df46-5410-90c8-f2779963d0ec"
version = "5.2.2+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "48b5d4c75b2c9078ead62e345966fa51a25c05ad"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.82.2+1"

[[deps.Graphics]]
deps = ["Colors", "LinearAlgebra", "NaNMath"]
git-tree-sha1 = "a641238db938fff9b2f60d08ed9030387daf428c"
uuid = "a2bd30eb-e257-5431-a919-1863eab51364"
version = "1.1.3"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "01979f9b37367603e2848ea225918a3b3861b606"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+1"

[[deps.GridLayoutBase]]
deps = ["GeometryBasics", "InteractiveUtils", "Observables"]
git-tree-sha1 = "dc6bed05c15523624909b3953686c5f5ffa10adc"
uuid = "3955a311-db13-416c-9275-1d80ed98e5e9"
version = "0.11.1"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "401e4f3f30f43af2c8478fc008da50096ea5240f"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "8.3.1+0"

[[deps.HypergeometricFunctions]]
deps = ["LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "b1c2585431c382e3fe5805874bda6aea90a95de9"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.25"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "b6d6bfdd7ce25b0f9b2f6b3dd56b2673a66c8770"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.5"

[[deps.ImageAxes]]
deps = ["AxisArrays", "ImageBase", "ImageCore", "Reexport", "SimpleTraits"]
git-tree-sha1 = "e12629406c6c4442539436581041d372d69c55ba"
uuid = "2803e5a7-5153-5ecf-9a86-9b4c37f5f5ac"
version = "0.6.12"

[[deps.ImageBase]]
deps = ["ImageCore", "Reexport"]
git-tree-sha1 = "eb49b82c172811fd2c86759fa0553a2221feb909"
uuid = "c817782e-172a-44cc-b673-b171935fbb9e"
version = "0.1.7"

[[deps.ImageCore]]
deps = ["ColorVectorSpace", "Colors", "FixedPointNumbers", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "PrecompileTools", "Reexport"]
git-tree-sha1 = "8c193230235bbcee22c8066b0374f63b5683c2d3"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.10.5"

[[deps.ImageIO]]
deps = ["FileIO", "IndirectArrays", "JpegTurbo", "LazyModules", "Netpbm", "OpenEXR", "PNGFiles", "QOI", "Sixel", "TiffImages", "UUIDs", "WebP"]
git-tree-sha1 = "696144904b76e1ca433b886b4e7edd067d76cbf7"
uuid = "82e4d734-157c-48bb-816b-45c225c6df19"
version = "0.6.9"

[[deps.ImageMetadata]]
deps = ["AxisArrays", "ImageAxes", "ImageBase", "ImageCore"]
git-tree-sha1 = "2a81c3897be6fbcde0802a0ebe6796d0562f63ec"
uuid = "bc367c6b-8a6b-528e-b4bd-a4b897500b49"
version = "0.9.10"

[[deps.Imath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "0936ba688c6d201805a83da835b55c61a180db52"
uuid = "905a6f67-0a94-5f89-b386-d35d92009cd1"
version = "3.1.11+0"

[[deps.IndirectArrays]]
git-tree-sha1 = "012e604e1c7458645cb8b436f8fba789a51b257f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "1.0.0"

[[deps.Inflate]]
git-tree-sha1 = "d1b1b796e47d94588b3757fe84fbf65a5ec4a80d"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.5"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "10bd689145d2c3b2a9844005d01087cc1194e79e"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2024.2.1+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "88a101217d7cb38a7b481ccd50d21876e1d1b0e0"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.15.1"
weakdeps = ["Unitful"]

    [deps.Interpolations.extensions]
    InterpolationsUnitfulExt = "Unitful"

[[deps.IntervalArithmetic]]
deps = ["CRlibm_jll", "LinearAlgebra", "MacroTools", "RoundingEmulator"]
git-tree-sha1 = "24c095b1ec7ee58b936985d31d5df92f9b9cfebb"
uuid = "d1acc4aa-44c8-5952-acd4-ba5d80a2a253"
version = "0.22.19"

    [deps.IntervalArithmetic.extensions]
    IntervalArithmeticDiffRulesExt = "DiffRules"
    IntervalArithmeticForwardDiffExt = "ForwardDiff"
    IntervalArithmeticIntervalSetsExt = "IntervalSets"
    IntervalArithmeticRecipesBaseExt = "RecipesBase"

    [deps.IntervalArithmetic.weakdeps]
    DiffRules = "b552c78f-8df3-52c6-915a-8e097449b14b"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"

[[deps.IntervalSets]]
git-tree-sha1 = "dba9ddf07f77f60450fe5d2e2beb9854d9a49bd0"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.10"
weakdeps = ["Random", "RecipesBase", "Statistics"]

    [deps.IntervalSets.extensions]
    IntervalSetsRandomExt = "Random"
    IntervalSetsRecipesBaseExt = "RecipesBase"
    IntervalSetsStatisticsExt = "Statistics"

[[deps.InverseFunctions]]
git-tree-sha1 = "a779299d77cd080bf77b97535acecd73e1c5e5cb"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.17"
weakdeps = ["Dates", "Test"]

    [deps.InverseFunctions.extensions]
    InverseFunctionsDatesExt = "Dates"
    InverseFunctionsTestExt = "Test"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.Isoband]]
deps = ["isoband_jll"]
git-tree-sha1 = "f9b6d97355599074dc867318950adaa6f9946137"
uuid = "f1662d9f-8043-43de-a69a-05efc1cc6ff4"
version = "0.1.1"

[[deps.IterTools]]
git-tree-sha1 = "42d5f897009e7ff2cf88db414a389e5ed1bdd023"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.10.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "be3dc50a92e5a386872a493a10050136d4703f9b"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.6.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JSON3]]
deps = ["Dates", "Mmap", "Parsers", "PrecompileTools", "StructTypes", "UUIDs"]
git-tree-sha1 = "1d322381ef7b087548321d3f878cb4c9bd8f8f9b"
uuid = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"
version = "1.14.1"

    [deps.JSON3.extensions]
    JSON3ArrowExt = ["ArrowTypes"]

    [deps.JSON3.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"

[[deps.JpegTurbo]]
deps = ["CEnum", "FileIO", "ImageCore", "JpegTurbo_jll", "TOML"]
git-tree-sha1 = "fa6d0bcff8583bac20f1ffa708c3913ca605c611"
uuid = "b835a17e-a41a-41e7-81f0-2f016b05efe0"
version = "0.1.5"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "25ee0be4d43d0269027024d75a24c24d6c6e590c"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.0.4+0"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "7d703202e65efa1369de1279c162b915e245eed1"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.9"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "170b660facf5df5de098d866564877e119141cbd"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.2+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LLVM]]
deps = ["CEnum", "LLVMExtra_jll", "Libdl", "Preferences", "Printf", "Unicode"]
git-tree-sha1 = "d422dfd9707bec6617335dc2ea3c5172a87d5908"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "9.1.3"

    [deps.LLVM.extensions]
    BFloat16sExt = "BFloat16s"

    [deps.LLVM.weakdeps]
    BFloat16s = "ab4f0b2a-ad5b-11e8-123f-65d77653426b"

[[deps.LLVMExtra_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "05a8bd5a42309a9ec82f700876903abce1017dd3"
uuid = "dad2f222-ce93-54a1-a47d-0025e8a3acab"
version = "0.0.34+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "78211fb6cbc872f77cad3fc0b6cf647d923f4929"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "18.1.7+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "854a9c268c43b77b0a27f22d7fab8d33cdb3a731"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.2+1"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.Latexify]]
deps = ["Format", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "ce5f5621cac23a86011836badfedf664a612cee4"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.5"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SparseArraysExt = "SparseArrays"
    SymEngineExt = "SymEngine"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LazyModules]]
git-tree-sha1 = "a560dd966b386ac9ae60bdd3a3d3a326062d3c3e"
uuid = "8cdb02fc-e678-4876-92c5-9defec4f444e"
version = "0.3.1"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.4.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.6.4+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll"]
git-tree-sha1 = "8be878062e0ffa2c3f67bb58a595375eda5de80b"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.11.0+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "ff3b4b9d35de638936a525ecd36e86a8bb919d11"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.7.0+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c6ce1e19f3aec9b59186bdf06cdf3c4fc5f5f3e6"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.50.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "61dfdba58e585066d8bce214c5a51eaa0539f269"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.17.0+1"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "84eef7acd508ee5b3e956a2ae51b05024181dee0"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.40.2+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "3eb79b0ca5764d4799c06699573fd8f533259713"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.4.0+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "edbf5309f9ddf1cab25afc344b1e8150b7c832f9"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.40.2+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "a2d09619db4e765091ee5c6ffe8872849de0feea"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.28"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "oneTBB_jll"]
git-tree-sha1 = "f046ccd0c6db2832a9f639e2c669c6fe867e5f4f"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2024.2.0+0"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "2fa9ee3e63fd3a4f7a9a4f4744a52f4856de82df"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.13"

[[deps.Makie]]
deps = ["Animations", "Base64", "CRC32c", "ColorBrewer", "ColorSchemes", "ColorTypes", "Colors", "Contour", "Dates", "DelaunayTriangulation", "Distributions", "DocStringExtensions", "Downloads", "FFMPEG_jll", "FileIO", "FilePaths", "FixedPointNumbers", "Format", "FreeType", "FreeTypeAbstraction", "GeometryBasics", "GridLayoutBase", "ImageBase", "ImageIO", "InteractiveUtils", "Interpolations", "IntervalSets", "InverseFunctions", "Isoband", "KernelDensity", "LaTeXStrings", "LinearAlgebra", "MacroTools", "MakieCore", "Markdown", "MathTeXEngine", "Observables", "OffsetArrays", "Packing", "PlotUtils", "PolygonOps", "PrecompileTools", "Printf", "REPL", "Random", "RelocatableFolders", "Scratch", "ShaderAbstractions", "Showoff", "SignedDistanceFields", "SparseArrays", "Statistics", "StatsBase", "StatsFuns", "StructArrays", "TriplotBase", "UnicodeFun", "Unitful"]
git-tree-sha1 = "5e4e0e027642293da251bf35dac408d692ccba8b"
uuid = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
version = "0.21.16"

[[deps.MakieCore]]
deps = ["ColorTypes", "GeometryBasics", "IntervalSets", "Observables"]
git-tree-sha1 = "ae4dbe0fcf1594ed98594e5f4ee685295a2a6f74"
uuid = "20f20a25-4f0e-4fdf-b5d1-57303727442b"
version = "0.8.10"

[[deps.MappedArrays]]
git-tree-sha1 = "2dab0221fe2b0f2cb6754eaa743cc266339f527e"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.2"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MathTeXEngine]]
deps = ["AbstractTrees", "Automa", "DataStructures", "FreeTypeAbstraction", "GeometryBasics", "LaTeXStrings", "REPL", "RelocatableFolders", "UnicodeFun"]
git-tree-sha1 = "f45c8916e8385976e1ccd055c9874560c257ab13"
uuid = "0a4f8689-d25c-4efe-a92b-7142dfc1aa53"
version = "0.6.2"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+1"

[[deps.MicroMamba]]
deps = ["Pkg", "Scratch", "micromamba_jll"]
git-tree-sha1 = "011cab361eae7bcd7d278f0a7a00ff9c69000c51"
uuid = "0b3b1443-0f03-428d-bdfb-f27f9c1191ea"
version = "0.1.14"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "7b86a5d4d70a9f5cdf2dacb3cbe6d251d1a61dbe"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.4"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.1.10"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.Netpbm]]
deps = ["FileIO", "ImageCore", "ImageMetadata"]
git-tree-sha1 = "d92b107dbb887293622df7697a2223f9f8176fcd"
uuid = "f09324ee-3d7c-5217-9330-fc30815ba969"
version = "1.1.1"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.ObjectFile]]
deps = ["Reexport", "StructIO"]
git-tree-sha1 = "7249afa1c4dfd86bfbcc9b28939ab6ef844f4e11"
uuid = "d8793406-e978-5875-9003-1fc021f44a92"
version = "0.4.2"

[[deps.Observables]]
git-tree-sha1 = "7438a59546cf62428fc9d1bc94729146d37a7225"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.5"

[[deps.OffsetArrays]]
git-tree-sha1 = "39d000d9c33706b8364817d8894fae1548f40295"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.14.2"
weakdeps = ["Adapt"]

    [deps.OffsetArrays.extensions]
    OffsetArraysAdaptExt = "Adapt"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.23+4"

[[deps.OpenEXR]]
deps = ["Colors", "FileIO", "OpenEXR_jll"]
git-tree-sha1 = "97db9e07fe2091882c765380ef58ec553074e9c7"
uuid = "52e1d378-f018-4a11-a4be-720524705ac7"
version = "0.3.3"

[[deps.OpenEXR_jll]]
deps = ["Artifacts", "Imath_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "8292dd5c8a38257111ada2174000a33745b06d4e"
uuid = "18a262bb-aa17-5467-a713-aee519bc75cb"
version = "3.2.4+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+2"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7493f61f55a6cce7325f197443aa80d32554ba10"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.0.15+1"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6703a85cb3781bd5909d48730a67205f3f31a575"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.3+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "12f1439c4f986bb868acda6ea33ebc78e19b95ad"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.7.0"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "949347156c25054de2db3b166c52ac4728cbad65"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.31"

[[deps.PNGFiles]]
deps = ["Base64", "CEnum", "ImageCore", "IndirectArrays", "OffsetArrays", "libpng_jll"]
git-tree-sha1 = "67186a2bc9a90f9f85ff3cc8277868961fb57cbd"
uuid = "f57f5aa1-a3ce-4bc8-8ab9-96f992907883"
version = "0.4.3"

[[deps.Packing]]
deps = ["GeometryBasics"]
git-tree-sha1 = "bc5bf2ea3d5351edf285a06b0016788a121ce92c"
uuid = "19eb6ba3-879d-56ad-ad62-d5c202156566"
version = "0.5.1"

[[deps.PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "0fac6313486baae819364c52b4f483450a9d793f"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.12"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e127b609fb9ecba6f201ba7ab753d5a605d53801"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.54.1+0"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

[[deps.Pidfile]]
deps = ["FileWatching", "Test"]
git-tree-sha1 = "2d8aaf8ee10df53d0dfb9b8ee44ae7c04ced2b03"
uuid = "fa939f87-e72e-5be4-a000-7fc836dbe307"
version = "1.3.0"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "35621f10a7531bc8fa58f74610b1bfb70a3cfc6b"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.43.4+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.10.0"

[[deps.PkgVersion]]
deps = ["Pkg"]
git-tree-sha1 = "f9501cc0430a26bc3d156ae1b5b0c1b47af4d6da"
uuid = "eebad327-c553-4316-9ea0-9fa01ccd7688"
version = "0.3.3"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "StableRNGs", "Statistics"]
git-tree-sha1 = "3ca9a356cd2e113c420f2c13bea19f8d3fb1cb18"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.3"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "eba4810d5e6a01f612b948c9fa94f905b49087b0"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.60"

[[deps.PolygonOps]]
git-tree-sha1 = "77b3d3605fc1cd0b42d95eba87dfcd2bf67d5ff6"
uuid = "647866c9-e3ac-4575-94e7-e3d426903924"
version = "0.1.2"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "9306f6085165d270f7e3db02af26a400d580f5c6"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.3"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Profile]]
deps = ["Printf"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "8f6bc219586aef8baf0ff9a5fe16ee9c70cb65e4"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.10.2"

[[deps.PtrArrays]]
git-tree-sha1 = "77a42d78b6a92df47ab37e177b2deac405e1c88f"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.2.1"

[[deps.PythonCall]]
deps = ["CondaPkg", "Dates", "Libdl", "MacroTools", "Markdown", "Pkg", "REPL", "Requires", "Serialization", "Tables", "UnsafePointers"]
git-tree-sha1 = "06a778ec6d6e76b0c2fb661436a18bce853ec45f"
uuid = "6099a3de-0909-46bc-b1f4-468b9a2dfc0d"
version = "0.9.23"

[[deps.QOI]]
deps = ["ColorTypes", "FileIO", "FixedPointNumbers"]
git-tree-sha1 = "8b3fc30bc0390abdce15f8822c889f669baed73d"
uuid = "4b34888f-f399-49d4-9bb3-47ed5cae4e65"
version = "1.0.1"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "cda3b045cf9ef07a08ad46731f5a3165e56cf3da"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.11.1"
weakdeps = ["Enzyme"]

    [deps.QuadGK.extensions]
    QuadGKEnzymeExt = "Enzyme"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RangeArrays]]
git-tree-sha1 = "b9039e93773ddcfc828f12aadf7115b4b4d225f5"
uuid = "b3c3ace0-ae52-54e7-9d0b-2c1406fd6b9d"
version = "0.3.2"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "1342a47bf3260ee108163042310d26f2be5ec90b"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.5"
weakdeps = ["FixedPointNumbers"]

    [deps.Ratios.extensions]
    RatiosFixedPointNumbersExt = "FixedPointNumbers"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "852bd0f55565a9e973fcfee83a84413270224dc4"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.8.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "58cdd8fb2201a6267e1db87ff148dd6c1dbd8ad8"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.5.1+0"

[[deps.RoundingEmulator]]
git-tree-sha1 = "40b9edad2e5287e05bd413a38f61a8ff55b9557b"
uuid = "5eaf0fd0-dfba-4ccb-bf02-d820a40db705"
version = "0.2.1"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SIMD]]
deps = ["PrecompileTools"]
git-tree-sha1 = "52af86e35dd1b177d051b12681e1c581f53c281b"
uuid = "fdea26ae-647d-5447-a871-4b548cad5224"
version = "3.7.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "3bac05bc7e74a75fd9cba4295cde4045d9fe2386"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.1"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.ShaderAbstractions]]
deps = ["ColorTypes", "FixedPointNumbers", "GeometryBasics", "LinearAlgebra", "Observables", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "79123bc60c5507f035e6d1d9e563bb2971954ec8"
uuid = "65257c39-d410-5151-9873-9b3e5be5013e"
version = "0.4.1"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SignedDistanceFields]]
deps = ["Random", "Statistics", "Test"]
git-tree-sha1 = "d263a08ec505853a5ff1c1ebde2070419e3f28e9"
uuid = "73760f76-fbc4-59ce-8f25-708e95d2df96"
version = "0.4.0"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[deps.Sixel]]
deps = ["Dates", "FileIO", "ImageCore", "IndirectArrays", "OffsetArrays", "REPL", "libsixel_jll"]
git-tree-sha1 = "2da10356e31327c7096832eb9cd86307a50b1eb6"
uuid = "45858cf5-a6b0-47a3-bbea-62219f50df47"
version = "0.1.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "66e0a8e672a0bdfca2c3f5937efb8538b9ddc085"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.10.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "2f5d4697f21388cbe1ff299430dd169ef97d7e14"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.4.0"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.StableRNGs]]
deps = ["Random"]
git-tree-sha1 = "83e6cce8324d49dfaf9ef059227f91ed4441a8e5"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.2"

[[deps.StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "46e589465204cd0c08b4bd97385e4fa79a0c770c"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.1"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "777657803913ffc7e8cc20f0fd04b634f871af8f"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.8"
weakdeps = ["ChainRulesCore", "Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "192954ef1208c7019899fbf8049e717f92959682"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.3"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.10.0"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "5cf7606d6cef84b543b483848d4ae08ad9832b21"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.3"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "b423576adc27097764a90e163157bcfc9acf0f46"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.3.2"
weakdeps = ["ChainRulesCore", "InverseFunctions"]

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

[[deps.StructArrays]]
deps = ["ConstructionBase", "DataAPI", "Tables"]
git-tree-sha1 = "9537ef82c42cdd8c5d443cbc359110cbb36bae10"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.21"

    [deps.StructArrays.extensions]
    StructArraysAdaptExt = "Adapt"
    StructArraysGPUArraysCoreExt = ["GPUArraysCore", "KernelAbstractions"]
    StructArraysLinearAlgebraExt = "LinearAlgebra"
    StructArraysSparseArraysExt = "SparseArrays"
    StructArraysStaticArraysExt = "StaticArrays"

    [deps.StructArrays.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    KernelAbstractions = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.StructIO]]
git-tree-sha1 = "c581be48ae1cbf83e899b14c07a807e1787512cc"
uuid = "53d494c1-5632-5724-8f4c-31dff12d585f"
version = "0.3.1"

[[deps.StructTypes]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "159331b30e94d7b11379037feeb9b690950cace8"
uuid = "856f2bd8-1eba-4b0a-8007-ebc267875bd4"
version = "1.11.0"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.2.1+1"

[[deps.SymPyCore]]
deps = ["CommonEq", "CommonSolve", "Latexify", "LinearAlgebra", "Markdown", "RecipesBase", "SpecialFunctions"]
git-tree-sha1 = "159306f3e8fa7a3380afd8daaa0f13efe86d7fde"
uuid = "458b697b-88f0-4a86-b56b-78b75cfb3531"
version = "0.2.6"

    [deps.SymPyCore.extensions]
    SymPyCoreTermInterfaceExt = "TermInterface"

    [deps.SymPyCore.weakdeps]
    TermInterface = "8ea1fca8-c5ef-4a55-8b96-4e9afe9c9a3c"

[[deps.SymPyPythonCall]]
deps = ["CommonEq", "CommonSolve", "CondaPkg", "LinearAlgebra", "PythonCall", "SpecialFunctions", "SymPyCore"]
git-tree-sha1 = "7aab2feff2eafa8016d91533fd71d112ccb19b13"
uuid = "bc8888f7-b21e-4b7c-a06a-5d9c9496438c"
version = "0.3.0"

    [deps.SymPyPythonCall.extensions]
    SymPyPythonCallSymbolicsExt = "Symbolics"

    [deps.SymPyPythonCall.weakdeps]
    Symbolics = "0c5d862f-8b57-4792-8d23-62f2024744c7"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "598cd7c1f68d1e205689b1c2fe65a9f85846f297"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.12.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TiffImages]]
deps = ["ColorTypes", "DataStructures", "DocStringExtensions", "FileIO", "FixedPointNumbers", "IndirectArrays", "Inflate", "Mmap", "OffsetArrays", "PkgVersion", "ProgressMeter", "SIMD", "UUIDs"]
git-tree-sha1 = "0248b1b2210285652fbc67fd6ced9bf0394bcfec"
uuid = "731e570b-9d59-4bfa-96dc-6df516fadf69"
version = "0.11.1"

[[deps.TimerOutputs]]
deps = ["ExprTools", "Printf"]
git-tree-sha1 = "d7298ebdfa1654583468a487e8e83fae9d72dac3"
uuid = "a759f4b9-e2f1-59dc-863e-4aeb61b1ea8f"
version = "0.5.26"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.Tricks]]
git-tree-sha1 = "7822b97e99a1672bfb1b49b668a6d46d58d8cbcb"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.9"

[[deps.TriplotBase]]
git-tree-sha1 = "4d4ed7f294cda19382ff7de4c137d24d16adc89b"
uuid = "981d1d27-644d-49a2-9326-4793e63143c3"
version = "0.1.0"

[[deps.URIs]]
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unitful]]
deps = ["Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "01915bfcd62be15329c9a07235447a89d588327c"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.21.1"
weakdeps = ["ConstructionBase", "InverseFunctions"]

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    InverseFunctionsUnitfulExt = "InverseFunctions"

[[deps.UnsafePointers]]
git-tree-sha1 = "c81331b3b2e60a982be57c046ec91f599ede674a"
uuid = "e17b2a0c-0bdf-430a-bd0c-3a23cae4ff39"
version = "1.0.0"

[[deps.WebP]]
deps = ["CEnum", "ColorTypes", "FileIO", "FixedPointNumbers", "ImageCore", "libwebp_jll"]
git-tree-sha1 = "aa1ca3c47f119fbdae8770c29820e5e6119b83f2"
uuid = "e3aaa7dc-3e4b-44e0-be63-ffb868ccd7c1"
version = "0.1.3"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "c1a7aa6219628fcd757dede0ca95e245c5cd9511"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "1.0.0"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "a2fccc6559132927d4c5dc183e3e01048c6dcbd6"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.13.5+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "7d1671acbe47ac88e981868a078bd6b4e27c5191"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.42+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "9dafcee1d24c4f024e7edc92603cedba72118283"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.6+1"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "2b0e27d52ec9d8d483e2ca0b72b3cb1a8df5c27a"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.11+1"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "02054ee01980c90297412e4c809c8694d7323af3"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.4+1"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "d7155fea91a4123ef59f42c4afb5ab3b4ca95058"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.6+1"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "47e45cd78224c53109495b3e324df0c37bb61fbe"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.11+0"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "fee57a273563e273f0f53275101cd41a8153517a"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.1+1"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "1a74296303b6524a0472a8cb12d3d87a78eb3612"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.0+1"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b9ead2d2bdb27330545eb14234a2e300da61232e"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.5.0+1"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "555d1076590a6cc2fdee2ef1469451f872d8b41b"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.6+1"

[[deps.isoband_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51b5eeb3f98367157a7a12a1fb0aa5328946c03c"
uuid = "9a68df92-36a6-505f-a73e-abb412b6bfb4"
version = "0.2.3+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1827acba325fdcdf1d2647fc8d5301dd9ba43a9d"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.9.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "e17c115d55c5fbb7e52ebedb427a0dca79d4484e"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.2+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.11.0+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a22cf860a7d27e4f3498a0fe0811a7957badb38"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.3+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "b70c870239dc3d7bc094eb2d6be9b73d27bef280"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.44+0"

[[deps.libsixel_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Pkg", "libpng_jll"]
git-tree-sha1 = "7dfa0fd9c783d3d0cc43ea1af53d69ba45c447df"
uuid = "075b6546-f08a-558a-be8f-8157d0f608a5"
version = "1.10.3+1"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "490376214c4721cdaca654041f635213c6165cb3"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+2"

[[deps.libwebp_jll]]
deps = ["Artifacts", "Giflib_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libglvnd_jll", "Libtiff_jll", "libpng_jll"]
git-tree-sha1 = "ccbb625a89ec6195856a50aa2b668a5c08712c94"
uuid = "c5f90fcd-3b7e-5836-afba-fc50a0988cb2"
version = "1.4.0+0"

[[deps.micromamba_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "b4a5a3943078f9fd11ae0b5ab1bdbf7718617945"
uuid = "f8abcde7-e9b7-5caa-b8af-a437887ae8e4"
version = "1.5.8+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.52.0+1"

[[deps.oneTBB_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7d0ea0f4895ef2f5cb83645fa689e52cb55cf493"
uuid = "1317d2d5-d96f-522e-a858-c73665f53c3e"
version = "2021.12.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "35976a1216d6c066ea32cba2150c4fa682b276fc"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "10164.0.0+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "dcc541bb19ed5b0ede95581fb2e41ecf179527d2"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.6.0+0"
"""

# ╔═╡ Cell order:
# ╟─e6c64c80-773b-11ef-2379-bf6609137e69
# ╟─cd57529b-64a3-43cd-8dbf-445583c8edcc
# ╠═a7d5feb5-c3c5-4d55-b721-f9a838a22e78
# ╟─ee241f76-b1d9-4c00-9b53-020a1ba012dd
# ╠═b8b74818-30f8-4219-b293-657025589a44
# ╟─127944b9-b3eb-4df8-acc3-36d9958218ca
# ╠═48768b89-edf4-4c69-99ad-d304d2700bb4
# ╠═43ef27fd-a67d-4cf8-a8e6-3d11880b5eac
# ╟─a89b42c6-913e-4970-bd9e-8f163a4b96d8
# ╠═2558317b-14e6-4871-9bd3-5b26be391019
# ╠═c660ccf1-5bff-44f1-92b5-10465011748e
# ╠═56f7e83d-6d8f-413c-a2bb-fd89bebd28b5
# ╠═ff0adc67-3dd9-4402-b033-63a843dc8790
# ╟─9777384f-dc51-4d6b-b55f-87b5fd2c5be7
# ╠═a7195ecb-5e70-4abc-880c-063533944e91
# ╟─be74e9c0-cd17-47c5-9656-8109db96897a
# ╠═dde0b734-1c54-4a54-b112-b256506fa0a6
# ╟─927cbfbd-daa8-47ff-b100-6c0d81b5ea82
# ╠═1437a1cc-dddd-4d4f-be18-81ab0f92950b
# ╠═af3ea3f6-0b4e-4cf1-9aa7-d619fa4e846c
# ╠═7c60e778-b116-4504-a60a-72c52d91fd9d
# ╠═1f97ab58-580d-4069-992f-f611312bc71f
# ╠═84a6bc63-8e55-45ed-a6b8-150a0bf3f22a
# ╠═a7cb831e-49bf-4a16-bc5e-118d31a069bd
# ╠═48b60e62-3879-4693-9540-c35a2a5d09c7
# ╠═d2014f18-ca17-4a6c-8bcc-efeecc7f29d2
# ╟─fd1c6e36-1309-4c85-b0be-6e7a215a4f54
# ╠═2efd0312-54b4-4c64-a505-11be21ab5c18
# ╠═a43b5324-fc81-4247-b6b1-4e8f5c367cbe
# ╠═79544ce2-6bf0-4cd6-a4f2-ab2122d9bc49
# ╟─fbc63b55-5d04-4b70-8dd4-f5b3eff6d99f
# ╠═d4ad273b-b548-43c3-9e69-807b47bace27
# ╠═bb22987a-c3b8-4242-8a3e-91f773841b40
# ╠═7421edea-ec1c-47cb-9b19-77d9203d6857
# ╠═7e76a3bf-a750-4f8c-a28a-d44505d3526f
# ╠═26659be3-07ea-4844-9f91-0490faa5a082
# ╠═5695da28-2297-418b-92db-4c2271edbefd
# ╠═e9e02a31-9b3c-412c-84a9-8490a664715b
# ╠═4e4ad771-736b-4789-8ed0-43d2e1799b40
# ╟─bc2c6be4-10a9-4cb9-be8e-aa64745c32ba
# ╠═d45ce0d3-d23d-41ce-af4d-4946c3a37253
# ╠═c9b56c8c-38af-4069-b2fe-4c4cf0753f6c
# ╟─625d13ec-ce64-4330-9842-2e1596827079
# ╠═26238234-602d-4117-ab56-120dbecb1130
# ╠═1664e082-9ae2-4deb-9268-c03861aa49b2
# ╟─0f210d38-f8be-470c-ab1e-1baeae951482
# ╠═f3a85308-8448-4458-b224-e2bd1ee48077
# ╠═17421bbd-9e8d-475e-be20-b46da8cc6449
# ╟─74ebd661-8242-431d-8163-d1bbfcee52da
# ╠═63c9cd60-64b6-4755-b3b2-96174882f618
# ╠═2f5bc87b-7675-42d0-bd86-6503403a3404
# ╠═af152a4f-46f9-4bc1-8b87-ff7b3fd908d5
# ╠═e128836c-1fac-46ba-9db5-3803fa090759
# ╟─acf3dc94-3a21-4bce-862e-93ccb4d44b3a
# ╠═252caf68-b944-4f30-98ae-dc40482ef1a6
# ╠═c6230689-5f2a-4f57-ad03-8d87423fa5a2
# ╟─33aae6b2-ddb7-4a09-b672-0aa426977971
# ╠═88dbd750-7ca4-42ee-8790-e198131e9a0f
# ╠═a748d342-8b42-4394-bc8f-905387b4f16a
# ╠═a2238c71-0467-4273-be7d-f4188cf4efb3
# ╠═bc49478e-523a-49a9-813b-36924c9a9814
# ╠═fb02594f-e6fb-40c0-908c-14f2db04bfe6
# ╠═ddf7d1b0-e4fb-4e0b-ab94-455b5751323c
# ╠═f9c7430c-e383-4611-974d-472735df3bbc
# ╟─484cad0d-f0b7-4349-ba5a-8bbc89a75c16
# ╠═dd01a246-d8f0-45bb-bd65-ceba082b8996
# ╠═56cf0311-7e97-432a-8671-2e44fb89e272
# ╠═28df9335-f5e3-4afe-8d8a-3ccde842682c
# ╟─b179851e-b248-4d14-ae88-451c59380125
# ╟─ee92845a-634b-4b93-9032-6cde78c7dc8b
# ╠═b1954b3c-3dbc-411a-a6ac-29fa39615b3d
# ╟─9481863b-5fa7-4f2d-af03-dd3ca7aa433c
# ╠═f3d17d53-2fa9-4cad-a632-2cecac4e875a
# ╠═14d32872-7a1e-44c4-a60f-5a9a809e7298
# ╟─d4f127d2-7f4a-47e0-bd53-ada2066e2585
# ╠═cadc4aab-f447-4662-b92d-6f8bf90340c1
# ╠═e508256a-4b90-43c2-bb32-3cbb52974ab6
# ╠═b183eea7-7939-4f7d-875c-8d7ac8dd15f5
# ╠═61cf5a89-6f00-4599-81f1-5c9ae90f1d37
# ╟─4c4a9e9f-73b6-44e7-accf-3300bbd353d9
# ╠═51f58054-50a7-4d3d-89ec-2f867007cd54
# ╠═c3f4cfc5-c90b-404e-a18f-13c59f0b8600
# ╠═e9c6b5d2-5d4a-489b-883c-9ba58b2539f2
# ╠═62bf4cc7-4804-4b30-b0bf-863e4b527cf2
# ╟─b0de5259-7237-442e-a058-2f4da305ce50
# ╠═8fcf4771-5a72-4ef7-b0ae-4d4805160c6d
# ╠═2f53b546-b02f-42cd-ab03-ec92e702868b
# ╠═7a1846df-7c5b-43af-86d9-d067d14d2752
# ╠═0967c13e-13ef-4a35-8045-36712f2feeeb
# ╠═9c20e8b7-7d90-4c68-b28c-f79feb0370ef
# ╠═9d838386-e87c-438d-90a6-56a443860249
# ╠═c651abd9-af17-4a75-821c-24eac406daf9
# ╠═505bad11-94e6-4e08-99c1-1537df969aee
# ╠═7392a184-40df-4735-8560-d82ff2e4bf70
# ╠═05c9713c-917e-4b9b-a53b-38cde05be415
# ╠═e569b49b-365e-4cee-88bf-33d583fd724c
# ╠═46d1fab5-4cab-4b3f-a4bc-b81878619503
# ╠═54d8fb0e-c42e-44c8-9357-28f8d30ac367
# ╟─a74e1db9-0f6f-48d9-8061-ea5b8d3b2f08
# ╟─698c4bd2-29f9-4830-918f-3b55395d861c
# ╟─96351793-9bcc-4376-9c95-b6b42f061ad8
# ╟─bc148aac-1ef7-4611-b187-72f1255ff05f
# ╟─92377a23-ac4f-4d5f-9d57-a0a03693307c
# ╟─4340e86a-e0fe-4cfe-9d1a-9bb686cbb2fd
# ╠═42fa44f5-06df-41a1-9b33-71386a0cb6d2
# ╟─e771a1f9-6813-4383-b34d-83530de4aa2e
# ╠═437a2d3f-7f19-4813-af1b-babd8b883310
# ╠═f05a5972-58b1-4788-a0a8-24966d6714da
# ╠═e21f7893-67e3-42ba-82e8-1297502cc1ea
# ╠═b0d18f0a-7ae7-4c9e-9e29-2f190aaae1c2
# ╠═02ed8724-fbe6-4cdd-bab6-9f7ccfed8380
# ╠═72f1a9ed-3047-4ab9-b038-10c76984c540
# ╠═fe0a3bf7-3105-437a-888b-94424ff94608
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
