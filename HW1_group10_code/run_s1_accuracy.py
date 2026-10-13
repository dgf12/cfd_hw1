"""Section 1.1 - error of the central-2 and backward-1 formulas for f'(x) at x = 40."""

import os
import sys

folder = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, folder)

# vai importar as funções das nossas bibliotecas 
from cfdlib.fd import fd_formula_text, truncation_term, accuracy_h_values, accuracy_study #diferenças finitas e cálculos de erros
from cfdlib.common import observed_order, save_figure #grava as figuras e mede o declive
from cfdlib.plots import plot_accuracy, finish_figures #gráficos

# Parametros aqui para não mexer nas funções
x_point = 40.0                          # onde calculamos a derivada
k_values = {1: 0.02, 2: 0.2}            # f1 e f2
schemes = ["central-2", "backward-1"]
h_min = 1e-8                            # h vai de 2*pi/k até h_min em n_points espaçados logaritmicamente devido à escala
n_points = 300
h_fit = (1e-3, 1e-1)                    # intervalo onde medimos o declive da reta do erro do gráfico - ordem da fórmula
x_limits = (1e-8, 1e3)                  #eixos
y_limits = (1e-16, 1e1)
fig_folder = os.path.join(folder, "figures") # entrega as figuras na pasta 
show_plots = True
# ----------------------

print("Formulas and truncation error:") # imprime as fórmulas e os seus erros de truncação
for scheme in schemes:
    print("  " + fd_formula_text(scheme))

for n, k in k_values.items(): # para correr f1 e f2 
    h = accuracy_h_values(k, h_min, n_points) # cria os valores de h
    errors = accuracy_study(x_point, k, schemes, h) # com o tal h, faz o erro real e truncação

    slopes = {}
    for scheme in schemes: # tirar o erro de cada esquema, mede e declive e vai comparar com a ordem teórica
        error, _ = errors[scheme]
        slopes[scheme] = observed_order(h, error, h_fit)
        expected = truncation_term(scheme)[0]
        print(f"f{n}, {scheme}: expected order {expected}, slope {slopes[scheme]:.3f}")

    fig = plot_accuracy(h, errors, slopes, f"$f_{n}$, $k={k}$", x_limits, y_limits)
    save_figure(fig, f"s1_accuracy_f{n}", fig_folder)

finish_figures(show_plots)