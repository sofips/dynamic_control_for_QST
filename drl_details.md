# Detalles de implementación para el algoritmos de DRL 

<table>
<tr><th> Parámetros de aprendizaje </th><th> Parámetros del sistema</th></tr>
<tr><td>



| Parámetro                        | Valor    |
|:----------------------------------------|:---------|
| Tamaño del minibatch                    | 32       |
| Tamaño de la memoria                    | 40000    |
| Tasa de aprendizaje ($\alpha$)          | 0.01     |
| Dec. de la recompensa ($\gamma$)        | 0.95     |
| $N^\circ$ de capas ocultas              | 2        |
| Neuronas por capa oculta                | 120      |
| Epsilon inicial/final ($\epsilon$)      | 1/0.01   |
| Decaimiento                             | 0.0001   |
| Frecuencia de aprendizaje               | 5        |

</td><td>


| Parámetros                            | Valor    |
|:----------------------------------------|:---------|
| Largo de cadena (N)                     | variable |
| Paso temporal ($dt$)                    | 0.15     |
| Tiempo máximo ($\tau$)                  | $5N \times dt$|
| Int. de acoplamientos ($J$)             | 1        |
| Campo externo ($B$)                     | 100      |
| Tolerancia (1-F) ($\zeta$)              | 0.05     |

</td></tr> </table>

### Función recompensa utilizada

$$
r(t) = 
\begin{cases}
    10F_N(t), & \text{si } F_N(t) \leq 0.8 \\\\
    \dfrac{100}{1 + \exp(10(1-\zeta-F(t)))}, & \text{si } 0.8 \leq F_N(t) \leq 1 - \zeta \\\\
    2500, & \text{si } F_N(t) > 1 - \zeta
\end{cases}
$$
