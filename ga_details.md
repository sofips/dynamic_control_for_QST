# Detalles de implementación para el algoritmos de DRL 

<table>
<tr>
<td style="vertical-align: top; padding-right: 40px">

### Parámetros Físicos

| Parámetro                              | Valor                 |
|----------------------------------------|------------------------|
| Paso temporal ($dt$)                   | 0.15                  |
| Acoplamientos ($J$)                    | 1                     |
| Campo externo ($B$)                    | 100                   |
| Número de genes                        | 5N ($t \simeq 1.5N$)  |
| Tolerancia (1-F) ($\zeta$)             | 0.01                  |

</td>
<td style="vertical-align: top">

### Parámetros del algoritmo genético

| Parámetro                              | Valor                    |
|----------------------------------------|---------------------------|
| Individuos en población                | 3000                      |
| Selección de padres                    |300 usando 'sss'          |
| Elitismo                               | 300   (10%)                   |
| Crossover                              | Uniforme (probabilidad = 0.8)              |
| Mutación                               | swap sobre N genes       |
| Saturación                             | 20                       |


</td>
</tr>
</table>
