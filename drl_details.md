# Detalles de implementación para el algoritmo de DRL 

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

Estos fueron los parámetros utilizados por defecto en todos los experimentos excepto en aquellos en los que se especifique alguna variación. El programa guarda automáticamente una tarjeta de configuración con los parámetros que usó al ejecutarse que queda almacenada en el mismo directorio que los resultados. 

### Función recompensa utilizada

La función recompensa utilizada es la que se usa en el trabajo original (Zhang, 2018)
![alt text](og_reward.png)