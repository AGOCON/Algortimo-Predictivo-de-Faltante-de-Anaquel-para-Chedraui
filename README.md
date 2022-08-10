# Algortimo Chedraui


## Objetivo
Desarrollar un algoritmo predictivo para determinar la presencia o ausencia de un producto que debiera estar exhibido en los muebles de las tiendas de la cadena Chedraui.

## Técnicas Utilizados
* Historial de auditorías de Ago para la cadena Chedraui.
* Aprendizaje automático.
* Estadística.

## Desarrollo
Las auditorías de tienda de Ago para la compañía Chedraui producen información descriptiva por SKU auditado. La dificultad principal de construcción de este software recae en el hecho de que gran parte de dicha información es de naturaleza categórica; es decir, la auditoría produce en gran parte sólo índices y etiquetas por SKU, luego las técnicas clásicas de regresión y redes neuronales están fuertemente desfavorecidas para resolver este problema. 


## Resumen de Propuestas Implementadas por el Equipo de Investigación y Desarrollo de Ago Consultores
* Usamos un algoritmo llamado [CatBoost](https://catboost.ai/en/docs/) desarrollado por la compañía [Yandex](https://yandex.com/) construído para minimizar la cantidad de falsas alertas de elementos faltantes. La idea crítica resulta ser que este algoritmo implementa las técnicas estado del arte para manejar datos tabulares (algortimos de árboles) y las combina con las mejores técnicas de predicción de variables continuas (algoritmo de descenso gradiente) para deformar de forma óptima las fronteras de decisión de los primeros.  
* Usamos un muestreo balanceado de SKUs faltantes contra encontrados en el conjunto de entrenamiento para prevenir sesgos predictivos del algoritmo. 
* Desarrollamos un algoritmo supervisa un ajuste de la frontera de decisión del algortimo catboost de forma que dicho ajuste minimize el número de falsos positivos para catboost.
* Para maximizar y balancear la precisión por formato de tienda proponemos un muestreo estratificado donde generamos un super-muestreo para las categorías de tienda desfavorecidas usando [redes neuronales generativas adversariales](https://arxiv.org/abs/1406.2661).




### Autores:
* Víctor Manuel Sánchez
* Francisco Polanco
* Ramiro López 

### Líder de Proyecto:
* Víctor Manuel Sánchez

### Responsable de este Repositorio:
* Ramiro López


# Algortimo-Predictivo-de-Faltante-de-Anaquel-para-Chedraui
