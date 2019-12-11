# Reconocimiento de Imágenes
___

<p align="center">
  <img src="https://drive.google.com/uc?export=view&id=1_ofjwM6j5GQ30WPOGifmI9eOLKeUo4NN" title="Carátula">
</p>

Repositorio donde se encuentran los archivos correspondientes a la tesis ***"Aplicación de redes neuronales convolucionales para el etiquetado de imágenes automático para personas con impedimentos visuales"*** presentada en el curso de **TESIS II** en la **Universidad Nacional de Ingeniería** Lima, Perú - Diciembre 2019.

Los archivos contienen los procesos desde la descarga de imágenes, el entrenamiento del modelo, y un prototipo de una aplicación  en android.

El método usado para el entrenamiento del modelo de reconocimiento de imágenes es Redes Neuronales Convolucionales o **CNN** por sus siglas en inglés Convolutional Neural Network.


____
## Modelo Solución 

![No se cargó Modelo Solución](https://drive.google.com/uc?export=view&id=1H5KUuD_5275dO8V4KE51aT6UB-Z6wOQa "Modelo Solución")
	
 Variable |	Descripción 
 ------ | --------- 
IE(i)	|Imágenes de entrada por cada clase i
MIE(i)(j)	|Métricas de cada imagen j de la clase i (peso Kb, tamaño píxeles)
NE	|Es el número de etiquetas que tendrá el modelo solución
FA(j)	|Es la función de activación que se usará para el entrenamiento, para cada capa (j).
FP(j)	|Es la función de pérdida que se usará para el entrenamiento, para cada capa (j).
ME	|Modelo entrenado
EI(i)	|Etiqueta de imagen i
AP	|Precisión media, métrica de eficiencia.
F-Score	|Puntación F, métrica de eficiencia.


## Comenzando 🚀

En este proyecto se usa el lenguaje Python, para realizar el proceso es necesario algunas librerías, que se describen a continuación.



### Pre-requisitos 📋

Las librerías más importantes necesarias son:

* Tensorflow 
	* Versión 1.13.1 (Entrenar modelo y exportar .pb)
	* Versión 1.14, 2.0 (Entrenar modelo)
* Keras


### Procesos 🔧

Los procesos a realizarse son:

![No se cargó procesos del Modelo Solución](https://drive.google.com/uc?export=view&id=1-zxpuenttT6lMkXtCKriVdkxYdW9KkOn "Procesos del Modelo Solución")

Los archivos que se tienen son los siguientes, la mayoría de los scripts fueron desarrollado en Jupyter Notebook:

1. **Descarga de Urls de imágenes para el entrenamiento**
	* [Archivo JS Descarga URL](http://www.dropwizard.io/1.0.2/docs/) - Descargar link de imágenes del buscador de Google
		* Link de referencia: [How to create a deep learning dataset using Google Images](https://www.pyimagesearch.com/2017/12/04/how-to-create-a-deep-learning-dataset-using-google-images/)
	* [Dataset Imagenet](http://image-net.org/synset?wnid=n02084071) - Descargar del dataset de Imagenet por synset
2. **Descarga de imágenes**
	* [Descarga Imágenes](http://www.dropwizard.io/1.0.2/docs/) - Archivo Jupyter Notebook
3. **Limpiar imágenes corruptas**
	* [Limpieza Imágenes vacías](http://www.dropwizard.io/1.0.2/docs/) - Archivo Jupyter Notebook
		* Imágenes descargadas que no tienen peso alguno.
4. **Limpieza de data**
	* Es necesario si no se tiene un dataset limpio.
	* Si es descargado de Imagenet o algún repositorio de imágenes validado y el dataset es limpio, de no ser así como Google Images deben de eliminarse las imágenes que no corresponde con la etiqueta o clase.
	* [Limpieza de data](http://www.dropwizard.io/1.0.2/docs/) - Descripción
5. **Split de Dataset**
	* Se realizan dos splits, 
	* El 75% para el entrenamiento.
		* Para entrenamiento del modelo 75%
		* Para validación del modelo 25%
	* El 25% para prueba.
	* [Split de Dataset](http://www.dropwizard.io/1.0.2/docs/) - Descripción
6. **Entrenamiento Modelo**
	* Se ha entrenado dos tipos de modelos
		* Un modelo único entrenado con 10 etiquetas o clases.
			* [Entrenamiento Modelo 10 Etiquetas](http://www.dropwizard.io/1.0.2/docs/) - Archivo Jupyter Notebook
			* Cantidad promedio de imágenes por clase: **824**
			* El dataset es obtenido de Imagenet y la cantidad de imágenes por etiqueta es la siguiente:

			Etiqueta|Número de imágenes
			------|------ 
			Ave|1308
			Gato|1044
			Perro|824
			Pez|742
			Flor|751
			Comida|1067
			Persona|844
			Reptil|705
			Árbol|60
			Utensilio|894
		* Un modelo multicapa, de varias capas o niveles:
			![No se cargó Modelo Multicapa](https://drive.google.com/uc?export=view&id=1OB5BB1tYygI5rHrnAeOf6BBw-lxX_oIw "Modelo Multicapa")
			* [Entrenamiento Modelo Multicapa](http://www.dropwizard.io/1.0.2/docs/) - Carpeta de archivos Jupyter Notebook
			* En este modelo se tiene 78 etiquetas o clases
			* Este modelo consiste en entrenar múltiples modelos, ya que cuando se tienen menor número de clases por etiqueta (en este caso **133** en promedio por clase), si fuera evaluado con un solo modelo tendría muy baja eficiencia, ya que las CNN necesitan una gran cantidad de muestras por clase para una correcta clasificación o predicción.
			* La imagen muestra cómo funciona el modelo multicapa, en este caso solo hasta el nivel 3, hay 10 modelos.
			![No se cargó Modelo Multicapa - Nivel 3](https://drive.google.com/uc?export=view&id=11fGewFUtEG-3VszPrUNi8uBSnbTvjoSv "Modelo Multicapa - Nivel 3")

			Número|Etiquetas|Número de imágenes|75%|25%|Etiquetas Nivel 2|Etiquetas Nivel 3
			------|------|------|------|------|------|------
			1|Agua Mineral_Cielo|144|108|36|Cielo|
			2|Agua Mineral_Otros|227|170|57|Otros|
			3|Agua Mineral_San Carlos|89|67|22|San Carlos|
			4|Agua Mineral_San Luis|144|108|36|San Luis|
			5|Agua Mineral_San Mateo|119|89|30|San Mateo|
			6|Café_Altomayo|76|57|19|Altomayo|
			7|Café_Cafetal|71|53|18|Cafetal|
			8|Café_Kirma|69|52|17|Kirma|
			9|Café_Nescafé|69|52|17|Nescafé|
			10|Café_Otros|223|167|56|Otros|
			11|Gaseosa_7up|71|53|18|7up|
			12|Gaseosa_Coca Cola|217|163|54|Coca Cola|
			13|Gaseosa_Crush|66|50|17|Crush|
			14|Gaseosa_Fanta|78|59|20|Fanta|
			15|Gaseosa_Guaraná|81|61|20|Guaraná|
			16|Gaseosa_Inka Kola|153|115|38|Inka Kola|
			17|Gaseosa_KR|70|53|18|KR|
			18|Gaseosa_Otros|156|117|39|Otros|
			19|Gaseosa_Pepsi|138|104|35|Pepsi|
			20|Gaseosa_Sprite|130|98|33|Sprite|
			21|Jabón_Asepxia|222|167|56|Asepxia|
			22|Jabón_Bolivar|80|60|20|Bolivar|
			23|Jabón_Camay|219|164|55|Camay|
			24|Jabón_Dove|295|221|74|Dove|
			25|Jabón_Johnson|237|178|59|Johnson|
			26|Jabón_Lux|295|221|74|Lux|
			27|Jabón_Neko|75|56|19|Neko|
			28|Jabón_Nivea|151|113|38|Nivea|
			29|Jabón_Otros|129|97|32|Otros|
			30|Jabón_Palmolive_Delicada Exfoliación|137|103|34|Palmolive_Delicada Exfoliación|Delicada Exfoliación
			31|Jabón_Palmolive_Otros|219|164|55|Palmolive_Otros|Otros
			32|Jabón_Palmolive_Sensación Humectante|151|113|38|Palmolive_Sensación Humectante|Sensación Humectante
			33|Jabón_Palmolive_Suavidad Relajante|128|96|32|Palmolive_Suavidad Relajante|Suavidad Relajante
			34|Jabón_Protex|146|110|37|Protex|
			35|Otros_Árboles|157|118|39|Árboles|
			36|Otros_Cama|175|131|44|Cama|
			37|Otros_Cepillo|75|56|19|Cepillo|
			38|Otros_Cubiertos|165|124|41|Cubiertos|
			39|Otros_Fondos de Colores|139|104|35|Fondos de Colores|
			40|Otros_Fondos de lugares dentro de casa|142|107|36|Fondos de lugares dentro de casa|
			41|Otros_Gato|78|59|20|Gato|
			42|Otros_Lapicero|83|62|21|Lapicero|
			43|Otros_Laptop|55|41|14|Laptop|
			44|Otros_Lata|78|59|20|Lata|
			45|Otros_Mano|75|56|19|Mano|
			46|Otros_Mesa|84|63|21|Mesa|
			47|Otros_Mouse|62|47|16|Mouse|
			48|Otros_Olla|81|61|20|Olla|
			49|Otros_Otros animales|167|125|42|Otros animales|
			50|Otros_Otros objetos|157|118|39|Otros objetos|
			51|Otros_Pasto|77|58|19|Pasto|
			52|Otros_Perro|29|22|7|Perro|
			53|Otros_Persona|106|80|27|Persona|
			54|Otros_Silla|163|122|41|Silla|
			55|Otros_Sillón|157|118|39|Sillón|
			56|Otros_Taza|85|64|21|Taza|
			57|Otros_Vaso|89|67|22|Vaso|
			58|Otros_Vegetación|95|71|24|Vegetación|
			59|Pasta Dental_Aquafresh|143|107|36|Aquafresh|
			60|Pasta Dental_Colgate|208|156|52|Colgate|
			61|Pasta Dental_Kolynos|83|62|21|Kolynos|
			62|Pasta Dental_Oral B|169|127|42|Oral B|
			63|Pasta Dental_Otros|161|121|40|Otros|
			64|Pasta Dental_Sensodyne|166|125|42|Sensodyne|
			65|Shampoo_Dove|184|138|46|Dove|
			66|Shampoo_Elvive|132|99|33|Elvive|
			67|Shampoo_H&S|197|148|49|H&S|
			68|Shampoo_Herbal Essences|112|84|28|Herbal Essences|
			69|Shampoo_Johnson|119|89|30|Johnson|
			70|Shampoo_Otros|189|142|47|Otros|
			71|Shampoo_Palmolive|224|168|56|Palmolive|
			72|Shampoo_Pantene_Fuerza y Reconstrucción|129|97|32|Pantene_Fuerza y Reconstrucción|Fuerza y Reconstrucción
			73|Shampoo_Pantene_Hidratación Extrema|75|56|19|Pantene_Hidratación Extrema|Hidratación Extrema
			74|Shampoo_Pantene_Otro|145|109|36|Pantene_Otro|Otro
			75|Shampoo_Pantene_Restauración|135|101|34|Pantene_Restauración|Restauración
			76|Shampoo_Pert Plus|145|109|36|Pert Plus|
			77|Shampoo_Savital|135|101|34|Savital|
			78|Shampoo_Sedal|143|107|36|Sedal|
			|**TOTAL**|**10443**|**7838**|**2618**||
			|**PROMEDIO**|**134**||||
7. **Prueba del Modelo**
	* [Prueba del Modelo Básico](http://www.dropwizard.io/1.0.2/docs/) - Archivo Jupyter Notebook
		* Obtener matriz de confusión.
		* Métrica de eficiencia: Average F1 Score - Mean Average Precision
8. **Evaluar modelo estático imagen**
	* [Prueba del Modelo Básico](http://www.dropwizard.io/1.0.2/docs/) - Archivo Jupyter Notebook
		* Obtener matriz de confusión.
	* [Prueba del Modelo Multicapa](http://www.dropwizard.io/1.0.2/docs/) - Archivo Jupyter Notebook
		* Obtener matriz de confusión.
9. **Evaluar modelo tiempo real opencv**
	* [Evaluar con Cámara - Modelo Básico](http://www.dropwizard.io/1.0.2/docs/) - Archivo Jupyter Notebook
10. **Optimizar modelo inferencia**
	* [Optimizar modelo para Inferencia](http://www.dropwizard.io/1.0.2/docs/) - Archivo Jupyter Notebook
11. **Aplicativo Android**
	* Se usa una plantilla de Tensorflow para probar el modelo en un teléfono inteligente.
	* [Aplicativo Android Modelo Básico](http://www.dropwizard.io/1.0.2/docs/) - Proyecto y APK
	* [Aplicativo Android Modelo Multicapa - Primer Nivel](http://www.dropwizard.io/1.0.2/docs/) - Proyecto y APK
	* [Plantilla tensorflow](http://www.dropwizard.io/1.0.2/docs/)

## Pruebas ⚙️


### Pruebas Unitarias - Modelo Básico 🔩

![No se cargó Ave](https://drive.google.com/uc?export=view&id=1gWeztvnwjmg84mBWyYFt2_mZKgvFNFPi "Ave")|![No se cargó Comida](https://drive.google.com/uc?export=view&id=1bdqy3jcfa8-NT4NAbAxzu_YKzO4Ku4Oe "Comida")|![No se cargó Flor](https://drive.google.com/uc?export=view&id=1DzET_2bvadkqn3a4ZiInp5tXjOgFJyTT "Flor")
------|------|------
Ave (score=0.55168)|Comida (score=0.99987)|Flor (score=0.95210)
Reptil (score=0.39194)|Flor (score=0.00166)|Comida (score=0.11947)
Utensilio (score=0.10660)|Pez (score=0.00120)|Pez (score=0.01694)
Pez (score=0.06095)|Gato (score=0.00013)|Persona (score=0.00461)
Perro (score=0.05183)|Utensilio (score=0.00013)|Gato (score=0.00333)

![No se cargó Gato](https://drive.google.com/uc?export=view&id=1IrWjYp1s96aPuqoLHVCIOYj6DuOBbXro "Gato Solución")|![No se cargó Perro](https://drive.google.com/uc?export=view&id=1JkSwsFnrY1fbhnZiidYmuTvnaBSEpnLV "Perro")|![No se cargó Persona](https://drive.google.com/uc?export=view&id=1glrMN8vxyd8sCMyoVMrTOtfO3r_WeffC "Persona")
------|------|------
Gato (score=0.98599)|Perro (score=0.84811)|Persona (score=0.36379)
Perro (score=0.05299)|Gato (score=0.24164)|Pez (score=0.11069)
Utensilio (score=0.02251)|Ave (score=0.10057)|Perro (score=0.09442)
Pez (score=0.01433)|Persona (score=0.05738)|Reptil (score=0.06520)
Persona (score=0.00624)|Reptil (score=0.01254)|Ave (score=0.06465)


### Pruebas Unitarias - Modelo Multicapa 🔩 FALTA


|    Modelo Nivel 1                    |    Modelo Nivel 2                        |    Modelo Nivel 3                          |
|--------------------------------------|------------------------------------------|--------------------------------------------|
|    Jabon   (score=0.99999)           |    Palmolive (score=0.99189)             |    Sensacion Humectante (score=0.99904)    |
|    Shampoo   (score=0.00567)         |    Lux (score=0.00000)                   |    Suavidad Relajante (score=0.01113)      |
|    Pasta Dental   (score=0.00000)    |    Protex (score=0.00000)                |    Delicada Exfoliacion (score=0.00012)    |
|    Gaseosa (score=0.00000)           |    Otros   (score=0.00000)               |    Otros   (score=0.00000)                 |
|    Cafe (score=0.00000)              |    Camay (score=0.00000)                 |                                            |
|                                      | **Jabon Palmolive Sensacion Humectante** |                                            |
|                                      | ![No se cargó Jabon Palmolive Sensacion Humectante](https://drive.google.com/uc?export=view&id=1jnq3Xld2inGI11Jnv8vv1gveDfdvBmLM "Jabon Palmolive Sensacion Humectante") |                                            |


|    Modelo Nivel 1                    |    Modelo Nivel 2                       |    Modelo Nivel 3                             |
|--------------------------------------|-----------------------------------------|-----------------------------------------------|
|    Shampoo   (score=0.99975)         |    Pantene (score=1.00000)              |    Hidratacion Extrema (score=1.00000)        |
|    Pasta Dental   (score=0.00196)    |    Dove (score=0.00000)                 |    Otro (score=0.00000)                       |
|    Cafe   (score=0.00003)            |    Savital (score=0.00000)              |    Restauracion (score=0.00000)               |
|    Agua Mineral   (score=0.00002)    |    Sedal (score=0.00000)                |    Fuerza y Reconstruccion (score=0.00000)    |
|    Jabon   (score=0.00001)           |    Pert Plus (score=0.00000)            |                                               |
|                                      | **Shampoo Pantene Hidratacion Extrema** |                                               |
|                                      | ![No se cargó Shampoo Pantene Hidratacion Extrema](https://drive.google.com/uc?export=view&id=1ciCdRT-cKvdU0xblkv5v8rh07aBq4u9Q "Shampoo Pantene Hidratacion Extrema")|                                               |



|    Modelo Nivel 1                    |    Modelo Nivel 2               |    Modelo Nivel 3    |
|--------------------------------------|---------------------------------|----------------------|
|    Gaseosa   (score=1.00000)         |    Coca Cola (score=0.99896)    |                      |
|    Cafe   (score=0.00000)            |    KR (score=0.00036)           |                      |
|    Pasta Dental   (score=0.00000)    |    Otros (score=0.00001)        |                      |
|    Otros   (score=0.00000)           |    Sprite (score=0.00000)       |                      |
|    Agua Mineral   (score=0.00000)    |    Pepsi (score=0.00000)        |                      |
|                                      | **Gaseosa Coca Cola**           |                      |
|                                      | ![No se cargó Gaseosa Coca Cola](https://drive.google.com/uc?export=view&id=1JDK01nj81lSNLaGunkFrZjWsYbeSwyHD "Gaseosa Coca Cola")           |                      |
### Resultado Aplicativo - Modelo Básico 🔩

| ![No se cargó Gato](https://drive.google.com/uc?export=view&id=1qy8hF5N_VYKReMLDY_5U2VsOo5Y9Qoi3 "Gato Solución")|![No se cargó Perro](https://drive.google.com/uc?export=view&id=1wn9p4YEfxXCKVdlrxJvdx_20_nGE-I_S "Perro")|![No se cargó Persona](https://drive.google.com/uc?export=view&id=108q6fodNxItlesMBbi8qDZhBu0AnROTX "Persona") |
|:-:|---|---|

## Análisis de resultados ⌨️


### Matriz de Confusión Model Básico 🔩

|              |           |     |      |       | **Valor predicho** |      |        |         |        |       |           |
|:------------:|-----------|-----|------|-------|----------------|------|--------|---------|--------|-------|-----------|
|              |           | Ave | Gato | Perro | Pez            | Flor | Comida | Persona | Reptil | Árbol | Utensilio |
|              | Ave       | 281 | 4    | 5     | 4              | 4    | 3      | 5       | 7      | 0     | 7         |
|              | Gato      | 24  | 227  | 7     | 8              | 6    | 9      | 14      | 8      | 0     | 17        |
|              | Perro     | 28  | 6    | 226   | 8              | 8    | 3      | 14      | 14     | 0     | 13        |
|              | Pez       | 28  | 2    | 2     | 244            | 3    | 9      | 6       | 9      | 0     | 17        |
| **Valor   Real** | Flor      | 6   | 0    | 0     | 0              | 308  | 4      | 0       | 1      | 0     | 1         |
|              | Comida    | 0   | 0    | 0     | 4              | 7    | 308    | 0       | 0      | 0     | 1         |
|              | Persona   | 9   | 1    | 3     | 6              | 2    | 4      | 273     | 6      | 1     | 15        |
|              | Reptil    | 17  | 2    | 5     | 9              | 12   | 7      | 4       | 253    | 0     | 11        |
|              | Árbol     | 10  | 0    | 0     | 10             | 35   | 15     | 0       | 0      | 230   | 0         |
|              | Utensilio | 10  | 1    | 4     | 3              | 1    | 5      | 8       | 5      | 0     | 283       |



### Medidas de eficiencia Modelo Básico 🔩

|                            | **Precision** | **Recall** | **F1 Score** |
|:--------------------------:|---------------|------------|--------------|
| **Ave**                    | 0.680387      | 0.878125   | 0.766712     |
| **Gato**                   | 0.934156      | 0.709375   | 0.806394     |
| **Perro**                  | 0.896825      | 0.70625    | 0.79021      |
| **Pez**                    | 0.824324      | 0.7625     | 0.792208     |
| **Flor**                   | 0.797927      | 0.9625     | 0.872521     |
| **Comida**                 | 0.839237      | 0.9625     | 0.896652     |
| **Persona**                | 0.842593      | 0.853125   | 0.847826     |
| **Reptil**                 | 0.834984      | 0.790625   | 0.812199     |
| **Árbol**                  | 0.995671      | 0.766667   | 0.86629      |
| **Utensilio**              | 0.775342      | 0.884375   | 0.826277     |
| **Average F1 Score**       | -             | -          | **82.7729**  |
| **Mean Average Precision** | **84.21448**  | -          | -            |



## Autores ✒️


* **Freddy Dick Salazar Valverde** - *Alumno Tesista* - [salazarvalverdefd](https://github.com/salazarvalverdefd) - [DickSalazar](http://www.ssyspe.org/memberlist.php?mode=viewprofile&u=1747)
* **Wester Zela Moraya** - *Asesor de Tesis* - [wester.zela](http://www.ssyspe.org/memberlist.php?mode=viewprofile&u=54)



---
 Por [Freddy Dick Salazar Valverde](https://github.com/salazarvalverdefd) - [DickSalazar](http://www.ssyspe.org/memberlist.php?mode=viewprofile&u=1747)😊
