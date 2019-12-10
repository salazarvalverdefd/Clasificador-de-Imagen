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
			![No se cargó Modelo Multicapa](https://drive.google.com/uc?export=view&id=1-zxpuenttT6lMkXtCKriVdkxYdW9KkOn "Modelo Multicapa")
			* [Entrenamiento Modelo Multicapa](http://www.dropwizard.io/1.0.2/docs/) - Carpeta de archivos Jupyter Notebook
			* En este modelo se tiene 78 etiquetas o clases
			* Este modelo consiste en entrenar múltiples modelos, ya que cuando se tienen menor número de clases por etiqueta (en este caso **133** en promedio por clase), si fuera evaluado con un solo modelo tendría muy baja eficiencia, ya que las CNN necesitan una gran cantidad de muestras por clase para una correcta clasificación o predicción.
			* La imagen muestra como funciona el modelo multicapa, en este caso solo hasta el nivel 3, hay 10 modelos.
			![No se cargó Modelo Multicapa - Nivel 3](https://drive.google.com/uc?export=view&id=1-zxpuenttT6lMkXtCKriVdkxYdW9KkOn "Modelo Multicapa - Nivel 3")
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


## Ejecutando las pruebas ⚙️

_Explica como ejecutar las pruebas automatizadas para este sistema_

### Analice las pruebas end-to-end 🔩

_Explica que verifican estas pruebas y por qué_

```
Da un ejemplo
```

### Y las pruebas de estilo de codificación ⌨️

_Explica que verifican estas pruebas y por qué_

```
Da un ejemplo
```

## Despliegue 📦

_Agrega notas adicionales sobre como hacer deploy_

## Construido con 🛠️

_Menciona las herramientas que utilizaste para crear tu proyecto_

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - El framework web usado
* [Maven](https://maven.apache.org/) - Manejador de dependencias
* [ROME](https://rometools.github.io/rome/) - Usado para generar RSS

## Contribuyendo 🖇️

Por favor lee el [CONTRIBUTING.md](https://gist.github.com/villanuevand/xxxxxx) para detalles de nuestro código de conducta, y el proceso para enviarnos pull requests.

## Wiki 📖

Puedes encontrar mucho más de cómo utilizar este proyecto en nuestra [Wiki](https://github.com/tu/proyecto/wiki)

## Versionado 📌

Usamos [SemVer](http://semver.org/) para el versionado. Para todas las versiones disponibles, mira los [tags en este repositorio](https://github.com/tu/proyecto/tags).

## Autores ✒️


* **Freddy Dick Salazar Valverde** - *Alumno Tesista* - [villanuevand](https://github.com/villanuevand)
* **Wester Zela Moraya** - *Asesor de Tesis* - [fulanitodetal](#fulanito-de-tal)

También puedes mirar la lista de todos los [contribuyentes](https://github.com/your/project/contributors) quíenes han participado en este proyecto. 

## Licencia 📄

Este proyecto está bajo la Licencia (Tu Licencia) - mira el archivo [LICENSE.md](LICENSE.md) para detalles

## Expresiones de Gratitud 🎁

* Comenta a otros sobre este proyecto 📢
* Invita una cerveza 🍺 o un café ☕ a alguien del equipo. 
* Da las gracias públicamente 🤓.
* etc.



---
⌨️ con ❤️ por [Villanuevand](https://github.com/Villanuevand) 😊
