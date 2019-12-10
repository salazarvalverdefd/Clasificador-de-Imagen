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
5. **Split de Dataset**
	* Se realizan dos splits, 
	* El 75% para el entrenamiento.
		* Para entrenamiento del modelo 75%
		* Para validación del modelo 25%
	* El 25% para prueba.
6. **Entrenamiento Modelo**
	* Se ha entrenado dos tipos de modelos
		* Un modelo único entrenado con 10 etiquetas o clases.
			* Cantidad promedio de imágenes por clase: **824**
			* El dataset es obtenido de Imagenet y la cantidad de imágenes por etiqueta es la siguiente:
			Etiqueta |	Número de imágenes
			 ------ | ------ 
			Ave |	1308
			Gato |	1044
			Perro |	824
			Pez |	742
			Flor |	751
			Comida |	1067
			Persona |	844
			Reptil |	705
			Árbol |	60
			Utensilio |	894
		* Un modelo múltiple, de varias capas o niveles:
			![No se cargó Modelo Multicapa](https://drive.google.com/uc?export=view&id=1-zxpuenttT6lMkXtCKriVdkxYdW9KkOn "Modelo Multicapa")
			* En este modelo se tiene 78 etiquetas o clases
			* Este modelo consiste en entrenar múltiples modelos, ya que cuando se tienen menor número de clases por etiqueta (en este caso **133** en promedio por clase), si fuera evaluado con un solo modelo tendría muy baja eficiencia, ya que las CNN necesitan una gran cantidad de muestras por clase para una correcta clasificación o predicción.
			* La imagen muestra como funciona el modelo multicapa, en este caso solo hasta el nivel 3, hay 10 modelos.
			![No se cargó Modelo Multicapa - Nivel 3](https://drive.google.com/uc?export=view&id=1-zxpuenttT6lMkXtCKriVdkxYdW9KkOn "Modelo Multicapa - Nivel 3")
			Etiquetas|Número de imágenes|75%|25%|Etiquetas Nivel 2|Etiquetas Nivel 3
			------|------|------|------|------|------
			Agua Mineral_Cielo|144|108|36|Cielo|
			Agua Mineral_Otros|227|170.25|56.75|Otros|
			Agua Mineral_San Carlos|89|66.75|22.25|San Carlos|
			Agua Mineral_San Luis|144|108|36|San Luis|
			Agua Mineral_San Mateo|119|89.25|29.75|San Mateo|
			Café_Altomayo|76|57|19|Altomayo|
			Café_Cafetal|71|53.25|17.75|Cafetal|
			Café_Kirma|69|51.75|17.25|Kirma|
			Café_Nescafé|69|51.75|17.25|Nescafé|
			Café_Otros|223|167.25|55.75|Otros|
			Gaseosa_7up|71|53.25|17.75|7up|
			Gaseosa_Coca Cola|217|162.75|54.25|Coca Cola|
			Gaseosa_Crush|66|49.5|16.5|Crush|
			Gaseosa_Fanta|78|58.5|19.5|Fanta|
			Gaseosa_Guaraná|81|60.75|20.25|Guaraná|
			Gaseosa_Inka Kola|153|114.75|38.25|Inka Kola|
			Gaseosa_KR|70|52.5|17.5|KR|
			Gaseosa_Otros|156|117|39|Otros|
			Gaseosa_Pepsi|138|103.5|34.5|Pepsi|
			Gaseosa_Sprite|130|97.5|32.5|Sprite|
			Jabón_Asepxia|222|166.5|55.5|Asepxia|
			Jabón_Bolivar|80|60|20|Bolivar|
			Jabón_Camay|219|164.25|54.75|Camay|
			Jabón_Dove|295|221.25|73.75|Dove|
			Jabón_Johnson|237|177.75|59.25|Johnson|
			Jabón_Lux|295|221.25|73.75|Lux|
			Jabón_Neko|75|56.25|18.75|Neko|
			Jabón_Nivea|151|113.25|37.75|Nivea|
			Jabón_Otros|129|96.75|32.25|Otros|
			Jabón_Palmolive_Delicada Exfoliación|137|102.75|34.25|Palmolive_Delicada Exfoliación|Delicada Exfoliación
			Jabón_Palmolive_Otros|219|164.25|54.75|Palmolive_Otros|Otros
			Jabón_Palmolive_Sensación Humectante|151|113.25|37.75|Palmolive_Sensación Humectante|Sensación Humectante
			Jabón_Palmolive_Suavidad Relajante|128|96|32|Palmolive_Suavidad Relajante|Suavidad Relajante
			Jabón_Protex|146|109.5|36.5|Protex|
			Otros_Árboles|157|117.75|39.25|Árboles|
			Otros_Cama|175|131.25|43.75|Cama|
			Otros_Cepillo|75|56.25|18.75|Cepillo|
			Otros_Cubiertos|165|123.75|41.25|Cubiertos|
			Otros_Fondos de Colores|139|104.25|34.75|Fondos de Colores|
			Otros_Fondos de lugares dentro de casa|142|106.5|35.5|Fondos de lugares dentro de casa|
			Otros_Gato|78|58.5|19.5|Gato|
			Otros_Lapicero|83|62.25|20.75|Lapicero|
			Otros_Laptop|55|41.25|13.75|Laptop|
			Otros_Lata|78|58.5|19.5|Lata|
			Otros_Mano|75|56.25|18.75|Mano|
			Otros_Mesa|84|63|21|Mesa|
			Otros_Mouse|62|46.5|15.5|Mouse|
			Otros_Olla|81|60.75|20.25|Olla|
			Otros_Otros animales|167|125.25|41.75|Otros animales|
			Otros_Otros objetos|157|117.75|39.25|Otros objetos|
			Otros_Pasto|77|57.75|19.25|Pasto|
			Otros_Perro|29|21.75|7.25|Perro|
			Otros_Persona|106|79.5|26.5|Persona|
			Otros_Silla|163|122.25|40.75|Silla|
			Otros_Sillón|157|117.75|39.25|Sillón|
			Otros_Taza|85|63.75|21.25|Taza|
			Otros_Vaso|89|66.75|22.25|Vaso|
			Otros_Vegetación|95|71.25|23.75|Vegetación|
			Pasta Dental_Aquafresh|143|107.25|35.75|Aquafresh|
			Pasta Dental_Colgate|208|156|52|Colgate|
			Pasta Dental_Kolynos|83|62.25|20.75|Kolynos|
			Pasta Dental_Oral B|169|126.75|42.25|Oral B|
			Pasta Dental_Otros|161|120.75|40.25|Otros|
			Pasta Dental_Sensodyne|166|124.5|41.5|Sensodyne|
			Shampoo_Dove|184|138|46|Dove|
			Shampoo_Elvive|132|99|33|Elvive|
			Shampoo_H&S|197|147.75|49.25|H&S|
			Shampoo_Herbal Essences|112|84|28|Herbal Essences|
			Shampoo_Johnson|119|89.25|29.75|Johnson|
			Shampoo_Otros|189|141.75|47.25|Otros|
			Shampoo_Palmolive|224|168|56|Palmolive|
			Shampoo_Pantene_Fuerza y Reconstrucción|129|96.75|32.25|Pantene_Fuerza y Reconstrucción|Fuerza y Reconstrucción
			Shampoo_Pantene_Hidratación Extrema|75|56.25|18.75|Pantene_Hidratación Extrema|Hidratación Extrema
			Shampoo_Pantene_Otro|145|108.75|36.25|Pantene_Otro|Otro
			Shampoo_Pantene_Restauración|135|101.25|33.75|Pantene_Restauración|Restauración
			Shampoo_Pert Plus|145|108.75|36.25|Pert Plus|
			Shampoo_Savital|135|101.25|33.75|Savital|
			Shampoo_Sedal|143|107.25|35.75|Sedal|
			TOTAL|10443|7832.25|2610.75||
			PROMEDIO|133.884615384615||||


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
