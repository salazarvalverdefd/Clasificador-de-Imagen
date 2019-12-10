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
			 ------ | --------- 
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
			Etiquetas|------|------|------|------|------
			Etiquetas|144|108|36|Cielo|
			Etiquetas|227|170.25|56.75|Otros|
			Etiquetas|89|66.75|22.25|San Carlos|
			Etiquetas|144|108|36|San Luis|
			Etiquetas|119|89.25|29.75|San Mateo|
			Etiquetas|76|57|19|Altomayo|
			Etiquetas|71|53.25|17.75|Cafetal|
			Etiquetas|69|51.75|17.25|Kirma|
			Etiquetas|69|51.75|17.25|Nescafé|
			Etiquetas|223|167.25|55.75|Otros|
			Etiquetas|71|53.25|17.75|7up|
			Etiquetas|217|162.75|54.25|Coca Cola|
			Etiquetas|66|49.5|16.5|Crush|
			Etiquetas|78|58.5|19.5|Fanta|
			Etiquetas|81|60.75|20.25|Guaraná|
			Etiquetas|153|114.75|38.25|Inka Kola|
			Etiquetas|70|52.5|17.5|KR|
			Etiquetas|156|117|39|Otros|
			Etiquetas|138|103.5|34.5|Pepsi|
			Etiquetas|130|97.5|32.5|Sprite|
			Etiquetas|222|166.5|55.5|Asepxia|
			Etiquetas|80|60|20|Bolivar|
			Etiquetas|219|164.25|54.75|Camay|
			Etiquetas|295|221.25|73.75|Dove|
			Etiquetas|237|177.75|59.25|Johnson|
			Etiquetas|295|221.25|73.75|Lux|
			Etiquetas|75|56.25|18.75|Neko|
			Etiquetas|151|113.25|37.75|Nivea|
			Etiquetas|129|96.75|32.25|Otros|
			Etiquetas|137|102.75|34.25|Palmolive_Delicada Exfoliación|Delicada Exfoliación
			Etiquetas|219|164.25|54.75|Palmolive_Otros|Otros
			Etiquetas|151|113.25|37.75|Palmolive_Sensación Humectante|Sensación Humectante
			Etiquetas|128|96|32|Palmolive_Suavidad Relajante|Suavidad Relajante
			Etiquetas|146|109.5|36.5|Protex|
			Etiquetas|157|117.75|39.25|Árboles|
			Etiquetas|175|131.25|43.75|Cama|
			Etiquetas|75|56.25|18.75|Cepillo|
			Etiquetas|165|123.75|41.25|Cubiertos|
			Etiquetas|139|104.25|34.75|Fondos de Colores|
			Etiquetas|142|106.5|35.5|Fondos de lugares dentro de casa|
			Etiquetas|78|58.5|19.5|Gato|
			Etiquetas|83|62.25|20.75|Lapicero|
			Etiquetas|55|41.25|13.75|Laptop|
			Etiquetas|78|58.5|19.5|Lata|
			Etiquetas|75|56.25|18.75|Mano|
			Etiquetas|84|63|21|Mesa|
			Etiquetas|62|46.5|15.5|Mouse|
			Etiquetas|81|60.75|20.25|Olla|
			Etiquetas|167|125.25|41.75|Otros animales|
			Etiquetas|157|117.75|39.25|Otros objetos|
			Etiquetas|77|57.75|19.25|Pasto|
			Etiquetas|29|21.75|7.25|Perro|
			Etiquetas|106|79.5|26.5|Persona|
			Etiquetas|163|122.25|40.75|Silla|
			Etiquetas|157|117.75|39.25|Sillón|
			Etiquetas|85|63.75|21.25|Taza|
			Etiquetas|89|66.75|22.25|Vaso|
			Etiquetas|95|71.25|23.75|Vegetación|
			Etiquetas|143|107.25|35.75|Aquafresh|
			Etiquetas|208|156|52|Colgate|
			Etiquetas|83|62.25|20.75|Kolynos|
			Etiquetas|169|126.75|42.25|Oral B|
			Etiquetas|161|120.75|40.25|Otros|
			Etiquetas|166|124.5|41.5|Sensodyne|
			Etiquetas|184|138|46|Dove|
			Etiquetas|132|99|33|Elvive|
			Etiquetas|197|147.75|49.25|H&S|
			Etiquetas|112|84|28|Herbal Essences|
			Etiquetas|119|89.25|29.75|Johnson|
			Etiquetas|189|141.75|47.25|Otros|
			Etiquetas|224|168|56|Palmolive|
			Etiquetas|129|96.75|32.25|Pantene_Fuerza y Reconstrucción|Fuerza y Reconstrucción
			Etiquetas|75|56.25|18.75|Pantene_Hidratación Extrema|Hidratación Extrema
			Etiquetas|145|108.75|36.25|Pantene_Otro|Otro
			Etiquetas|135|101.25|33.75|Pantene_Restauración|Restauración
			Etiquetas|145|108.75|36.25|Pert Plus|
			Etiquetas|135|101.25|33.75|Savital|
			Etiquetas|143|107.25|35.75|Sedal|
			Etiquetas|10443|7832.25|2610.75||

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
