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
	* Se realiza dos 
	* El 75% para el entrenamiento.
		* Para entrenamiento del modelo 75%
		* Para validación del modelo 25%
	* El 25% para prueba.
6. **Entrenamiento Modelo**
	* Se ha entrenado dos tipos de modelos
		* Un modelo único entrenado con 10 etiquetas o clases.
			* Cantidad promedio de imágenes por clase: **824**
			* El dataset es obtenido de Imagenet y la cantidad de imágenes por etiqueta es la siguiente:

Etiqueta	Número de imágenes
Ave	1308
Gato	1044
Perro	824
Pez	742
Flor	751
Comida	1067
Persona	844
Reptil	705
Árbol	60
Utensilio	894



		* Un modelo múltiple, de varias capas o niveles:
			* En este modelo se tiene 78 etiquetas o clases
			* Este modelo consiste en entrenar múltiples modelos, ya que cuando se tienen menor número de clases por etiqueta (en este caso **133** en promedio por clase), si fuera evaluado con un solo modelo tendría muy baja eficiencia, ya que las CNN necesitan una gran cantidad de muestras por clases para una correcta clasificación o predicción.
			* La imagen muestra como funciona el modelo multicapa, en este caso solo hasta el nivel 3.
7. **Prueba del Modelo**
	* [Prueba del modelo](http://www.dropwizard.io/1.0.2/docs/) - Archivo Jupyter Notebook
		* Obtener matriz de confusión.
		* Métrica de eficiencia: Average F1 Score - Mean Average Precision
8. **Evaluar modelo estático imagen**
	* [Prueba del modelo](http://www.dropwizard.io/1.0.2/docs/) - Archivo Jupyter Notebook
		* Obtener matriz de confusión.
	* [Prueba del modelo](http://www.dropwizard.io/1.0.2/docs/) - Archivo Jupyter Notebook
		* Obtener matriz de confusión.
9. **Evaluar modelo tiempo real opencv**
	* 
10. **Optimizar modelo inferencia**
	* 


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
