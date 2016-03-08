# NeuralNetwork_MNIST
Implementación en python de una red neuronal sencilla que clasifica el conjunto de imágenes MNIST

#Introducción:
Las redes neuronales son ya una técnica clásica bio-inspirada con multitud de aplicaciones,
generalmente centradas alrededor de la clasificación de patrones complejos de entrada. Existen
generalmente dos aproximaciones fundamentalmente diferentes a las redes neuronales artificiales
(ANN): las ANN supervisadas y no supervisadas (o auto-organizativas). La diferencia entre ambas
radica fundamentalmente en si podemos aportar la respuesta “correcta” que esperamos de la red o
no.

#Problema a resolver:
Vamos a partir de la colección de imágenes de dígitos manuscritos del NIST
(http://yann.lecun.com/exdb/mnist/). Esta colección consta de 60000 imágenes para entrenamiento y
20000 para test. Son imágenes en escala de grises (de 0 a 1), de 28x28 píxeles cada una. En el curso
online se ponen las imágenes, y un script de Matlab para cargarlas en memoria.
La codificación de cada imágen es una ristra de 784 reales, colocando fila tras fila. Es decir, los
primeros 28 números corresponden a la primera columna, los siguientes a las segunda y así hasta la
columna 28 que acaba con el dígito número 784.

La estructura de la red es:

![alt tag](https://github.com/BesayMontesdeoca/NeuralNetwork_MNIST/blob/master/model.png)

