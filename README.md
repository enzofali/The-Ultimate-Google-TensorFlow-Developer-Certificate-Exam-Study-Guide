# The Ultimate Google TensorFlow Developer Certificate Exam Study Guide

The purpose of this repository is to detail the necessary resources to pass the TensorFlow Developer Certificate in the shortest time possible, minimizing the effort required by centralizing relevant information. You will find study material, course references, and notebooks examples. Keep in mind that the specified resources aren't free, so by following this guide, you must have a minimum budget. Let's begin!

## Honorable Mentions

Consider the [Tensorflow Certification Study Guide](https://github.com/nicholasjhana/tensorflow-certification-study-guide#tensorflow-certification-study-guide) laid the foundation for this repository. That repo is excellent and includes more information and available material.

## Required Knowledge

### Courses
Before moving on to Deep Learning, there are some basic concepts you should manage with ease. First, you need to achieve the minimum understanding of Mathematics (Multivariable Calculus, Linear Algebra, and Probability and Statistics). On the other hand, you need to have strong Python programming skills. Below are the recommended materials to achieve the necessary knowledge in these areas.

Math Courses:
- [Mathematics for Machine Learning Specialization](https://www.coursera.org/specializations/mathematics-machine-learning#courses)
- [Introduction to Statistics](https://www.coursera.org/learn/stanford-statistics)

Programing with Python Courses:
- [Python for Everybody Specialization](https://www.coursera.org/specializations/python#courses)
- [Python 3 Programming Specialization](https://www.coursera.org/specializations/python-3-programming#courses)
- [CS50’s Introduction to Programming with Python](https://cs50.harvard.edu/python/2022/psets/0/) (free resource)

Now that you have achieved the minimum knowledge requirements, let's jump to the 🌶️ things. 

First of all, even though it is not necessary to know about Machine Learning to obtain the Tensorflow Developer Certificate (only Deep Learning is required), the following course turns out to be optional on this path. Nevertheless, it is an excellent specialization to start with, and I strongly recommend doing it.

1. [Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction) (opcional)

Now, the following specializations are *essential* to obtain the certification. The Deep Learning Specialization and The Tensorflow Developer Specialization are indeed complementary. The Deep Learning Specialization develops all the necessary concepts by adopting a more theoretical approach. While the Tensorflow Developer Specialization adopts a more practical and applied approach, applying everything learned in the previous specialization and putting hands into practice.

2. [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning)

3. [Coursera Tensorflow Developer Certificate](https://www.coursera.org/professional-certificates/tensorflow-in-practice) 

#### Deeper Dive into Tensorflow Developer Certificate Specialization

We will be particularly interested in further developing the material given in this specialization. That is because the Google TensorFlow Developer Certification exam forms its bases on this specialization, so completing it is crucial. All the material listed below is available in the Deep Learning AI public repo: [DeepLearning.AI TensorFlow Developer Public Repo](https://github.com/https-deeplearning-ai/tensorflow-1-public). Below you can see all the notebooks given in the courses, excluding their weekly assignments, which are a good practice resource for exam preparation.

Finally, the ⭐ indicates the highly recommended notebooks to review before taking the exam. Try to focus on implementing and writing the models on your own, tuning the hyperparameters, and enhancing the model predictions with the corresponding datasets, like manually defining the learning rate decay. These notebooks are a faithful representation of how the exam exercises are developed, so it's important to achieve good results in their corresponding metrics.

**[Introduction to TensorFlow for Artificial Intelligence, Machine Learning, and Deep Learning](https://www.coursera.org/learn/introduction-tensorflow?specialization=tensorflow-in-practice)**

- [The Hello World of Deep Learning with Neural Networks](https://github.com/https-deeplearning-ai/tensorflow-1-public/blob/main/C1/W1/ungraded_lab/C1_W1_Lab_1_hello_world_nn.ipynb) ⭐
- [Beyond Hello World, A Computer Vision Example](https://github.com/https-deeplearning-ai/tensorflow-1-public/blob/main/C1/W2/ungraded_labs/C1_W2_Lab_1_beyond_hello_world.ipynb) ⭐
- [Using Callbacks to Control Training](https://github.com/https-deeplearning-ai/tensorflow-1-public/blob/main/C1/W2/ungraded_labs/C1_W2_Lab_2_callbacks.ipynb)
- [Improving Computer Vision Accuracy using Convolutions](https://github.com/https-deeplearning-ai/tensorflow-1-public/blob/main/C1/W3/ungraded_labs/C1_W3_Lab_1_improving_accuracy_using_convolutions.ipynb)
- [Exploring Convolutions](https://github.com/https-deeplearning-ai/tensorflow-1-public/blob/main/C1/W3/ungraded_labs/C1_W3_Lab_2_exploring_convolutions.ipynb)
- [Training with ImageDataGenerator](https://github.com/https-deeplearning-ai/tensorflow-1-public/blob/main/C1/W4/ungraded_labs/C1_W4_Lab_1_image_generator_no_validation.ipynb)
- [ImageDataGenerator with a Validation Set](https://github.com/https-deeplearning-ai/tensorflow-1-public/blob/main/C1/W4/ungraded_labs/C1_W4_Lab_2_image_generator_with_validation.ipynb) 
- [Effect of Compacted Images in Training](https://github.com/https-deeplearning-ai/tensorflow-1-public/blob/main/C1/W4/ungraded_labs/C1_W4_Lab_3_compacted_images.ipynb)

**[Convolutional Neural Networks in TensorFlow](https://www.coursera.org/learn/convolutional-neural-networks-tensorflow?specialization=tensorflow-in-practice)**

- [Using more sophisticated images with Convolutional Neural Networks](https://github.com/https-deeplearning-ai/tensorflow-1-public/blob/main/C2/W1/ungraded_lab/C2_W1_Lab_1_cats_vs_dogs.ipynb)
- [Data Augmentation](https://github.com/https-deeplearning-ai/tensorflow-1-public/blob/main/C2/W2/ungraded_labs/C2_W2_Lab_1_cats_v_dogs_augmentation.ipynb)
- [Data Augmentation on the Horses or Humans Dataset](https://github.com/https-deeplearning-ai/tensorflow-1-public/blob/main/C2/W2/ungraded_labs/C2_W2_Lab_2_horses_v_humans_augmentation.ipynb) ⭐
- [Transfer Learning](https://github.com/https-deeplearning-ai/tensorflow-1-public/blob/main/C2/W3/ungraded_lab/C2_W3_Lab_1_transfer_learning.ipynb)
- [Multi-class Classifier](https://github.com/https-deeplearning-ai/tensorflow-1-public/blob/main/C2/W4/ungraded_lab/C2_W4_Lab_1_multi_class_classifier.ipynb) ⭐

**[Natural Language Processing in TensorFlow](https://www.coursera.org/learn/natural-language-processing-tensorflow?specialization=tensorflow-in-practice)**

- [Tokenizer Basics](https://github.com/https-deeplearning-ai/tensorflow-1-public/blob/main/C3/W1/ungraded_labs/C3_W1_Lab_1_tokenize_basic.ipynb)
- [Generating Sequences and Padding](https://github.com/https-deeplearning-ai/tensorflow-1-public/blob/main/C3/W1/ungraded_labs/C3_W1_Lab_2_sequences_basic.ipynb)
- [Tokenizing the Sarcasm Dataset](https://github.com/https-deeplearning-ai/tensorflow-1-public/blob/main/C3/W1/ungraded_labs/C3_W1_Lab_3_sarcasm.ipynb)
- [Training a binary classifier with the IMDB Reviews Dataset](https://github.com/https-deeplearning-ai/tensorflow-1-public/blob/main/C3/W2/ungraded_labs/C3_W2_Lab_1_imdb.ipynb)
- [Training a binary classifier with the Sarcasm Dataset](https://github.com/https-deeplearning-ai/tensorflow-1-public/blob/main/C3/W2/ungraded_labs/C3_W2_Lab_2_sarcasm_classifier.ipynb)
- [Subword Tokenization with the IMDB Reviews Dataset](https://github.com/https-deeplearning-ai/tensorflow-1-public/blob/main/C3/W2/ungraded_labs/C3_W2_Lab_3_imdb_subwords.ipynb)
- [Single Layer LSTM](https://github.com/https-deeplearning-ai/tensorflow-1-public/blob/main/C3/W3/ungraded_labs/C3_W3_Lab_1_single_layer_LSTM.ipynb)
- [Multiple LSTMs](https://github.com/https-deeplearning-ai/tensorflow-1-public/blob/main/C3/W3/ungraded_labs/C3_W3_Lab_2_multiple_layer_LSTM.ipynb)
- [Using Convolutional Neural Networks](https://github.com/https-deeplearning-ai/tensorflow-1-public/blob/main/C3/W3/ungraded_labs/C3_W3_Lab_3_Conv1D.ipynb)
- [Building Models for the IMDB Reviews Dataset](https://github.com/https-deeplearning-ai/tensorflow-1-public/blob/main/C3/W3/ungraded_labs/C3_W3_Lab_4_imdb_reviews_with_GRU_LSTM_Conv1D.ipynb)
- [Training a Sarcasm Detection Model using Bidirectional LSTMs](https://github.com/https-deeplearning-ai/tensorflow-1-public/blob/main/C3/W3/ungraded_labs/C3_W3_Lab_5_sarcasm_with_bi_LSTM.ipynb) ⭐
- [Training a Sarcasm Detection Model using a Convolution Layer](https://github.com/https-deeplearning-ai/tensorflow-1-public/blob/main/C3/W3/ungraded_labs/C3_W3_Lab_6_sarcasm_with_1D_convolutional.ipynb) ⭐
- [Generating Text with Neural Networks](https://github.com/https-deeplearning-ai/tensorflow-1-public/blob/main/C3/W4/ungraded_labs/C3_W4_Lab_1.ipynb)
- [Generating Text from Irish Lyrics](https://github.com/https-deeplearning-ai/tensorflow-1-public/blob/main/C3/W4/ungraded_labs/C3_W4_Lab_2_irish_lyrics.ipynb)

**[Sequences, Time Series and Prediction](https://www.coursera.org/learn/tensorflow-sequences-time-series-and-prediction?specialization=tensorflow-in-practice)**

- [Introduction to Time Series Plots](https://github.com/https-deeplearning-ai/tensorflow-1-public/blob/main/C4/W1/ungraded_labs/C4_W1_Lab_1_time_series.ipynb)
- [Statistical Forecasting on Synthetic Data](https://github.com/https-deeplearning-ai/tensorflow-1-public/blob/main/C4/W1/ungraded_labs/C4_W1_Lab_2_forecasting.ipynb)
- [Preparing Time Series Features and Labels](https://github.com/https-deeplearning-ai/tensorflow-1-public/blob/main/C4/W2/ungraded_labs/C4_W2_Lab_1_features_and_labels.ipynb)
- [Training a Single Layer Neural Network with Time Series Data](https://github.com/https-deeplearning-ai/tensorflow-1-public/blob/main/C4/W2/ungraded_labs/C4_W2_Lab_2_single_layer_NN.ipynb)
- [Training a Deep Neural Network with Time Series Data](https://github.com/https-deeplearning-ai/tensorflow-1-public/blob/main/C4/W2/ungraded_labs/C4_W2_Lab_3_deep_NN.ipynb)
- [Using a Simple RNN for forecasting](https://github.com/https-deeplearning-ai/tensorflow-1-public/blob/main/C4/W3/ungraded_labs/C4_W3_Lab_1_RNN.ipynb)
- [Using a multi-layer LSTM for forecasting](https://github.com/https-deeplearning-ai/tensorflow-1-public/blob/main/C4/W3/ungraded_labs/C4_W3_Lab_2_LSTM.ipynb)
- [Using Convolutions with LSTMs](https://github.com/https-deeplearning-ai/tensorflow-1-public/blob/main/C4/W4/ungraded_labs/C4_W4_Lab_1_LSTM.ipynb)
- [Predicting Sunspots with Neural Networks (DNN only)](https://github.com/https-deeplearning-ai/tensorflow-1-public/blob/main/C4/W4/ungraded_labs/C4_W4_Lab_2_Sunspots_DNN.ipynb)
- [Predicting Sunspots with Neural Networks](https://github.com/https-deeplearning-ai/tensorflow-1-public/blob/main/C4/W4/ungraded_labs/C4_W4_Lab_3_Sunspots_CNN_RNN_DNN.ipynb) ⭐

## More Study Resourse

https://www.tensorflow.org/learn

https://www.tensorflow.org/overview

el curso no da nada de multivariate time serias y n steps forcasting, (guia de TF)

hice otro repo con los resourses

[My Material](https://github.com/Enzofali/TensorFlowDeveloperCertificateMaterial-)

### Convolutional Neural Network
- [Convolutional Neural Network for Binary Classification](https://github.com/Enzofali/TensorFlowDeveloperCertificateMaterial-/blob/main/Convolutional%20Neural%20Network/1_BinaryCNN.ipynb)
- [Convolutional Neural Network for Multi-class Classification](https://github.com/Enzofali/TensorFlowDeveloperCertificateMaterial-/blob/main/Convolutional%20Neural%20Network/2_MultiClassCNN.ipynb)
- [Transfer learning](https://github.com/Enzofali/TensorFlowDeveloperCertificateMaterial-/blob/main/Convolutional%20Neural%20Network/3_TransferLearning.ipynb)

### Natural Language Processing
- [DNN with Word-based Text Encoding](https://github.com/Enzofali/TensorFlowDeveloperCertificateMaterial-/blob/main/Natural%20Language%20Processing/1_WordTokenizationDNN.ipynb)
- [Conv1D with Subword Text Encoding](https://github.com/Enzofali/TensorFlowDeveloperCertificateMaterial-/blob/main/Natural%20Language%20Processing/2_SubwordTokenization.ipynb)
- [Multiple Model Architectures with Word-based Text Encoding](https://github.com/Enzofali/TensorFlowDeveloperCertificateMaterial-/blob/main/Natural%20Language%20Processing/3_PreDefinedEmbeddings.ipynb)
- [Text Generator](https://github.com/Enzofali/TensorFlowDeveloperCertificateMaterial-/blob/main/Natural%20Language%20Processing/4_TextGenerator.ipynb)

### Time Series Forecasting
- [LSTM for Time Series Forcasting](https://github.com/Enzofali/TensorFlowDeveloperCertificateMaterial-/blob/main/Time%20Series%20Forecasting/1_TimeSeriesLSTM.ipynb)
- [Conv1D for Time Series Forcasting](https://github.com/Enzofali/TensorFlowDeveloperCertificateMaterial-/blob/main/Time%20Series%20Forecasting/2_TimeSeriesConv1D.ipynb)
- [Time Series Forcasting from CSV](https://github.com/Enzofali/TensorFlowDeveloperCertificateMaterial-/blob/main/Time%20Series%20Forecasting/3_TimeSeriesCSV.ipynb)
- [N Steps Multi-Variate Time Series Forcasting](https://github.com/Enzofali/TensorFlowDeveloperCertificateMaterial-/blob/main/Time%20Series%20Forecasting/4_Multi-VariateTimeSeries.ipynb)

## Usefull Tips and Tricks

pagina oficial de la certificacion, aqui se encuentra toda la informacion relevante con respecto al examen. 
- [TensorFlow Certificate Home](https://www.tensorflow.org/certificate)

hacel el set up previo al examen y correr los modelos locamente con las versioner de pycharm tensorflow, numpy yt pandas especificadas. 
- [Environment Setup:](https://www.tensorflow.org/extras/cert/Setting_Up_TF_Developer_Certificate_Exam.pdf?authuser=4) Exam takes place in a PyCharm environment. 

PyCharm 2021.3 (either PyCharm Professional or PyCharm Community Edition)

Python 3.8.0 

tensorflow==2.9.0
tensorflow-datasets==4.6.0
Pillow==9.1.1
pandas==1.4.2
numpy==1.22.4
scipy==1.7.3

muy importante usar esta guia para saber que tenemos el conocimiento necesarioa para salvar el examen, una vez que hallamos completado la check list en toda su completitud estaremos prontos para dar el examen
- [Certificate Handbook](https://www.tensorflow.org/extras/cert/TF_Certificate_Candidate_Handbook.pdf) ⭐

5 horas
Sabes a tiempo real el resultado
el minimo puntaje por ejercicio es de 3
es decir si nuestro punteje es de 5,5,5,5,2 perdemos el examen 
los ejercicios son incrementales en dificultas (empezar por el ultimo) y supuestamente tienen una ponderacion d epeso en la nota en funcion de su dificultad
se sube solo el modelo guardado localemtne, se puede entrenar en colab bajar el modelo y subirlo
esoecial cuidado en la dimensionalidad del output del modelo, puede ser que el mismo entrene localmente, pero si el modelo no poseee la dimensionalidad del output como especifica cada ejercicio, el coso no los podra evaluar

## FAQs
Please PR the repo for FAQs
