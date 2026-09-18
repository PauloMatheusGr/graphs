#!/usr/bin/env python
# coding: utf-8


import SimpleITK as sitk
import numpy as np
import os
import pandas as pd

# Loop para leitura e iteração sobre os aquivos CSVs com os caminhos das imagens
##for n, id_ in enumerate(ids):
df = pd.read_csv('')
population = []
# Loop para composição da série de imagens para compor o groupwise de cada faixa etária após o corregistro afim
for index, row in df.iterrows():
    population.append('' + row["Image Filename"])
    
# Monta a imagem 4D com o conjunto de imagens 3D
vectorOfImages = sitk.VectorOfImage()
for filename in population:
    vectorOfImages.push_back(sitk.ReadImage(filename))

image = sitk.JoinSeries(vectorOfImages)
    
# Realiza o corregistro
elastixImageFilter = sitk.ElastixImageFilter()
elastixImageFilter.SetFixedImage(image)
elastixImageFilter.SetMovingImage(image)

# Lê os parâmetros
elastixImageFilter.SetParameterMap(sitk.ReadParameterFile(""))
elastixImageFilter.Execute()
    
# Obtém a imagem 4D resultante, faz a extração das imagens 3D e calcula a imagem média
transformed_img = elastixImageFilter.GetResultImage()

# Grava a imagem 4D resultante
sitk.WriteImage(transformed_img, '')







