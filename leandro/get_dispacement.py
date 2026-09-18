#!/usr/bin/env python
# coding: utf-8


# imports
import numpy as np
import SimpleITK as sitk
import os
import pandas as pd




# csv com as imagens
df = pd.read_csv('')
population = []



for index, row in df.iterrows():
    population.append(row["Image Filename"])




for filename in population:
    # Realiza o corregistro afim
    # Leitura das imagens
    fix_img = sitk.ReadImage('',sitk.sitkFloat32)
    mov_img = sitk.ReadImage('' + filename, sitk.sitkFloat32)
    
    #Realiza o corregistro
    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(fix_img)
    elastixImageFilter.SetMovingImage(mov_img)
    elastixImageFilter.SetParameterMap(sitk.ReadParameterFile(""))
    elastixImageFilter.Execute()
    
    # Obtém a imagem resultante
    res_img =  elastixImageFilter.GetResultImage() 
    
    #Realiza o coregistro deformável
    fix_img = sitk.ReadImage('',sitk.sitkFloat32)
    mov_img = res_img
    
    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(fix_img)
    elastixImageFilter.SetMovingImage(mov_img)
    elastixImageFilter.SetParameterMap(sitk.ReadParameterFile(""))
    elastixImageFilter.Execute()
    
    #Obtém o displacement Field
    logdir = ''
    transformixImageFilter = sitk.TransformixImageFilter()
    transformixImageFilter.SetMovingImage(mov_img)
    transformixImageFilter.SetTransformParameterMap(elastixImageFilter.GetTransformParameterMap())
    transformixImageFilter.ComputeDeformationFieldOn()
    transformixImageFilter.ComputeDeterminantOfSpatialJacobianOn()
    transformixImageFilter.ComputeSpatialJacobianOn()
    transformixImageFilter.LogToConsoleOn()
    transformixImageFilter.SetOutputDirectory(logdir)
    transformixImageFilter.Execute()
    
    resultDeformationField = transformixImageFilter.GetDeformationField()
    #Salva os Displacements
    sitk.WriteImage(resultDeformationField,'' + filename)
    
    

