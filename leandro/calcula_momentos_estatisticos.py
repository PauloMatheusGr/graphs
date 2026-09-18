#!/usr/bin/env python
# coding: utf-8



import numpy as np
import SimpleITK as sitk
import os
import pandas as pd
import math
from scipy.stats import skew
from scipy.stats import kurtosis



df = pd.read_csv('\csv_das_imagens')

population = []


# Leitura das máscaras

mask_53 = sitk.ReadImage('/mask_path',sitk.sitkFloat64)
mask_53_array = sitk.GetArrayFromImage(mask_53)
mask_53_ok = mask_53_array > 0.0



mask_17 = sitk.ReadImage('/mask_path',sitk.sitkFloat64)
mask_17_array = sitk.GetArrayFromImage(mask_17)
mask_17_ok = mask_17_array > 0.0

mask_1000 = sitk.ReadImage('/mask_path',sitk.sitkFloat64)
mask_1000_array = sitk.GetArrayFromImage(mask_1000)
mask_1000_ok = mask_1000_array > 0.0

mask_1001 = sitk.ReadImage('/mask_path',sitk.sitkFloat64)
mask_1001_array = sitk.GetArrayFromImage(mask_1001)
mask_1001_ok = mask_1001_array > 0.0

mask_1002 = sitk.ReadImage('/mask_path',sitk.sitkFloat64)
mask_1002_array = sitk.GetArrayFromImage(mask_1002)
mask_1002_ok = mask_1002_array > 0.0

mask_1003 = sitk.ReadImage('/mask_path',sitk.sitkFloat64)
mask_1003_array = sitk.GetArrayFromImage(mask_1003)
mask_1003_ok = mask_1003_array > 0.0

mask_1007 = sitk.ReadImage('/mask_path',sitk.sitkFloat64)
mask_1007_array = sitk.GetArrayFromImage(mask_1007)
mask_1007_ok = mask_1007_array > 0.0

mask_1009 = sitk.ReadImage('/mask_path',sitk.sitkFloat64)
mask_1009_array = sitk.GetArrayFromImage(mask_1009)
mask_1009_ok = mask_1009_array > 0.0

mask_1015 = sitk.ReadImage('/mask_path',sitk.sitkFloat64)
mask_1015_array = sitk.GetArrayFromImage(mask_1015)
mask_1015_ok = mask_1015_array > 0.0

mask_1016 = sitk.ReadImage('/mask_path',sitk.sitkFloat64)
mask_1016_array = sitk.GetArrayFromImage(mask_1016)
mask_1016_ok = mask_1016_array > 0.0

mask_1026 = sitk.ReadImage('/mask_path',sitk.sitkFloat64)
mask_1026_array = sitk.GetArrayFromImage(mask_1026)
mask_1026_ok = mask_1026_array > 0.0

mask_1027 = sitk.ReadImage('/mask_path',sitk.sitkFloat64)
mask_1027_array = sitk.GetArrayFromImage(mask_1027)
mask_1027_ok = mask_1027_array > 0.0

mask_1033 = sitk.ReadImage('/mask_path',sitk.sitkFloat64)
mask_1033_array = sitk.GetArrayFromImage(mask_1033)
mask_1033_ok = mask_1033_array > 0.0

mask_11 = sitk.ReadImage('/mask_path',sitk.sitkFloat64)
mask_11_array = sitk.GetArrayFromImage(mask_11)
mask_11_ok = mask_11_array > 0.0

mask_12 = sitk.ReadImage('/mask_path',sitk.sitkFloat64)
mask_12_array = sitk.GetArrayFromImage(mask_12)
mask_12_ok = mask_12_array > 0.0

mask_18 = sitk.ReadImage('/mask_path',sitk.sitkFloat64)
mask_18_array = sitk.GetArrayFromImage(mask_18)
mask_18_ok = mask_18_array > 0.0

mask_2000 = sitk.ReadImage('/mask_path',sitk.sitkFloat64)
mask_2000_array = sitk.GetArrayFromImage(mask_2000)
mask_2000_ok = mask_2000_array > 0.0

mask_2001 = sitk.ReadImage('/mask_path',sitk.sitkFloat64)
mask_2001_array = sitk.GetArrayFromImage(mask_2001)
mask_2001_ok = mask_2001_array > 0.0

mask_2002 = sitk.ReadImage('/mask_path',sitk.sitkFloat64)
mask_2002_array = sitk.GetArrayFromImage(mask_2002)
mask_2002_ok = mask_2002_array > 0.0

mask_2003 = sitk.ReadImage('/mask_path',sitk.sitkFloat64)
mask_2003_array = sitk.GetArrayFromImage(mask_2003)
mask_2003_ok = mask_2003_array > 0.0

mask_2007 = sitk.ReadImage('/mask_path',sitk.sitkFloat64)
mask_2007_array = sitk.GetArrayFromImage(mask_2007)
mask_2007_ok = mask_2007_array > 0.0

mask_2009 = sitk.ReadImage('/mask_path',sitk.sitkFloat64)
mask_2009_array = sitk.GetArrayFromImage(mask_2009)
mask_2009_ok = mask_2009_array > 0.0

mask_2015 = sitk.ReadImage('/mask_path',sitk.sitkFloat64)
mask_2015_array = sitk.GetArrayFromImage(mask_2015)
mask_2015_ok = mask_2015_array > 0.0

mask_2016 = sitk.ReadImage('/mask_path',sitk.sitkFloat64)
mask_2016_array = sitk.GetArrayFromImage(mask_2016)
mask_2016_ok = mask_2016_array > 0.0

mask_2026 = sitk.ReadImage('/mask_path',sitk.sitkFloat64)
mask_2026_array = sitk.GetArrayFromImage(mask_2026)
mask_2026_ok = mask_2026_array > 0.0

mask_2027 = sitk.ReadImage('/mask_path',sitk.sitkFloat64)
mask_2027_array = sitk.GetArrayFromImage(mask_2027)
mask_2027_ok = mask_2027_array > 0.0

mask_2033 = sitk.ReadImage('/mask_path',sitk.sitkFloat64)
mask_2033_array = sitk.GetArrayFromImage(mask_2033)
mask_2033_ok = mask_2033_array > 0.0

mask_26 = sitk.ReadImage('/mask_path',sitk.sitkFloat64)
mask_26_array = sitk.GetArrayFromImage(mask_26)
mask_26_ok = mask_26_array > 0.0

mask_43 = sitk.ReadImage('/mask_path',sitk.sitkFloat64)
mask_43_array = sitk.GetArrayFromImage(mask_43)
mask_43_ok = mask_43_array > 0.0

mask_4 = sitk.ReadImage('/mask_path',sitk.sitkFloat64)
mask_4_array = sitk.GetArrayFromImage(mask_4)
mask_4_ok = mask_4_array > 0.0

mask_508 = sitk.ReadImage('/mask_path',sitk.sitkFloat64)
mask_508_array = sitk.GetArrayFromImage(mask_508)
mask_508_ok = mask_508_array > 0.0

mask_509 = sitk.ReadImage('/mask_path',sitk.sitkFloat64)
mask_509_array = sitk.GetArrayFromImage(mask_509)
mask_509_ok = mask_509_array > 0.0

mask_50 = sitk.ReadImage('/mask_path',sitk.sitkFloat64)
mask_50_array = sitk.GetArrayFromImage(mask_50)
mask_50_ok = mask_50_array > 0.0

mask_510 = sitk.ReadImage('/mask_path',sitk.sitkFloat64)
mask_510_array = sitk.GetArrayFromImage(mask_510)
mask_510_ok = mask_510_array > 0.0

mask_511 = sitk.ReadImage('/mask_path',sitk.sitkFloat64)
mask_511_array = sitk.GetArrayFromImage(mask_511)
mask_511_ok = mask_511_array > 0.0

mask_512 = sitk.ReadImage('/mask_path',sitk.sitkFloat64)
mask_512_array = sitk.GetArrayFromImage(mask_512)
mask_512_ok = mask_512_array > 0.0

mask_513 = sitk.ReadImage('/mask_path',sitk.sitkFloat64)
mask_513_array = sitk.GetArrayFromImage(mask_513)
mask_513_ok = mask_513_array > 0.0

mask_514 = sitk.ReadImage('/mask_path',sitk.sitkFloat64)
mask_514_array = sitk.GetArrayFromImage(mask_514)
mask_514_ok = mask_514_array > 0.0

mask_515 = sitk.ReadImage('/mask_path',sitk.sitkFloat64)
mask_515_array = sitk.GetArrayFromImage(mask_515)
mask_515_ok = mask_515_array > 0.0

mask_516 = sitk.ReadImage('/mask_path',sitk.sitkFloat64)
mask_516_array = sitk.GetArrayFromImage(mask_516)
mask_516_ok = mask_516_array > 0.0

mask_517 = sitk.ReadImage('/mask_path',sitk.sitkFloat64)
mask_517_array = sitk.GetArrayFromImage(mask_517)
mask_517_ok = mask_517_array > 0.0

mask_518 = sitk.ReadImage('/mask_path',sitk.sitkFloat64)
mask_518_array = sitk.GetArrayFromImage(mask_518)
mask_518_ok = mask_518_array > 0.0

mask_519 = sitk.ReadImage('/mask_path',sitk.sitkFloat64)
mask_519_array = sitk.GetArrayFromImage(mask_519)
mask_519_ok = mask_519_array > 0.0

mask_51 = sitk.ReadImage('/mask_path',sitk.sitkFloat64)
mask_51_array = sitk.GetArrayFromImage(mask_51)
mask_51_ok = mask_51_array > 0.0

mask_520 = sitk.ReadImage('/mask_path',sitk.sitkFloat64)
mask_520_array = sitk.GetArrayFromImage(mask_520)
mask_520_ok = mask_520_array > 0.0

mask_521 = sitk.ReadImage('/mask_path',sitk.sitkFloat64)
mask_521_array = sitk.GetArrayFromImage(mask_521)
mask_521_ok = mask_521_array > 0.0

mask_522 = sitk.ReadImage('/mask_path',sitk.sitkFloat64)
mask_522_array = sitk.GetArrayFromImage(mask_522)
mask_522_ok = mask_522_array > 0.0

mask_523 = sitk.ReadImage('/mask_path',sitk.sitkFloat64)
mask_523_array = sitk.GetArrayFromImage(mask_523)
mask_523_ok = mask_523_array > 0.0

mask_524 = sitk.ReadImage('/mask_path',sitk.sitkFloat64)
mask_524_array = sitk.GetArrayFromImage(mask_524)
mask_524_ok = mask_524_array > 0.0

mask_525 = sitk.ReadImage('/mask_path',sitk.sitkFloat64)
mask_525_array = sitk.GetArrayFromImage(mask_525)
mask_525_ok = mask_525_array > 0.0

mask_54 = sitk.ReadImage('/mask_path',sitk.sitkFloat64)
mask_54_array = sitk.GetArrayFromImage(mask_54)
mask_54_ok = mask_54_array > 0.0

mask_58 = sitk.ReadImage('/mask_path',sitk.sitkFloat64)
mask_58_array = sitk.GetArrayFromImage(mask_58)
mask_58_ok = mask_58_array > 0.0







s = np.zeros(shape=(256,256,256))

# Combina as máscaras das regiões
mask_r1_l = mask_17_ok
mask_r1_r = mask_53_ok

mask_r2_l = mask_18_ok
mask_r2_r = mask_54_ok

mask_r3_l = mask_1015_ok
mask_r3_r = mask_2015_ok

mask_r4_l = mask_1016_ok
mask_r4_r = mask_2016_ok

mask_r5_l = mask_1009_ok
mask_r5_r = mask_2009_ok

mask_r6_l = mask_11_ok
mask_r6_r = mask_50_ok

mask_r7_l = mask_1001_ok
mask_r7_r = mask_2001_ok

mask_r8_l = mask_1000_ok
mask_r8_r = mask_2000_ok

mask_r9_l = mask_12_ok
mask_r9_r = mask_51_ok

mask_r10_l = mask_26_ok
mask_r10_r = mask_58_ok

mask_r11_l = mask_1033_ok
mask_r11_r = mask_2033_ok




mask_r12_l = np.ma.mask_or(mask_509_ok, mask_511_ok)
mask_r12_l = np.ma.mask_or(mask_r12_l, mask_513_ok)
mask_r12_l = np.ma.mask_or(mask_r12_l, mask_515_ok)
mask_r12_l = np.ma.mask_or(mask_r12_l, mask_517_ok)
mask_r12_l = np.ma.mask_or(mask_r12_l, mask_519_ok)
mask_r12_l = np.ma.mask_or(mask_r12_l, mask_521_ok)
mask_r12_l = np.ma.mask_or(mask_r12_l, mask_523_ok)
mask_r12_l = np.ma.mask_or(mask_r12_l, mask_525_ok)


mask_r12_r = np.ma.mask_or(mask_508_ok, mask_510_ok)
mask_r12_r = np.ma.mask_or(mask_r12_r, mask_512_ok)
mask_r12_r = np.ma.mask_or(mask_r12_r, mask_514_ok)
mask_r12_r = np.ma.mask_or(mask_r12_r, mask_516_ok)
mask_r12_r = np.ma.mask_or(mask_r12_r, mask_518_ok)
mask_r12_r = np.ma.mask_or(mask_r12_r, mask_520_ok)
mask_r12_r = np.ma.mask_or(mask_r12_r, mask_522_ok)
mask_r12_r = np.ma.mask_or(mask_r12_r, mask_524_ok)



mask_r13_l = mask_1007_ok
mask_r13_r = mask_2007_ok


mask_r14_l = np.ma.mask_or(mask_1002_ok, mask_1026_ok)
mask_r14_r = np.ma.mask_or(mask_2002_ok, mask_2026_ok)


mask_r15_l = np.ma.mask_or(mask_1003_ok, mask_1027_ok)
mask_r15_r = np.ma.mask_or(mask_2003_ok, mask_2027_ok)



#Cria o dataframe que será preenchido com os resultados

df_mag = pd.DataFrame(columns=['Imagem', 
    'im_CN_r1_mean_l', 'im_CN_r1_var_l', 'im_CN_r1_kurt_l', 'im_CN_r1_skew_l', 'im_CN_r1_sum_l',
    'im_CN_r2_mean_l', 'im_CN_r2_var_l', 'im_CN_r2_kurt_l', 'im_CN_r2_skew_l', 'im_CN_r2_sum_l',
    'im_CN_r3_mean_l', 'im_CN_r3_var_l', 'im_CN_r3_kurt_l', 'im_CN_r3_skew_l', 'im_CN_r3_sum_l',
    'im_CN_r4_mean_l', 'im_CN_r4_var_l', 'im_CN_r4_kurt_l', 'im_CN_r4_skew_l', 'im_CN_r4_sum_l',
    'im_CN_r5_mean_l', 'im_CN_r5_var_l', 'im_CN_r5_kurt_l', 'im_CN_r5_skew_l', 'im_CN_r5_sum_l',
    'im_CN_r6_mean_l', 'im_CN_r6_var_l', 'im_CN_r6_kurt_l', 'im_CN_r6_skew_l', 'im_CN_r6_sum_l',
    'im_CN_r7_mean_l', 'im_CN_r7_var_l', 'im_CN_r7_kurt_l', 'im_CN_r7_skew_l', 'im_CN_r7_sum_l',
    'im_CN_r8_mean_l', 'im_CN_r8_var_l', 'im_CN_r8_kurt_l', 'im_CN_r8_skew_l', 'im_CN_r8_sum_l',
    'im_CN_r9_mean_l', 'im_CN_r9_var_l', 'im_CN_r9_kurt_l', 'im_CN_r9_skew_l', 'im_CN_r9_sum_l',
    'im_CN_r10_mean_l', 'im_CN_r10_var_l', 'im_CN_r10_kurt_l', 'im_CN_r10_skew_l', 'im_CN_r10_sum_l',
    'im_CN_r11_mean_l', 'im_CN_r11_var_l', 'im_CN_r11_kurt_l', 'im_CN_r11_skew_l', 'im_CN_r11_sum_l',
    'im_CN_r12_mean_l', 'im_CN_r12_var_l', 'im_CN_r12_kurt_l', 'im_CN_r12_skew_l', 'im_CN_r12_sum_l',
    'im_CN_r13_mean_l', 'im_CN_r13_var_l', 'im_CN_r13_kurt_l', 'im_CN_r13_skew_l', 'im_CN_r13_sum_l',
    'im_CN_r14_mean_l', 'im_CN_r14_var_l', 'im_CN_r14_kurt_l', 'im_CN_r14_skew_l', 'im_CN_r14_sum_l',
    'im_CN_r15_mean_l', 'im_CN_r15_var_l', 'im_CN_r15_kurt_l', 'im_CN_r15_skew_l', 'im_CN_r15_sum_l',
    'im_CN_r1_mean_r', 'im_CN_r1_var_r', 'im_CN_r1_kurt_r', 'im_CN_r1_skew_r', 'im_CN_r1_sum_r',
    'im_CN_r2_mean_r', 'im_CN_r2_var_r', 'im_CN_r2_kurt_r', 'im_CN_r2_skew_r', 'im_CN_r2_sum_r',
    'im_CN_r3_mean_r', 'im_CN_r3_var_r', 'im_CN_r3_kurt_r', 'im_CN_r3_skew_r', 'im_CN_r3_sum_r',
    'im_CN_r4_mean_r', 'im_CN_r4_var_r', 'im_CN_r4_kurt_r', 'im_CN_r4_skew_r', 'im_CN_r4_sum_r',
    'im_CN_r5_mean_r', 'im_CN_r5_var_r', 'im_CN_r5_kurt_r', 'im_CN_r5_skew_r', 'im_CN_r5_sum_r',
    'im_CN_r6_mean_r', 'im_CN_r6_var_r', 'im_CN_r6_kurt_r', 'im_CN_r6_skew_r', 'im_CN_r6_sum_r',
    'im_CN_r7_mean_r', 'im_CN_r7_var_r', 'im_CN_r7_kurt_r', 'im_CN_r7_skew_r', 'im_CN_r7_sum_r',
    'im_CN_r8_mean_r', 'im_CN_r8_var_r', 'im_CN_r8_kurt_r', 'im_CN_r8_skew_r', 'im_CN_r8_sum_r',
    'im_CN_r9_mean_r', 'im_CN_r9_var_r', 'im_CN_r9_kurt_r', 'im_CN_r9_skew_r', 'im_CN_r9_sum_r',
    'im_CN_r10_mean_r', 'im_CN_r10_var_r', 'im_CN_r10_kurt_r', 'im_CN_r10_skew_r', 'im_CN_r10_sum_r',
    'im_CN_r11_mean_r', 'im_CN_r11_var_r', 'im_CN_r11_kurt_r', 'im_CN_r11_skew_r', 'im_CN_r11_sum_r',
    'im_CN_r12_mean_r', 'im_CN_r12_var_r', 'im_CN_r12_kurt_r', 'im_CN_r12_skew_r', 'im_CN_r12_sum_r',
    'im_CN_r13_mean_r', 'im_CN_r13_var_r', 'im_CN_r13_kurt_r', 'im_CN_r13_skew_r', 'im_CN_r13_sum_r',
    'im_CN_r14_mean_r', 'im_CN_r14_var_r', 'im_CN_r14_kurt_r', 'im_CN_r14_skew_r', 'im_CN_r14_sum_r',
    'im_CN_r15_mean_r', 'im_CN_r15_var_r', 'im_CN_r15_kurt_r', 'im_CN_r15_skew_r', 'im_CN_r15_sum_r',
    'Sex', 'Age', 'Research_group'])




# Faz a iteração sobre as imagens
for index, row in df.iterrows():
    displacement_img = sitk.ReadImage('/dvf_path', sitk.sitkVectorFloat64)
    df_array = sitk.GetArrayFromImage(displacement_img)
    
    #inicializa as variáveis
    
    s = np.zeros(shape=(256,256,256))
        
        
    mean_r1_l = 0
    mean_r2_l = 0
    mean_r3_l = 0
    mean_r4_l = 0
    mean_r5_l = 0
    mean_r6_l = 0
    mean_r7_l = 0
    mean_r8_l = 0
    mean_r9_l = 0
    mean_r10_l = 0
    mean_r11_l = 0
    mean_r12_l = 0
    mean_r13_l = 0
    mean_r14_l = 0
    mean_r15_l = 0
    
    
    mean_r1_r = 0
    mean_r2_r = 0
    mean_r3_r = 0
    mean_r4_r = 0
    mean_r5_r = 0
    mean_r6_r = 0
    mean_r7_r = 0
    mean_r8_r = 0
    mean_r9_r = 0
    mean_r10_r = 0
    mean_r11_r = 0
    mean_r12_r = 0
    mean_r13_r = 0
    mean_r14_r = 0
    mean_r15_r = 0
    
    
    var_r1_l = 0
    var_r2_l = 0
    var_r3_l = 0
    var_r4_l = 0
    var_r5_l = 0
    var_r6_l = 0
    var_r7_l = 0
    var_r8_l = 0
    var_r9_l = 0
    var_r10_l = 0
    var_r11_l = 0
    var_r12_l = 0
    var_r13_l = 0
    var_r14_l = 0
    var_r15_l = 0
    
    var_r1_r = 0
    var_r2_r = 0
    var_r3_r = 0
    var_r4_r = 0
    var_r5_r = 0
    var_r6_r = 0
    var_r7_r = 0
    var_r8_r = 0
    var_r9_r = 0
    var_r10_r = 0
    var_r11_r = 0
    var_r12_r = 0
    var_r13_r = 0
    var_r14_r = 0
    var_r15_r = 0
    
    
    skew_r1_l = 0
    skew_r2_l = 0
    skew_r3_l = 0
    skew_r4_l = 0
    skew_r5_l = 0
    skew_r6_l = 0
    skew_r7_l = 0
    skew_r8_l = 0
    skew_r9_l = 0
    skew_r10_l = 0
    skew_r11_l = 0
    skew_r12_l = 0
    skew_r13_l = 0
    skew_r14_l = 0
    skew_r15_l = 0
    
    skew_r1_r = 0
    skew_r2_r = 0
    skew_r3_r = 0
    skew_r4_r = 0
    skew_r5_r = 0
    skew_r6_r = 0
    skew_r7_r = 0
    skew_r8_r = 0
    skew_r9_r = 0
    skew_r10_r = 0
    skew_r11_r = 0
    skew_r12_r = 0
    skew_r13_r = 0
    skew_r14_r = 0
    skew_r15_r = 0
    
    
    kurt_r1_l = 0
    kurt_r2_l = 0
    kurt_r3_l = 0
    kurt_r4_l = 0
    kurt_r5_l = 0
    kurt_r6_l = 0
    kurt_r7_l = 0
    kurt_r8_l = 0
    kurt_r9_l = 0
    kurt_r10_l = 0
    kurt_r11_l = 0
    kurt_r12_l = 0
    kurt_r13_l = 0
    kurt_r14_l = 0
    kurt_r15_l = 0
    
    
    kurt_r1_r = 0
    kurt_r2_r = 0
    kurt_r3_r = 0
    kurt_r4_r = 0
    kurt_r5_r = 0
    kurt_r6_r = 0
    kurt_r7_r = 0
    kurt_r8_r = 0
    kurt_r9_r = 0
    kurt_r10_r = 0
    kurt_r11_r = 0
    kurt_r12_r = 0
    kurt_r13_r = 0
    kurt_r14_r = 0
    kurt_r15_r = 0

    sum_r1_l = 0
    sum_r2_l = 0
    sum_r3_l = 0
    sum_r4_l = 0
    sum_r5_l = 0
    sum_r6_l = 0
    sum_r7_l = 0
    sum_r8_l = 0
    sum_r9_l = 0
    sum_r10_l = 0
    sum_r11_l = 0
    sum_r12_l = 0
    sum_r13_l = 0
    sum_r14_l = 0
    sum_r15_l = 0
    
    
    sum_r1_r = 0
    sum_r2_r = 0
    sum_r3_r = 0
    sum_r4_r = 0
    sum_r5_r = 0
    sum_r6_r = 0
    sum_r7_r = 0
    sum_r8_r = 0
    sum_r9_r = 0
    sum_r10_r = 0
    sum_r11_r = 0
    sum_r12_r = 0
    sum_r13_r = 0
    sum_r14_r = 0
    sum_r15_r = 0
    
    
    
    
                
      
    
       
    
    mean_r1_l = np.mean(df_array[mask_r1_l])
    mean_r2_l = np.mean(df_array[mask_r2_l])
    mean_r3_l = np.mean(df_array[mask_r3_l])
    mean_r4_l = np.mean(df_array[mask_r4_l])
    mean_r5_l = np.mean(df_array[mask_r5_l])
    mean_r6_l = np.mean(df_array[mask_r6_l])
    mean_r7_l = np.mean(df_array[mask_r7_l])
    mean_r8_l = np.mean(df_array[mask_r8_l])
    mean_r9_l = np.mean(df_array[mask_r9_l])
    mean_r10_l = np.mean(df_array[mask_r10_l])
    mean_r11_l = np.mean(df_array[mask_r11_l])
    mean_r12_l = np.mean(df_array[mask_r12_l])
    mean_r13_l = np.mean(df_array[mask_r13_l])
    mean_r14_l = np.mean(df_array[mask_r14_l])
    mean_r15_l = np.mean(df_array[mask_r15_l])
    
    mean_r1_r = np.mean(df_array[mask_r1_r])
    mean_r2_r = np.mean(df_array[mask_r2_r])
    mean_r3_r = np.mean(df_array[mask_r3_r])
    mean_r4_r = np.mean(df_array[mask_r4_r])
    mean_r5_r = np.mean(df_array[mask_r5_r])
    mean_r6_r = np.mean(df_array[mask_r6_r])
    mean_r7_r = np.mean(df_array[mask_r7_r])
    mean_r8_r = np.mean(df_array[mask_r8_r])
    mean_r9_r = np.mean(df_array[mask_r9_r])
    mean_r10_r = np.mean(df_array[mask_r10_r])
    mean_r11_r = np.mean(df_array[mask_r11_r])
    mean_r12_r = np.mean(df_array[mask_r12_r])
    mean_r13_r = np.mean(df_array[mask_r13_r])
    mean_r14_r = np.mean(df_array[mask_r14_r])
    mean_r15_r = np.mean(df_array[mask_r15_r])
    
    
    var_r1_l = np.var(df_array[mask_r1_l])
    var_r2_l = np.var(df_array[mask_r2_l])
    var_r3_l = np.var(df_array[mask_r3_l])
    var_r4_l = np.var(df_array[mask_r4_l])
    var_r5_l = np.var(df_array[mask_r5_l])
    var_r6_l = np.var(df_array[mask_r6_l])
    var_r7_l = np.var(df_array[mask_r7_l])
    var_r8_l = np.var(df_array[mask_r8_l])
    var_r9_l = np.var(df_array[mask_r9_l])
    var_r10_l = np.var(df_array[mask_r10_l])
    var_r11_l = np.var(df_array[mask_r11_l])
    var_r12_l = np.var(df_array[mask_r12_l])
    var_r13_l = np.var(df_array[mask_r13_l])
    var_r14_l = np.var(df_array[mask_r14_l])
    var_r15_l = np.var(df_array[mask_r15_l])
    
    var_r1_r = np.var(df_array[mask_r1_r])
    var_r2_r = np.var(df_array[mask_r2_r])
    var_r3_r = np.var(df_array[mask_r3_r])
    var_r4_r = np.var(df_array[mask_r4_r])
    var_r5_r = np.var(df_array[mask_r5_r])
    var_r6_r = np.var(df_array[mask_r6_r])
    var_r7_r = np.var(df_array[mask_r7_r])
    var_r8_r = np.var(df_array[mask_r8_r])
    var_r9_r = np.var(df_array[mask_r9_r])
    var_r10_r = np.var(df_array[mask_r10_r])
    var_r11_r = np.var(df_array[mask_r11_r])
    var_r12_r = np.var(df_array[mask_r12_r])
    var_r13_r = np.var(df_array[mask_r13_r])
    var_r14_r = np.var(df_array[mask_r14_r])
    var_r15_r = np.var(df_array[mask_r15_r])
    
    
    skew_r1_l = skew(df_array[mask_r1_l], axis=0, bias=True)
    skew_r2_l = skew(df_array[mask_r2_l], axis=0, bias=True)
    skew_r3_l = skew(df_array[mask_r3_l], axis=0, bias=True)
    skew_r4_l = skew(df_array[mask_r4_l], axis=0, bias=True)
    skew_r5_l = skew(df_array[mask_r5_l], axis=0, bias=True)
    skew_r6_l = skew(df_array[mask_r6_l], axis=0, bias=True)
    skew_r7_l = skew(df_array[mask_r7_l], axis=0, bias=True)
    skew_r8_l = skew(df_array[mask_r8_l], axis=0, bias=True)
    skew_r9_l = skew(df_array[mask_r9_l], axis=0, bias=True)
    skew_r10_l = skew(df_array[mask_r10_l], axis=0, bias=True)
    skew_r11_l = skew(df_array[mask_r11_l], axis=0, bias=True)
    skew_r12_l = skew(df_array[mask_r12_l], axis=0, bias=True)
    skew_r13_l = skew(df_array[mask_r13_l], axis=0, bias=True)
    skew_r14_l = skew(df_array[mask_r14_l], axis=0, bias=True)
    skew_r15_l = skew(df_array[mask_r15_l], axis=0, bias=True)
    
    skew_r1_r = skew(df_array[mask_r1_r], axis=0, bias=True)
    skew_r2_r = skew(df_array[mask_r2_r], axis=0, bias=True)
    skew_r3_r = skew(df_array[mask_r3_r], axis=0, bias=True)
    skew_r4_r = skew(df_array[mask_r4_r], axis=0, bias=True)
    skew_r5_r = skew(df_array[mask_r5_r], axis=0, bias=True)
    skew_r6_r = skew(df_array[mask_r6_r], axis=0, bias=True)
    skew_r7_r = skew(df_array[mask_r7_r], axis=0, bias=True)
    skew_r8_r = skew(df_array[mask_r8_r], axis=0, bias=True)
    skew_r9_r = skew(df_array[mask_r9_r], axis=0, bias=True)
    skew_r10_r = skew(df_array[mask_r10_r], axis=0, bias=True)
    skew_r11_r = skew(df_array[mask_r11_r], axis=0, bias=True)
    skew_r12_r = skew(df_array[mask_r12_r], axis=0, bias=True)
    skew_r13_r = skew(df_array[mask_r13_r], axis=0, bias=True)
    skew_r14_r = skew(df_array[mask_r14_r], axis=0, bias=True)
    skew_r15_r = skew(df_array[mask_r15_r], axis=0, bias=True)
    
    
    kurt_r1_l = kurtosis(df_array[mask_r1_l], axis=0, bias=True)
    kurt_r2_l = kurtosis(df_array[mask_r2_l], axis=0, bias=True)
    kurt_r3_l = kurtosis(df_array[mask_r3_l], axis=0, bias=True)
    kurt_r4_l = kurtosis(df_array[mask_r4_l], axis=0, bias=True)
    kurt_r5_l = kurtosis(df_array[mask_r5_l], axis=0, bias=True)
    kurt_r6_l = kurtosis(df_array[mask_r6_l], axis=0, bias=True)
    kurt_r7_l = kurtosis(df_array[mask_r7_l], axis=0, bias=True)
    kurt_r8_l = kurtosis(df_array[mask_r8_l], axis=0, bias=True)
    kurt_r9_l = kurtosis(df_array[mask_r9_l], axis=0, bias=True)
    kurt_r10_l = kurtosis(df_array[mask_r10_l], axis=0, bias=True)
    kurt_r11_l = kurtosis(df_array[mask_r11_l], axis=0, bias=True)
    kurt_r12_l = kurtosis(df_array[mask_r12_l], axis=0, bias=True)
    kurt_r13_l = kurtosis(df_array[mask_r13_l], axis=0, bias=True)
    kurt_r14_l = kurtosis(df_array[mask_r14_l], axis=0, bias=True)
    kurt_r15_l = kurtosis(df_array[mask_r15_l], axis=0, bias=True)
    
    kurt_r1_r = kurtosis(df_array[mask_r1_r], axis=0, bias=True)
    kurt_r2_r = kurtosis(df_array[mask_r2_r], axis=0, bias=True)
    kurt_r3_r = kurtosis(df_array[mask_r3_r], axis=0, bias=True)
    kurt_r4_r = kurtosis(df_array[mask_r4_r], axis=0, bias=True)
    kurt_r5_r = kurtosis(df_array[mask_r5_r], axis=0, bias=True)
    kurt_r6_r = kurtosis(df_array[mask_r6_r], axis=0, bias=True)
    kurt_r7_r = kurtosis(df_array[mask_r7_r], axis=0, bias=True)
    kurt_r8_r = kurtosis(df_array[mask_r8_r], axis=0, bias=True)
    kurt_r9_r = kurtosis(df_array[mask_r9_r], axis=0, bias=True)
    kurt_r10_r = kurtosis(df_array[mask_r10_r], axis=0, bias=True)
    kurt_r11_r = kurtosis(df_array[mask_r11_r], axis=0, bias=True)
    kurt_r12_r = kurtosis(df_array[mask_r12_r], axis=0, bias=True)
    kurt_r13_r = kurtosis(df_array[mask_r13_r], axis=0, bias=True)
    kurt_r14_r = kurtosis(df_array[mask_r14_r], axis=0, bias=True)
    kurt_r15_r = kurtosis(df_array[mask_r15_r], axis=0, bias=True)

    sum_r1_l = np.sum(df_array[mask_r1_l])
    sum_r2_l = np.sum(df_array[mask_r2_l])
    sum_r3_l = np.sum(df_array[mask_r3_l])
    sum_r4_l = np.sum(df_array[mask_r4_l])
    sum_r5_l = np.sum(df_array[mask_r5_l])
    sum_r6_l = np.sum(df_array[mask_r6_l])
    sum_r7_l = np.sum(df_array[mask_r7_l])
    sum_r8_l = np.sum(df_array[mask_r8_l])
    sum_r9_l = np.sum(df_array[mask_r9_l])
    sum_r10_l = np.sum(df_array[mask_r10_l])
    sum_r11_l = np.sum(df_array[mask_r11_l])
    sum_r12_l = np.sum(df_array[mask_r12_l])
    sum_r13_l = np.sum(df_array[mask_r13_l])
    sum_r14_l = np.sum(df_array[mask_r14_l])
    sum_r15_l = np.sum(df_array[mask_r15_l])
    
    sum_r1_r = np.sum(df_array[mask_r1_r])
    sum_r2_r = np.sum(df_array[mask_r2_r])
    sum_r3_r = np.sum(df_array[mask_r3_r])
    sum_r4_r = np.sum(df_array[mask_r4_r])
    sum_r5_r = np.sum(df_array[mask_r5_r])
    sum_r6_r = np.sum(df_array[mask_r6_r])
    sum_r7_r = np.sum(df_array[mask_r7_r])
    sum_r8_r = np.sum(df_array[mask_r8_r])
    sum_r9_r = np.sum(df_array[mask_r9_r])
    sum_r10_r = np.sum(df_array[mask_r10_r])
    sum_r11_r = np.sum(df_array[mask_r11_r])
    sum_r12_r = np.sum(df_array[mask_r12_r])
    sum_r13_r = np.sum(df_array[mask_r13_r])
    sum_r14_r = np.sum(df_array[mask_r14_r])
    sum_r15_r = np.sum(df_array[mask_r15_r])

    
    

                    
    #Salva a soma das magnitudes em uma planilha 
    new_row = {
    'Imagem': row["ID"],
    # Médias
    'im_CN_r1_mean_l': mean_r1_l, 'im_CN_r2_mean_l': mean_r2_l, 'im_CN_r3_mean_l': mean_r3_l, 'im_CN_r4_mean_l': mean_r4_l,
    'im_CN_r5_mean_l': mean_r5_l, 'im_CN_r6_mean_l': mean_r6_l, 'im_CN_r7_mean_l': mean_r7_l, 'im_CN_r8_mean_l': mean_r8_l,
    'im_CN_r9_mean_l': mean_r9_l, 'im_CN_r10_mean_l': mean_r10_l, 'im_CN_r11_mean_l': mean_r11_l, 'im_CN_r12_mean_l': mean_r12_l,
    'im_CN_r13_mean_l': mean_r13_l, 'im_CN_r14_mean_l': mean_r14_l, 'im_CN_r15_mean_l': mean_r15_l,

    'im_CN_r1_mean_r': mean_r1_r, 'im_CN_r2_mean_r': mean_r2_r, 'im_CN_r3_mean_r': mean_r3_r, 'im_CN_r4_mean_r': mean_r4_r,
    'im_CN_r5_mean_r': mean_r5_r, 'im_CN_r6_mean_r': mean_r6_r, 'im_CN_r7_mean_r': mean_r7_r, 'im_CN_r8_mean_r': mean_r8_r,
    'im_CN_r9_mean_r': mean_r9_r, 'im_CN_r10_mean_r': mean_r10_r, 'im_CN_r11_mean_r': mean_r11_r, 'im_CN_r12_mean_r': mean_r12_r,
    'im_CN_r13_mean_r': mean_r13_r, 'im_CN_r14_mean_r': mean_r14_r, 'im_CN_r15_mean_r': mean_r15_r,

    # Variâncias
    'im_CN_r1_var_l': var_r1_l, 'im_CN_r2_var_l': var_r2_l, 'im_CN_r3_var_l': var_r3_l, 'im_CN_r4_var_l': var_r4_l,
    'im_CN_r5_var_l': var_r5_l, 'im_CN_r6_var_l': var_r6_l, 'im_CN_r7_var_l': var_r7_l, 'im_CN_r8_var_l': var_r8_l,
    'im_CN_r9_var_l': var_r9_l, 'im_CN_r10_var_l': var_r10_l, 'im_CN_r11_var_l': var_r11_l, 'im_CN_r12_var_l': var_r12_l,
    'im_CN_r13_var_l': var_r13_l, 'im_CN_r14_var_l': var_r14_l, 'im_CN_r15_var_l': var_r15_l,

    'im_CN_r1_var_r': var_r1_r, 'im_CN_r2_var_r': var_r2_r, 'im_CN_r3_var_r': var_r3_r, 'im_CN_r4_var_r': var_r4_r,
    'im_CN_r5_var_r': var_r5_r, 'im_CN_r6_var_r': var_r6_r, 'im_CN_r7_var_r': var_r7_r, 'im_CN_r8_var_r': var_r8_r,
    'im_CN_r9_var_r': var_r9_r, 'im_CN_r10_var_r': var_r10_r, 'im_CN_r11_var_r': var_r11_r, 'im_CN_r12_var_r': var_r12_r,
    'im_CN_r13_var_r': var_r13_r, 'im_CN_r14_var_r': var_r14_r, 'im_CN_r15_var_r': var_r15_r,

    # Curtose
    'im_CN_r1_kurt_l': kurt_r1_l, 'im_CN_r2_kurt_l': kurt_r2_l, 'im_CN_r3_kurt_l': kurt_r3_l, 'im_CN_r4_kurt_l': kurt_r4_l,
    'im_CN_r5_kurt_l': kurt_r5_l, 'im_CN_r6_kurt_l': kurt_r6_l, 'im_CN_r7_kurt_l': kurt_r7_l, 'im_CN_r8_kurt_l': kurt_r8_l,
    'im_CN_r9_kurt_l': kurt_r9_l, 'im_CN_r10_kurt_l': kurt_r10_l, 'im_CN_r11_kurt_l': kurt_r11_l, 'im_CN_r12_kurt_l': kurt_r12_l,
    'im_CN_r13_kurt_l': kurt_r13_l, 'im_CN_r14_kurt_l': kurt_r14_l, 'im_CN_r15_kurt_l': kurt_r15_l,

    'im_CN_r1_kurt_r': kurt_r1_r, 'im_CN_r2_kurt_r': kurt_r2_r, 'im_CN_r3_kurt_r': kurt_r3_r, 'im_CN_r4_kurt_r': kurt_r4_r,
    'im_CN_r5_kurt_r': kurt_r5_r, 'im_CN_r6_kurt_r': kurt_r6_r, 'im_CN_r7_kurt_r': kurt_r7_r, 'im_CN_r8_kurt_r': kurt_r8_r,
    'im_CN_r9_kurt_r': kurt_r9_r, 'im_CN_r10_kurt_r': kurt_r10_r, 'im_CN_r11_kurt_r': kurt_r11_r, 'im_CN_r12_kurt_r': kurt_r12_r,
    'im_CN_r13_kurt_r': kurt_r13_r, 'im_CN_r14_kurt_r': kurt_r14_r, 'im_CN_r15_kurt_r': kurt_r15_r,

    # Assimetria (Skewness)
    'im_CN_r1_skew_l': skew_r1_l, 'im_CN_r2_skew_l': skew_r2_l, 'im_CN_r3_skew_l': skew_r3_l, 'im_CN_r4_skew_l': skew_r4_l,
    'im_CN_r5_skew_l': skew_r5_l, 'im_CN_r6_skew_l': skew_r6_l, 'im_CN_r7_skew_l': skew_r7_l, 'im_CN_r8_skew_l': skew_r8_l,
    'im_CN_r9_skew_l': skew_r9_l, 'im_CN_r10_skew_l': skew_r10_l, 'im_CN_r11_skew_l': skew_r11_l, 'im_CN_r12_skew_l': skew_r12_l,
    'im_CN_r13_skew_l': skew_r13_l, 'im_CN_r14_skew_l': skew_r14_l, 'im_CN_r15_skew_l': skew_r15_l,

    'im_CN_r1_skew_r': skew_r1_r, 'im_CN_r2_skew_r': skew_r2_r, 'im_CN_r3_skew_r': skew_r3_r, 'im_CN_r4_skew_r': skew_r4_r,
    'im_CN_r5_skew_r': skew_r5_r, 'im_CN_r6_skew_r': skew_r6_r, 'im_CN_r7_skew_r': skew_r7_r, 'im_CN_r8_skew_r': skew_r8_r,
    'im_CN_r9_skew_r': skew_r9_r, 'im_CN_r10_skew_r': skew_r10_r, 'im_CN_r11_skew_r': skew_r11_r, 'im_CN_r12_skew_r': skew_r12_r,
    'im_CN_r13_skew_r': skew_r13_r, 'im_CN_r14_skew_r': skew_r14_r, 'im_CN_r15_skew_r': skew_r15_r,

    # Soma das magnitudes
    'im_CN_r1_sum_l': sum_r1_l, 'im_CN_r2_sum_l': sum_r2_l, 'im_CN_r3_sum_l': sum_r3_l, 'im_CN_r4_sum_l': sum_r4_l,
    'im_CN_r5_sum_l': sum_r5_l, 'im_CN_r6_sum_l': sum_r6_l, 'im_CN_r7_sum_l': sum_r7_l, 'im_CN_r8_sum_l': sum_r8_l,
    'im_CN_r9_sum_l': sum_r9_l, 'im_CN_r10_sum_l': sum_r10_l, 'im_CN_r11_sum_l': sum_r11_l, 'im_CN_r12_sum_l': sum_r12_l,
    'im_CN_r13_sum_l': sum_r13_l, 'im_CN_r14_sum_l': sum_r14_l, 'im_CN_r15_sum_l': sum_r15_l,

    'im_CN_r1_sum_r': sum_r1_r, 'im_CN_r2_sum_r': sum_r2_r, 'im_CN_r3_sum_r': sum_r3_r, 'im_CN_r4_sum_r': sum_r4_r,
    'im_CN_r5_sum_r': sum_r5_r, 'im_CN_r6_sum_r': sum_r6_r, 'im_CN_r7_sum_r': sum_r7_r, 'im_CN_r8_sum_r': sum_r8_r,
    'im_CN_r9_sum_r': sum_r9_r, 'im_CN_r10_sum_r': sum_r10_r, 'im_CN_r11_sum_r': sum_r11_r, 'im_CN_r12_sum_r': sum_r12_r,
    'im_CN_r13_sum_r': sum_r13_r, 'im_CN_r14_sum_r': sum_r14_r, 'im_CN_r15_sum_r': sum_r15_r,

    'Sex': row["Sex"], 'Age': row["Age"], 'Research_group': 'Converter'
}

    
    
    
    
    df_mag = pd.concat([df_mag, pd.DataFrame([new_row])], ignore_index=True)
   
       




df_mag = df_mag.round(2)





df_mag.to_csv('/caminho_pra_salvar_os_atributos_extraídos')












