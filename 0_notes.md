# Notas importantes para o artigo

além do late fusion ter melhores resultados em termos de auc mean patient, a escolha final por late fusion é em razão de: 

Late (o que chamas multiclasse de imagem): cinco SVMs, cada um na sua família já filtrada; junta-se a média dos scores. Forma não vê as 144 GLCM. Volume não vê firstorder. Não existe um classificador com “todas as famílias × 3 tempos”. O risco 
p≫n da união não se aplica ao late. O que cresce é o número de modelos, não a dimensão de um só.

Early: concatena colunas depois junta a selecção num saco. Cinco famílias IBSI-cheias × Q4 → centenas de colunas, 
n=120. Aí sim p∼n ou p>n, pool instável, overfitting. Já tinhas late > early na âncora compacta: o early largo piora isso. Por isso a grade 232 e o early IBSI-cheio são má ideia mesmo que o late seja inofensivo em dimensão.

Quatro famílias **já no código** (`ablation_prep.py`). Firstorder **ainda não está no repo** — compacto acordado: 4 momentos.

Cada sufixo × **L e R**. `t1_only` = ×1 (T1). Q4 = ×3 (T1, D21, D32).

| Família | Sufixos (por lado) | `t1_only` L+R | Q4 L+R |
|---|---|---|---|
| vol | 4 | 8 | 24 |
| shape | 5 | 10 | 30 |
| texture | 4 | 8 | 24 |
| disp | 4 | 8 | 24 |
| firstorder | 4 | 8 | 24 |

**vol** — fracção na ROI, malha /ICV  
- `gm_norm`  
- `wm_norm`  
- `csf_norm`  
- `original_shape_MeshVolume`  

**shape** — geometria (sem tamanho de malha)  
- `original_shape_SurfaceArea`  
- `original_shape_SurfaceVolumeRatio`  
- `original_shape_Sphericity`  
- `original_shape_Elongation`  
- `original_shape_Flatness`  

**texture** — 4 GLCM  
- `original_glcm_Contrast`  
- `original_glcm_Correlation`  
- `original_glcm_Idm`  
- `original_glcm_JointEntropy`  

**disp** — DVF vs atlas CN  
- `mag_mean`  
- `strain_fro_mean`  
- `strain_fro_variance`  
- `logjac_variance`  

**firstorder** (a implementar) — momentos 1–4 do histograma  
- `original_firstorder_Mean`  
- `original_firstorder_Variance`  
- `original_firstorder_Skewness`  
- `original_firstorder_Kurtosis`  
