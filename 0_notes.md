# Notas importantes para o artigo

além do late fusion ter melhores resultados em termos de auc mean patient, a escolha final por late fusion é em razão de: 

Late (o que chamas multiclasse de imagem): cinco SVMs, cada um na sua família já filtrada; junta-se a média dos scores. Forma não vê as 144 GLCM. Volume não vê firstorder. Não existe um classificador com “todas as famílias × 3 tempos”. O risco 
p≫n da união não se aplica ao late. O que cresce é o número de modelos, não a dimensão de um só.

Early: concatena colunas depois junta a selecção num saco. Cinco famílias IBSI-cheias × Q4 → centenas de colunas, 
n=120. Aí sim p∼n ou p>n, pool instável, overfitting. Já tinhas late > early na âncora compacta: o early largo piora isso. Por isso a grade 232 e o early IBSI-cheio são má ideia mesmo que o late seja inofensivo em dimensão.

Cinco famílias compactas no código (`ablation_prep.py`). Firstorder = 4 momentos, não IBSI-cheio.

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

**firstorder** — momentos 1–4 do histograma  
- `original_firstorder_Mean`  
- `original_firstorder_Variance`  
- `original_firstorder_Skewness`  
- `original_firstorder_Kurtosis`  

---

## Achados (SVM, nocombat, AUC paciente, `t1_d21_d32` vs `t1_only`)

Fonte: `csvs/cohort_comparison/cohort_results.csv` (`n_boot=2000`). Claim = `48m_12m`. Longitudinal = Q4 (T1+D21+D32). Não coroar `wide`/`abs`/`t1_deltas`/`t1_deltas_rel`/`t1_ma`.

### Unimodal: Q4 não bate baseline em todas as famílias

Δ = Q4 − `t1_only`. Só **texture** ganha nas 4 coortes.

| Coorte | vol | shape | texture | disp | firstorder |
|--------|-----|-------|---------|------|------------|
| 36m_6m | +0.016 | +0.003 | **+0.035** | −0.030 | +0.001 |
| 36m_12m | −0.024 | −0.015 | **+0.035** | −0.086 | −0.012 |
| 48m_6m | +0.037 | −0.008 | **+0.036** | −0.032 | −0.003 |
| 48m_12m | +0.026 | +0.007 | **+0.047** | +0.023 | −0.002 |
| Q4>t1 | 3/4 | 2/4 | **4/4** | 1/4 | 1/4 |

- **Texture** — único sinal temporal estável. GLCM em \(t_0\) ~0.70–0.73; D21/D32 sobem ~+0.035 a +0.047 (máx. claim: 0.695 → 0.742).
- **Shape** — teto estático. `t1_only` 0.74–0.80; Q4 ≈ empate ou piora. Geometria em \(t_0\) já separa sMCI/pMCI.
- **Vol** — ganho em 3/4; quebra em `36m_12m` (−0.024).
- **Firstorder** — empate (~0). Intensidade global em \(t_0\) já no modelo; momentos-Δ não repetem o ganho GLCM (controlo: nem todo radiomics ganha com deltas).
- **Disp** — Q4 piora em 3/4. DVF vs atlas CN não carrega conversão neste desenho.

### União (late) ≠ maior vitória do longitudinal

Dois contrastes. AUC mais alto = late (complementaridade de famílias). Ganho Q4−t1 **maior na texture unimodal**, não na união.

Late oficial (5 SVMs, média de scores):

| Coorte | late tudo-t1 | late tudo-Q4 | âncora (shape@t1 ∪ resto Q4) |
|--------|--------------|--------------|------------------------------|
| 36m_6m | 0.822 | 0.823 | 0.824 |
| 36m_12m | 0.782 | 0.783 | 0.782 |
| 48m_6m | 0.822 | 0.832 | 0.834 |
| 48m_12m | 0.765 | 0.790 | 0.785 |

Δ longitudinal **dentro** da união (late Q4 − late t1) vs Δ texture unimodal:

| Coorte | Δ texture unimodal | Δ late Q4−t1 | Δ âncora − late t1 |
|--------|--------------------|--------------|---------------------|
| 36m_6m | **+0.035** | +0.001 | +0.002 |
| 36m_12m | **+0.035** | +0.001 | −0.001 |
| 48m_6m | **+0.036** | +0.009 | +0.012 |
| 48m_12m | **+0.047** | +0.025 | +0.020 |

Âncora vs shape só \(t_0\) sobe nas 4 (+0.024 a +0.041). Isso é **juntar famílias**, não “longitudinal vence baseline”. Contra late já-tudo-t1, extra Q4 quase some excepto na claim.

**Paper:** unimodal = vitórias pontuais do tempo (GLCM). Late = teto porque junta papéis (forma estática + textura-Δ). Não inverter: união não é onde o longitudinal mais vence o baseline.

### AUC > 0.8

Só nas coortes gap **6m**. Claim e `36m_12m` ficam abaixo.

| Coorte | late (3 specs oficiais) | unimodal imagem |
|--------|-------------------------|-----------------|
| 36m_6m | 3/3 >0.8 (máx. âncora 0.824) | shape t1 = 0.788 (não passa) |
| 48m_6m | 3/3 >0.8 (máx. âncora 0.834) | shape t1 = 0.802 |
| 36m_12m | 0/3 (máx. 0.783) | nenhum |
| 48m_12m claim | 0/3 (máx. late Q4 0.790) | nenhum |

Clinic+img na claim (~0.85) **não** conta como união de famílias de imagem.
