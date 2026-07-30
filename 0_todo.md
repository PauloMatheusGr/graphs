# O que falta para fechar o artigo

Actualizado 2026-07-30. 

TODO

Ablação threshold análise de estabilidade {0.5,0.7,0.9} e verificar comportamento das modalidades.

Shape nos deltas deveria superar baseline, verificar.

Displacement muito ruim (perto do acaso), porém teoricamente deveria ter resultados próximos dos resultados do volume, pois volume mostra atrofia nas parcelas normalizadas de gm,csf,wm enquanto que atributos dvf mostra o quanto se deformou, solução é tentar outra abordagem: dado o conjunto de imagens longitudinais i={i1,i2,i3} realizar:

O corregistro deformavel entre i2 e i1 sendo i1 fixa e i2 móvel para gerar o campo de deformação entre i1 e i2.
O corregistro deformavel entre i3 e i1 sendo i1 fixa e i3 móvel para gerar o campo de deformação entre i1 e i3.
O corregistro deformavel entre i3 e i2 sendo i2 fixa e i3 móvel para gerar o campo de deformação entre i2 e i3.

A partir disso calcular os atributos do dvf.

