# Mapeamento de ervas daninhas em imagens de satélite

Este projeto realiza o **mapeamento de cobertura vegetal, solo exposto e infestação por plantas daninhas** em lavouras agrícolas, utilizando **redes neurais supervisionadas** e imagens multiespectrais com 4 bandas (RGB + Alpha). O objetivo é oferecer uma análise visual e estatística para apoio em **tomada de decisão agronômica**, como aplicação de defensivos agrícolas e acompanhamento da lavouras em todos os estágios do ciclo.

---

## Etapas do Processo

1. **Leitura da imagem multiespectral (.tif)**
2. **Coleta de amostras** via polígonos shapefile (.shp) de:
   - Solo exposto
   - Vegetação saudável
   - Daninhas
3. **Treinamento de uma rede neural MLP (Keras)** para classificar pixels pela cor
4. **Predição da imagem inteira** com base no modelo treinado
5. **Visualização da imagem classificada e segmentada**
6. **Geração de estatísticas**:
   - Proporção por classe
   - Área total em hectares por classe
   - Índice de infestação
   - Consumo estimado de glifosato, água e diesel
7. **Exportação de relatórios gráficos e imagens**

---

## Insights gerados 📊

- Imagem original e imagem predita
<p align="center">
 <img src="https://github.com/phaa/weed-drone-detector/blob/main/outputs/comparsion.png" title="book" width="800" />
</p>

- Gráficos de distribuição de áreas
<p align="center">
 <img src="https://github.com/phaa/weed-drone-detector/blob/main/outputs/charts.png" title="book" width="800" />
</p>

- Relatório de insumos:


---

## Tecnologias Utilizadas

- Python 3.x
- NumPy, Pandas
- Rasterio
- Matplotlib
- TensorFlow / Keras
- geopandas / shapely

---

## Aplicações ✅

- Agricultura de Precisão
- Prescrição de defensivos agrícolas
- Geração de mapas segmentados
- Monitoramento de infestação e cobertura vegetal

---

## 📌 Observações

- A imagem deve conter canal alpha para separar pixels válidos (valor 255).
- As amostras são fundamentais para treinar um modelo eficiente. Use shapefiles com polígonos bem definidos.
- A área total é convertida de pixels para hectares com base na resolução da imagem (resolução espacial).

---

## Como executar

### 1. Clone o repositório

```bash
git clone https://github.com/seu-usuario/nome-do-repo.git
cd nome-do-repo
```

### 2. Inicie seu ambiente

```bash
conda activate env
```

### 3. Abra o Jupyter lab

```bash
jupyter lab
```

### 4. Execute o notebook
Todas as dependencias são instaladas diretamente pelo notebook

## 📚 Créditos

Desenvolvido com ❤️ por <a href='https://www.linkedin.com/in/pedro-henrique-amorim-de-azevedo/' target='_blank'>Pedro Henrique Amorim de Azevedo</a>

---
