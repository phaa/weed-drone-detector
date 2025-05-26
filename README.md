# Weed Mapping in Satellite Images

## Overview

<p align="center">
 <img src="https://i.ytimg.com/vi/P2YPG8PO9JU/maxresdefault.jpg" title="Spraying with drones" width="800" />
</p>

This project demonstrates the application of deep learning (specifically, neural networks) for the task of pixel-level weed segmentation in satellite imagery of agricultural fields. By accurately classifying pixels within the image into categories such as Soil, Vegetation, and Weeds, the project enables valuable agronomic analysis and supports precision agriculture practices like targeted spraying.

The workflow involves data loading and preparation, training a neural network model, evaluating its performance, and finally applying the model to classify an entire satellite image. The resulting classified image retains its georeferencing, making it directly applicable for use with GIS software and precision farming equipment (e.g., agricultural drones).

## Process Steps

*   Visualize original satellite imagery with georeferenced sample points.
*   Analyze the distribution of sample data classes.
*   Preprocess and split image data for model training and evaluation.
*   Design and train a sequential neural network model for pixel classification.
*   Evaluate model performance using various metrics, including handling class imbalance.
*   Classify a full satellite image, producing a georeferenced output image.
*   Extract and visualize key agronomic statistics based on the classified image.
*   Estimate costs related to treating weed-infested areas in the context of precision spraying.

---

## Using Georeferenced Imagery in Agriculture

Segmented and georeferenced imagery plays a crucial role in modern precision agriculture. By mapping field boundaries and detecting weed-infested areas at the pixel level, farmers can plan highly targeted interventions — especially when leveraging cutting-edge technologies such as agricultural drones.

When combined with drone technology, georeferenced data enables:

- **Targeted spraying**, significantly reducing herbicide waste  
- **Optimized route planning**, saving time and fuel  
- **Access to difficult terrain**, where tractors can't operate  
- **Soil preservation**, as drones avoid compaction entirely  
- **Lower operational costs**, requiring less labor and fewer inputs  

This approach turns conventional spraying into precise, site-specific treatment, reducing costs, environmental impact, and damage to soil structure.

Based on this scenario, we will now estimate the total cost required to treat all weed-affected areas identified in the classified image.

---

## Results

The project generates:
*   Plots illustrating data distribution and model performance.
*   A georeferenced GeoTIFF image (`mapa_classificado.tif`) showing the pixel-level classification of Soil, Vegetation, and Weeds.
*   Agronomic statistics and visualizations detailing area distribution and estimated treatment costs.

- Original image and predicted image
<p align="center">
 <img src="https://github.com/phaa/weed-drone-detector/blob/main/outputs/comparsion.png" title="book" width="800" />
</p>

- Area distribution charts
<p align="center">
 <img src="https://github.com/phaa/weed-drone-detector/blob/main/outputs/charts.png" title="book" width="800" />
</p>

These outputs can be used to inform decision-making in agricultural planning and execution, especially for precision spraying using drones or other variable-rate application technologies.
---

## Technologies Used

- Python 3.x
- NumPy, Pandas
- Rasterio
- Matplotlib
- TensorFlow / Keras
- geopandas

---


## Observations

- The image must contain an alpha channel to separate valid pixels (value 255).
- Samples are fundamental to train an efficient model. Use shapefiles with well-defined polygons.
- The total area is converted from pixels to hectares based on the image resolution (spatial resolution).

---

## How to run

### 1. Clone the repository

```bash
git clone https://github.com/phaa/weed-drone-detector.git
cd weed-drone-detector
```

### 2. Start your environment

```bash
conda activate seu-env
```

### 3. Open Jupyter lab

```bash
jupyter lab
```

### 4. Run the notebook
NOTE: All dependencies are installed directly through the notebook

### 5. Ensure the `datasets` folder is populated** with the required `.tif` and `.shp` files.

## Usage

Open and run the cells sequentially in the `index.ipynb` notebook. The notebook guides you through each step of the project, from data loading and visualization to model training, prediction, and agronomic analysis.

The notebook includes steps to:
*   Load and display the satellite image and sample points.
*   Prepare data for the neural network.
*   Train the neural network model.
*   Evaluate the trained model's performance.
*   Classify the entire satellite image.
*   Generate and save a georeferenced classified image (`mapa_classificado.tif`) in the `outputs` directory.
*   Calculate and visualize area statistics based on the classification.
*   Estimate potential costs for weed treatment.

## Credits

Developed by <a href='https://www.linkedin.com/in/pedro-henrique-amorim-de-azevedo/' target='_blank'>Pedro Henrique Amorim de Azevedo</a>

---
