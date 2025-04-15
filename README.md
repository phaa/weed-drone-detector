# Weed Mapping in Satellite Images

This project performs **vegetation cover, exposed soil, and weed infestation mapping** in agricultural crops, using **supervised neural networks** and multispectral images with 4 bands (RGB + Alpha). The objective is to provide visual and statistical analysis to support **agronomic decision-making**, such as agricultural pesticide application and crop monitoring throughout all stages of the cycle.

---

# Using Georeferenced Imagery in Agriculture

Segmented and georeferenced imagery plays a crucial role in modern precision agriculture. By mapping field boundaries and detecting weed-infested areas at the pixel level, farmers can plan highly targeted interventions — especially when leveraging cutting-edge technologies such as agricultural drones.

<p align="center">
 <img src="https://i.ytimg.com/vi/P2YPG8PO9JU/maxresdefault.jpg" title="Spraying with drones" width="800" />
</p>

When combined with drone technology, georeferenced data enables:

- **Targeted spraying**, significantly reducing herbicide waste  
- **Optimized route planning**, saving time and fuel  
- **Access to difficult terrain**, where tractors can't operate  
- **Soil preservation**, as drones avoid compaction entirely  
- **Lower operational costs**, requiring less labor and fewer inputs  

This approach turns conventional spraying into precise, site-specific treatment, reducing costs, environmental impact, and damage to soil structure.

Based on this scenario, we will now estimate the total cost required to treat all weed-affected areas identified in the classified image.

---

## Process Steps

1. **Reading the multispectral image (.tif)**
2. **Sample collection** via shapefile polygons (.shp) of:
   - Exposed soil
   - Healthy vegetation
   - Weeds
3. **Training an MLP neural network (Keras)** to classify pixels by color
4. **Predicting the entire image** based on the trained model
5. **Visualization of the classified and segmented image**
6. **Generation of statistics**:
   - Proportion by class
   - Total area in hectares by class
   - Infestation index
   - Estimated consumption of herbicide, water, and energy
7. **Export of graphical reports and images**

---

## Generated Insights 📊

- Original image and predicted image
<p align="center">
 <img src="https://github.com/phaa/weed-drone-detector/blob/main/outputs/comparsion.png" title="book" width="800" />
</p>

- Area distribution charts
<p align="center">
 <img src="https://github.com/phaa/weed-drone-detector/blob/main/outputs/charts.png" title="book" width="800" />
</p>

---

## Technologies Used

- Python 3.x
- NumPy, Pandas
- Rasterio
- Matplotlib
- TensorFlow / Keras
- geopandas / shapely

---

## Applications ✅

- Precision Agriculture
- Agricultural pesticide prescription
- Generation of segmented maps
- Monitoring of infestation and vegetation cover

---

## 📌 Observations

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

## 📚 Credits

Developed with ❤️ by <a href='https://www.linkedin.com/in/pedro-henrique-amorim-de-azevedo/' target='_blank'>Pedro Henrique Amorim de Azevedo</a>

---
