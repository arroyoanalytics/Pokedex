# Pokédex (Pokemon Image Classifier)

A multi-class image classification project identifying all 151 Generation 1 Pokémon 
across multiple art styles using Random Forest, Neural Network, and CNN models.

## Project Overview
This project was completed as part of CAP5610 Machine Learning at University of Central Florida.
The goal was to build and compare three machine learning models capable of identifying 
Pokémon from images spanning anime screenshots, Trading Card art, in-game models, and merchandise.

## Dataset
- 34,000+ raw images scraped from Bulbagarden
- 6,040 final images (40 per Pokémon across 151 classes)
- Available on Kaggle: [Pokémon Image Classifier Dataset](https://www.kaggle.com/datasets/nyahssp/pokmon-image-classifier-dataset-cleaned)

## Pipeline
1. **Web scraping**: images pulled from Bulbagarden archive
2. **Manual cleaning**: custom tkinter GUI with background removal, magic wand, and brush tools
3. **Preprocessing**: background removal, stray pixel cleanup, 256×256 resizing
4. **Augmentation**: two pipelines (5-aug and 10-aug) applied to training set only
5. **Modeling**: Random Forest (HOG), Neural Network (MLP), CNN

## Results

| Model | Test Accuracy | Macro F1 | ROC-AUC |
|---|---|---|---|
| Random Forest (5-Aug) | 29.14% | 0.27 | 0.80 |
| Random Forest (10-Aug) | 30.13% | 0.28 | 0.81 |
| Neural Network (5-Aug) | 38.08% | 0.38 | 0.94 |
| Neural Network (10-Aug) | 38.74% | 0.38 | 0.95 |
| CNN (5-Aug) | 65.23% | 0.65 | 0.99 |
| CNN (10-Aug) | 67.22% | 0.67 | 0.99 |

## Project Structure

pokemon-image-classifier/
  - README.md
  - requirements.txt
  - data_pipeline/
    - cleaning_gui.py  (Standalone tkinter image cleaning tool)
  - notebooks/
    - pokemon_classifier.ipynb  (Full modeling pipeline)
  - assets/
      - sample_images/  (Example cleaned images)

## Setup

1. Clone this repo
2. Install dependencies: `pip install -r requirements.txt`
3. Download the dataset from Kaggle and update `DATA_PATH` in the notebook
4. Run `python data_pipeline/cleaning_gui.py` for the cleaning tool
5. Open `notebooks/pokemon_classifier.ipynb` to run the models

## Team

- Alejandro Arroyo: Data pipeline, manual cleaning, preprocessing
- Nyah Posey: Neural Network model
- Jessica Wentworth: EDA, Random Forest
- Vivi Lazo: CNN model

## Course

CAP5610 Machine Learning. University of Central Florida, Spring 2026
