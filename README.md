## Introduction
PathoTransformer is designed to leverage the advancements in transformer models for the rapid and accurate classification of pathogens from genomic sequences. This tool aims to improve upon traditional sequence alignment methods, which are computationally intensive and less effective against novel or rapidly mutating viruses. By integrating large-scale genomic data with deep learning techniques, PathoTransformer offers a robust solution for modern infectious disease control and outbreak management.

## Project Overview
Our project builds on previous research that utilized traditional machine learning models, such as LSTM networks, and introduces the PathoTransformer model. This model benefits from the enhanced attention mechanisms of transformer architectures to improve classification accuracy and speed. Our contributions include:
- Development of the PathoTransformer model using transformer-based deep learning techniques.
- Creation of a comprehensive dataset combining the DeepPredictor dataset with additional pathogen sequences from the NCBI and ENA databases.
- Establishment of a pipeline for robust evaluation of model performance, utilizing metrics such as the F1 score to assess accuracy against various sequence perturbations.


Download the dataset from 

https://drive.google.com/file/d/1d3A8EQm6VZqnLGJq_Qnv1S07nTiJSBF2/view?usp=drive_link

Install the package:

```bash
pip install -r requirements.txt
```