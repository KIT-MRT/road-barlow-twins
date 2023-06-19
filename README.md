# Road Barlow Twins
Self-supervised pre-training method for HD map assisted motion prediction.

## Overview
![Model architecture](assets/road-bralow-twins.png "Model architecture")

Left: Pre-training and fine-tuning objectives. Pre-training: Maximize the similarity of moderately augmented views of HD maps. Fine-tuning: Motion prediction with six trajectory proposals per agent.

Right: Proposed DNN model for motion prediction with seperate encoders for map data and past agent trajectories.

## Getting started  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1vC5lqRVicGsmx8bkSlxrH4Tm9cTUr4e9?usp=sharing)

Open the Colab notebook above for a demo of our REDMotion model. The demo shows how to create a dataset, run inference, and visualize the predicted trajectories.

## Prepare waymo open motion prediction dataset
Register and download the dataset from [here](https://waymo.com/open).
Clone [this repo](https://github.com/kbrodt/waymo-motion-prediction-2021) and use the prerender script as described in the readme.

### Acknowledgement
The code in this repo builds up-on the work by [Konev et al.](https://arxiv.org/abs/2206.02163).
