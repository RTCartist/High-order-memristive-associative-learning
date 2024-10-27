# High-order-memristive-associative-learning
This repository contains circuits and code required to reproduce all results from the manuscript "High-Order Associative Learning Based on Memristive Circuits for Efficient Learning" (https://arxiv.org/abs/2410.16734).

## Repository Contents
1. Circuit: Contains circuit diagrams and PWL files to perform second-order and higher-order associative learning experiments.
2. MNN for Image Recognition: Includes the original code (main.py) and datasets.
3. SPICE Resources: Provides .sub and .asy files needed for circuit design.

## Required Software and Version:
1. LTspiceXVII
2. Python 3.9

## Python Dependencies:
1. numpy
2. matplotlib
3. opencv-python (cv2)
4. seaborn
5. os

## LTspice configuration instruction:
1. Download the binary file for LTspice XVII from http://ltspice.analog.com and execute the installation program. It is recommended to install the software in the directory 'C:\Program Files\LTC\LTspiceXVII' to facilitate ease of access for subsequent simulations.
2. Add the SPICE_models to the Library Search Path in LTspice XVII manually. (Detailed instructions can be found in https://uspas.fnal.gov/materials/17NIU/LTspiceXVII%20Installation.pdf.) Notice: in the first use, please modify the .lib instructions in the .asc to guarantee successful simulation in LTspice.

Contact information: shengb.wang@gmail.com or shuo_gao@buaa.edu.cn.