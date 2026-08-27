# Robust e-waste recycling facility location problem with decision-dependent uncertain customer behavior
This public repository includes detailed descriptions of the data sources and individual results for each instance in the paper Robust e-waste recycling facility location problem with decision-dependent uncertain customer behavior. The parameter generation setting is described in Section 5.1 Experiment setup.

## Data and Description

| File | Description |
|---|---|
| I.npy | number of candidate CCs |
| J.npy | number of candidate RFs |
| K.npy | number of customer sites |
| fc.npy | cost for building CCs |
| fr.npy | cost for building RFs | 
| c.npy | unit transportation cost from CCs to RFs | 
| r.npy | unit profit margin for processing e-waste in RFs | 
| fp.npy | unit penalty for unprocessed e-waste in CCs, including acquisition price and outsourcing cost |
| Capacity.npy | production capacity of RFs |
| u.npy | utility of customers patronizing CCs | 
| d.npy | potential recyclable e-waste amount at customer sites | 
| b.npy | $\kappa$, the size of uncertainty set |
| CC_point.npy | location of candidate CCs |
| RF_point.npy | location of candidate RFs |
| customer_point.npy | location of customer sites|
| theta.npy | nominal recycling rate when ignoring decision dependence |

### Data in Section 5.4 & Section 3 of supplementary material
The basic data of these instances are the same as in Common data, and the changes are described in the paper:

Section 5.4.1: κ (the size of uncertainty set) increases from 0.00 to 0.20 in steps of 0.01

Section 5.4.2: norm type changes as L1 / L2 / L_inf 

Section 3.1 of suppementary material: d_max decreases from 80 km to 40 km in steps of 5 km, which influence u.npy

Section 3.2 of suppementary material: drop-out utility increases from 6 to 14 in
increments of 1, which influence u.npy

Section 3.3 of suppementary material: unit transportationi cost per (ton * km) increases from 80 yuan to 140 yuan in increments of 10, which influence c.npy

## Results

Established CCs and RFs are recorded in .txt files. We also provide data related to each figure in the paper in .xlsx files.

### Results in Section 5.3

DDP.pkl, DDN.pkl, DDC-10.pkl, DDC-15.pkl, DIN.pkl, DIC-10.pkl and DIC-15.pkl record second stage solutions and worst-case uncertainty realization, i.e., transportation w, outsourcing h and worst-case choice probability p.
