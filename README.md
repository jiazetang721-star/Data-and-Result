# Robust e-waste recycling facility location problem with decision-dependent uncertain customer behavior
This public repository includes detailed descriptions of the data sources and individual results for each instance in the paper Robust e-waste recycling facility location problem with decision-dependent uncertain customer behavior. The parameter generation setting is described in Section 5.1 Experiment setup. Some Additional details are provided in Description.txt in each folder.

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
| b.npy | $kappa$, the size of uncertainty set |
| CC_point.npy | location of candidate CCs |
| RF_point.npy | location of candidate RFs |
| customer_point.npy | location of customer sites|
| theta.npy | nominal recycling rate when ignoring decision dependence |
