# HetGL2R
Learning to rank critical road segments via heterogeneous graphs with origin-destination flow integration


## Framework of the proposed HetGL2R method

<img width="4939" height="2382" alt="GF" src="https://github.com/user-attachments/assets/7702b6e9-71c6-4ba9-afca-8e5d58a2edfa" />


## Joint random walk sampling procedure

<img width="3207" height="1424" alt="HetGWalk23" src="https://github.com/user-attachments/assets/f60136a6-7b0e-4dee-b893-64383d1e4a17" />


## Citation

If this code is useful for your research, please cite our paper:

```bibtex
@article{HetGL2R,
title = {Learning to rank critical road segments via heterogeneous graphs with origin-destination flow integration},
journal = {Information Processing & Management},
volume = {63},
number = {6},
pages = {104702},
year = {2026},
issn = {0306-4573},
doi = {https://doi.org/10.1016/j.ipm.2026.104702},
url = {https://www.sciencedirect.com/science/article/pii/S0306457326000932},
author = {Ming Xu and Jinrong Xiang and Zilong Xie and Xiangfu Meng},
keywords = {Learning to rank, Heterogeneous graph, Random walk, Ranking, Road networks},
abstract = {Existing learning-to-rank methods for road networks often fail to incorporate origin-destination (OD) flows and route information, limiting their ability to model long-range spatial dependencies. To address this gap, we propose HetGL2R, a heterogeneous graph learning framework for ranking road-segment importance. HetGL2R builds a tripartite graph that unifies OD flows, routes, and network topology, and further introduces attribute-guided graphs that elevate node attributes into explicit nodes to model functional similarity. A heterogeneous joint random walk algorithm (HetGWalk) jointly samples both graph types to generate context-rich node sequences. These sequences are encoded using a Transformer to learn embeddings that capture long-range structural dependencies induced by OD flows and route configurations, as well as functional associations derived from attribute similarity. Finally, a listwise ranking strategy with a KL-divergence loss evaluates and ranks segment importance. Experiments on three SUMO-generated simulated networks of different scales show that, against state-of-the-art methods, HetGL2R achieves average improvements of approximately 7.52%, 4.40% and 3.57% in ranking performance.}
}
```

## Reproducibility and Data Availability Statement 
﻿
The code provided in this repository was organized from the research code used during the experimental stage of the paper. Since this code was developed at an early stage of my programming practice, its engineering quality and standardization may still be limited. Therefore, reproducing the results in different environments may require additional effort. <br>
﻿ <br>
All experimental data used in this work were generated from the laboratory simulation platform. Due to data management and usage restrictions, these data cannot be publicly released at this time. <br>
﻿ <br>
If you need the original road network data files or require assistance with code reproduction, please contact me at: <br>
﻿ <br>
xiangjinrong117@gmail.com
