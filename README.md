# SA-GCN

#### A Spatial-temporal graph convolutional network with self-attention for city-level cellular network traffic prediction

#### 

A precise and prompt estimation of cellular network traffic is essential for improving user quality of experience.
However, there are many obstacles to accomplishing this aim because of the dynamic and complex structure of
spatial-temporal connections. We present unique multi-dimensional features fusion self-attention-based spatial-temporal
Graph Convolutional Networks (MF-SAGCN) to tackle these problems. This model improves the accuracy of cellular traffic
prediction by capturing dynamic spatial-temporal properties and unstable relationships in a synergistic and fusion way.
MF-SAGCN consists of two key modules: a spatial transformer and a temporal transformer. In the spatial domain, the
spatial transformer and the multi-self attention mechanism are utilized to capture static and dynamic spatial features,
respectively, and a gating mechanism seamlessly fuses the static and dynamic features. The temporal transformer utilizes
a self-attention mechanism in the temporal domain to capture the non-stationary temporal correlations in traffic data.
Then, a spatial-temporal transformer block consisting of multiple temporal and spatial transformers is connected by skip
connection to achieve the fusion of deep features of different dimensions. Ultimately, the local spatial-temporal
dependencies in the globally encoded features are mined by the designed densely connected convolution module. To
validate the effectiveness of MF-SAGCN, we conduct extensive experiments on a publicly available real cellular network
traffic dataset. The results show that MF-SAGCN has a competitive advantage over popular state-of-the-art methods.


#### requirement
requirements.txt



