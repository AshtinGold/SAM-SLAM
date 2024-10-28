# SAM-SLAM - Towards online volumetric open-vocabulary mapping

Author: King Hang, Wong

Abstract:
We propose a framework that enables online, open-vocabulary querying within a volumetric SLAM setting. Open-vocabulary language fields facilitate complex robot-scene interactions, which are crucial for practical robot deployment. Many novel view synthesis methods require optimizing the entire scene before integrating an associated language field. Our approach leverages the online reconstruction properties of neural SLAM to enable high-fidelity, open-vocabulary visual-language queries simultaneously with map reconstruction.

We utilize an explicit 3D Gaussian Splatting representation, which offers advantages such as easy editability, fast rendering speeds, and prevention of catastrophic forgetting—a common issue in neural implicit representations. To address memory explosion problems caused by naively splatting CLIP embedding vectors into each Gaussian point, we learn low-dimensional, scene-wise compressed embeddings represented by additional Gaussian channels. Our hierarchical method paves the way for online, queryable open-vocabulary semantic fields with high-fidelity, editable geometry.
