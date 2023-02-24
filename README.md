#Multi-Camera Person Re-Identification using Spatiotemporal Context Modeling
## Abstract
Person Re-Identification (ReID) aims at identifying a person of interest (POI) across multiple non-overlapping cameras. The POI can either be in an image or a video sequence. Factors such as occlusion, variable viewpoint, misalignment, unrestrained poses, background clutter, etc. are the major challenges in developing robust, person ReID models. To address these issues, an attention mechanism that comprises local part/region aggregated feature representation learning is presented in this paper by incorporating long-range local and global context modeling. The part-aware local attention blocks are aggregated into the widely used modified pre-trained ResNet50 CNN architecture as a backbone employing two attention blocks i.e., Spatio-Temporal Attention Module (STAM) and Channel Attention Mod-ule (CAM). The spatial attention block of STAM can learn contextual dependencies between different human body parts/regions like head, upper body, lower body, and shoes from a single frame. On the other hand, the temporal attention modality can learn temporal con-textual dependencies of the same person’s body parts across all video frames. Lastly, the channel-based attention modality i.e., CAM can model semantic connections between the channels of feature maps. These STAM and CAM blocks are combined sequentially to form a unified attention network named as Spatio-Temporal Channel Attention Network (STCANet) that will be able to learn both short-range and long-range global feature maps respectively. Extensive experiments are carried out to study the effectiveness of STCANet on three image-based and two video-based benchmark datasets i.e. Market-1501, DukeMTMC-ReID, MSMT17, DukeMTC-VideoReID, and MARS. K-reciprocal re-ranking of gallery set is also applied in which the proposed network showed significant improvement over these datasets in comparison to state-of-the-art. Lastly, to study the generalizability of STCANet on unseen test instances, cross-validation on external cohorts is also applied that showed the robustness of the proposed model that can be easily deployed to the real world for prac-tical applications.
