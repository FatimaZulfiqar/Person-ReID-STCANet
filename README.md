# Person ReID STCANet
This repository contain code of the paper 
### Multi-Camera Person Re-Identification using Spatiotemporal Context Modeling <br>
Fatima Zulfiqar, Usama Ijaz Bajwa, Rana Hammad Raza

## Abstract
Person Re-Identification (ReID) aims at identifying a person of interest (POI) across multiple non-overlapping cameras. The POI can either be in an image or a video sequence. Factors such as occlusion, variable viewpoint, misalignment, unrestrained poses, background clutter, etc. are the major challenges in developing robust, person ReID models. To address these issues, an attention mechanism that comprises local part/region aggregated feature representation learning is presented in this paper by incorporating long-range local and global context modeling. The part-aware local attention blocks are aggregated into the widely used modified pre-trained ResNet50 CNN architecture as a backbone employing two attention blocks i.e., Spatio-Temporal Attention Module (STAM) and Channel Attention Mod-ule (CAM). The spatial attention block of STAM can learn contextual dependencies between different human body parts/regions like head, upper body, lower body, and shoes from a single frame. On the other hand, the temporal attention modality can learn temporal con-textual dependencies of the same person’s body parts across all video frames. Lastly, the channel-based attention modality i.e., CAM can model semantic connections between the channels of feature maps. These STAM and CAM blocks are combined sequentially to form a unified attention network named as Spatio-Temporal Channel Attention Network (STCANet) that will be able to learn both short-range and long-range global feature maps respectively. Extensive experiments are carried out to study the effectiveness of STCANet on three image-based and two video-based benchmark datasets i.e. Market-1501, DukeMTMC-ReID, MSMT17, DukeMTC-VideoReID, and MARS. K-reciprocal re-ranking of gallery set is also applied in which the proposed network showed significant improvement over these datasets in comparison to state-of-the-art. Lastly, to study the generalizability of STCANet on unseen test instances, cross-validation on external cohorts is also applied that showed the robustness of the proposed model that can be easily deployed to the real world for practical applications.

Installation
---------------

Make sure `conda <https://www.anaconda.com/distribution/>`_ is installed.


    # cd to your preferred directory and clone this repo
    git clone https://github.com/FatimaZulfiqar/Person-ReID-STCANet

    # create virtual environment
    cd Person-ReID-STCANet/
    conda create --name personreid python=3.7
    conda activate personreid

    # install dependencies
    # make sure `which python` and `which pip` point to the correct path
    pip install -r requirements.txt

    # install torch and torchvision (select the proper cuda version to suit your machine)
    conda install pytorch torchvision cudatoolkit=9.0 -c pytorch
    
## Training 
1. Generate part-masks of image-based person ReID datasets using the code https://github.com/hyk1996/Single-Human-Parsing-LIP
2. Run jupyter notebook file Or
3. Run python train.py
4. For Testing run test.py

## Platform
This code was developed and tested with pytorch version 1.0.1. The experiments have been conducted on both Google Colab and Local Server.

## Acknowledgments
This code is based on the implementations of [**Deep person reID**](https://github.com/KaiyangZhou/deep-person-reid/tree/master/torchreid).

## Citation
If you use this code for your research, please cite our paper.

