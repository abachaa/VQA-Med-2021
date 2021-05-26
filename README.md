# VQA-Med-2021
-------------

Website: https://www.imageclef.org/2021/medical/vqa 

Mailing list: https://groups.google.com/d/forum/imageclef-vqa-med

Tasks: Visual Question Answering (VQA) and Visual Question Generation (VQG) in the medical domain. 

Results of the VQA-Med-2021 challenge on crowdAI: 

- VQA task: https://www.aicrowd.com/challenges/imageclef-2021-vqa-med-vqa 
- VQG task: https://www.aicrowd.com/challenges/imageclef-2021-vqa-med-vqg


Data: 
--------------

VQA Data:  
- Training set: We provided the VQA-Med 2020 training data including 4,500 radiology images and 4,500 question-answer pairs (https://www.aicrowd.com/challenges/imageclef-2020-vqa-med-vqa) 
- Validation set: Consists of 500 radiology images and associated questions/answers about Abnormality  
- Test set: 500 radiology images and 500 questions about abnormality. Participants were tasked with generating the answers based on the visual content of the images.  

The VQA-Med dataset was also used the ImageCLEF Caption & Concept Prediction Task: https://www.imageclef.org/2021/medical/caption 

VQG Data:
- The VQG 2021 validation set contains 200 questions associated with 85 radiology images. 

- The VQG 2021 test set includes 100 radiology images. Participants were tasked with generating distinct questions that are relevant to the visual content of the images. 

Evaluation Metrics
------------------

Accuracy: We used an adapted version of the accuracy metric from the general domain VQA task that considers exact matching of a participant provided answer and the ground truth answer.

BLEU: We used the BLEU metric to capture the similarity between a system-generated answer and the ground truth answer. 

The following preprocessing is applied before running the evaluation metrics on each answer: (i) each answer is converted to lower-case, and (ii) all punctuations are removed and the answer is tokenized to individual words. 

Code: 

Reference
----------

If you use the VQA-Med 2021 dataset, please cite our paper: "Overview of the VQA-Med Task at ImageCLEF 2021: Visual Question Answering and Generation in the Medical Domain". 
Asma Ben Abacha, Mourad Sarrouti, Dina Demner-Fushman, Sadid A. Hasan, and Henning Müller. CLEF 2021 Working Notes. 

@Inproceedings{ImageCLEF-VQA-Med2021,
author = {Asma {Ben Abacha} and Mourad Sarrouti and Dina Demner-Fushman and Sadid A. Hasan and Henning M\"uller},
title = {Overview of the VQA-Med Task at ImageCLEF 2021: Visual Question Answering and Generation in the Medical Domain},
booktitle = {CLEF 2021 Working Notes},
series = {{CEUR} Workshop Proceedings},
year = {2021},
volume = {},
publisher = {CEUR-WS.org},
pages = {},
month = {September 21-24},
address = {Bucharest, Romania}
}
    
    
Contact Information
--------------------

Asma Ben Abacha: asma.benabacha AT gmail.com   https://sites.google.com/site/asmabenabacha/ 
