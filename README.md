## Adversarial Attacks On FHE Encrypted Models

#### CNN_SUITE
Contains a quantized, then compiled VGG11 model with weights, based on examples by Concrete ML
* VGG11 model with quantized weights (CIFAR-10)
* FGSM attack
* PDG attack
* Metrics (confusion matrix)

#### LLM_SUITE
Contains a finetuned, then partially encrypted GPT2 model (supports encryption for any set of attention modules), again based on examples by Concrete ML
* GPT2 model finetuned for the SST2 sentiment classification task ([download weights here](https://drive.google.com/drive/folders/1zZCbSyZzW_pPDLlGcCTLSJj7f7jrMarr?usp=sharing)), place in ./LLM_SUITE/sst2_gpt2 (replacing files if desired)
* Imperceptible NLP attacks based on [\[Boucher et al. 2021\]](https://arxiv.org/pdf/2106.09898.pdf)
* Metrics (accuracy, precision, recall, cross entropy)