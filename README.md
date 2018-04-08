# Transcription Binding Prediction

Transcription is the process where a gene's DNA sequence is copied (transcribed) into an RNA molecule.
Transcription is a key step in using information from a gene to make a protein.

When a gene to be transcribed, the enzyme RNA polymerase, which makes a new RNA molecule from a
DNA template, must attach to the DNA of the gene. It attaches at a spot called the promoter. In human
RNA polymerase can attach to the promoter only with the help of proteins called basal transcription factors.
They are part of the cell's core transcription toolkit, needed for the transcription of any gene. Some
transcription factors activate transcription, but others can repress transcription.

The binding sites for transcription factors are often close to a gene's promoter. However, they can also be
found in other parts of the DNA, sometimes very far away from the promoter, and still affect transcription
of the gene. Binding of transcription factors to transcription factor binding sites (TFBSs) is key to the
mediation of transcriptional regulation. Information on experimentally validated functional TFBSs is
limited and consequently there is a need for accurate prediction of TFBSs for gene annotation and in
applications such as evaluating the effects of single nucleotide variations in causing disease.

Deep Neural Network implemented in TensorFlow and Python for predicting whether transcription  factors will bind
or not to given DNA sequence. 

Implemented architecture is a Convolutional Neural Network.

## Requirements 
- Python3
- Numpy
- Pandas
- TensorFlow
- Keras

To train model and predict for unknown data:
```buildoutcfg
python keras_cnn.py
```
