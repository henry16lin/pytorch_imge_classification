# Image classification with pytorch
just practice and record for future using  
where torchsummary is from https://github.com/sksq96/pytorch-summary


## Usage  
1. create data folder with structure like ```training_data```  
2. you can have seperated folder structure on training and validation set otherwise ``train.py`` will split val data from training set  
3. set training parameter in ```train.py```   
4. select loss and network structure you want  
5. run ```train.py```   
6. given data folder and model.pt, `inference.py` will inference all image in the folder
