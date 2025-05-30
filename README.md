
## 0 MODEL SIZE

We ensemble two models for our classificaiton problem. They are found at:

    resnet50/best_model.pth # 94.5 MB
    Language Processing/model # 4.4 MB

All other models included are part of exploration

## 1 Usage

#### 1.1 Setting up environment

Step 1. To use the system, first create a virtual environment,

    conda create --name <env_name> python=3.9

Step 2. Then activate your conda environment with,

    conda activate <env_name>

Step 3. Install the following packages with,

     pip install pyyaml torch torchvision scikit-learn matplotlib pillow numpy pandas

#### 1.2 Training the models

Our code is split up into two sections for our final model. The finetuned resnet50 architecture and the captions LSTM 

#### 1.2.1 Training/Inferring Resnet50 model

Step 1. 
    
    cd resnet50

Step 2. 
    
    python3 train.py

This training will produce best_model.pth which is the first model used, it sits at 94.5 MB in size. It ran for 1 hour and 27 minutes when trained

Step 3.

    python3 infer.py

This inference stage produces image_output.csv in the root directory. This contains the labels produced by the resnet50 model.

#### 1.2.2 Training/Inferring Resnet50 LSTM

Step 1. Navigate to Language Processing/processing.ipynb 

Step 2. Run each code block. This will train, hyperparameter fine-tune an LSTM model. This model is stored at Language Processing/model It will then run the model and run produce image_output.csv. The model is 4.4 MB

#### 1.3 Producing the final output

Step 1. Navigate to the root directory and run:
    
    combine.py 
    
This will combine the captions_output.csv and image_output.csv to produce Predicted_labels.txt

Step 2. View the final Predicted_labels.txt

## 2 Remaining code base
The rest of the code base exists as experiments on our way to building the most accurate model.

LVLMs contains exploration

CNNs contains abalation studies, with other models. It was not used in the final result.