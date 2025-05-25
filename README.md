#READ ME

### 1 Usage

#### 1.1 Setting up environment

Step 1. To use the system, first create a virtual environment,

    conda create --name <env_name> python=3.9

Step 2. Then activate your conda environment with,

    conda activate <env_name>

Step 3. Install the following packages with,

     pip install pyyaml torch torchvision scikit-learn matplotlib pillow numpy pandas

#### 1.2 Training the models

Our code is split up into two sections for our final model. The finetuned resnet50 architecture and the captions LSTM 

#### 1.2.1 Training the Resnet50 model

Step 1. cd resnet50

Step 2. python3 train.py

This training will produce best_model.pth which is the first model used, it sits at 94.5 MB in size

#### 1.2.2 Training the LSTM

Step 1. 

Step 2. 

#### 1.3 Running the model for inference



## guidelines:
- variable names should be in camel case (variableName)
- function names should be in pascal case (FunctionName)
- global variables should be in all caps snake case (GLOBAL_VARIABLE)
- global variables should be treated as read-only
- hard coded values should be global variables
- variable names should not be shortened (use dataframe instead of df. Some exceptions apply ie i and j for counters)
- variable names and function names should be descriptive at a glance
- avoid function side effects (a method that is meant to print out information should not alter outside variables)
- independent code blocks should be seperated into individual functions (if you think you need a comment to explain a code block, you should extract it into a seperate function)
