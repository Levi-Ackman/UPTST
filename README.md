# Research Paper Code README
## This repository contains the experimental code for the paper titled "Leveraging Herpangina Data to Enhance Hospital-level Prediction of Hand-Foot-and-Mouth Disease Admissions Using UPTST."

## Usage
### UPTST Model
Navigate to the hfmd_uptst folder.

Modify the data_path in the "run.py " file to point to your dataset folder.

Run the code using the following command:

python run_scripts.py

After running, generate a CSV file of the results with:

python f1.py

The results will be saved as "uptst_results.csv."

Check the training results in the "test_dict " folder and view the prediction curves in the "visual" folder.

Please note that this process may take some time and requires a GPU.

### Other Models
If you wish to reproduce results from other models mentioned in the paper, follow a similar process in the hfmd_others folder.

### Exchange Dataset
To reproduce UPTST results on the exchange dataset:

Navigate to the ex_uptst folder.

Modify the data_path in the run.py file to point to your dataset folder.

Run the code using the following command:

bash scripts.sh

After running, check the training results in the "test_dict " folder and view the prediction curves in the "visual" folder.

### Customization
Feel free to explore each folder and select configurations of interest in the corresponding scripts folder. You can use bash commands to reproduce individual processes.

Enjoy your research and experimentation!
