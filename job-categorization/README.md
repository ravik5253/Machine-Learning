### Assignment

### Task
Make a classifier which takes in a job description and gives the department name for it.

*   Used a neural network model
*   Make use of a pre-trained Word Embeddings (GloVe)
*   Calculated the accuracy on a test set (data not used to train the model)

### DataSet

All the data required can be found in tha data folder

There are two sources for loading your training/test data

*   For *Job Description*:  
   **docs** folder contains around 1000 json files, each of which is a single job posting. You have to use the value of `description` field inside the `jd_information` field.

*   For *Job Department*:  
   **document_departments.csv** file contains the mapping of document id to department name where document id is the name of the corresponding file in docs folder.

