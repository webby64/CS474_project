# CS474_project

## Source code for CS474: Text Mining course at KAIST. 

The project focuses on implementing text mining concepts and techniques that we learned in CS474 - Text Mining class. Queries such as ‘Top 10’ or ‘Best 10’ are one of the most common search phrases on the Korea Herald News Dataset. 

We implemented k-means, DBSCAN clustering techniques, document level relation extraction with spacy and stanza, dimensionality reduction with PCA and UMAP, cosine-based document similarity, topic modelling using BERTopic and embedding with different pretrained models.

Make sure you've installed `python --version > 3.5` and `java`.

To install Java Runtime Environment from OpenJDK 11

    sudo apt install default-jre 
    
To run the codes, you need to install the dependencies first.
    pip install -r requirements.txt

There are three tasks in this project.

- *Task 1:* Issue Trend Analysis
- *Task 2:* On-issue Event Tracking
- *Task 3:* Related-issue Event Tracking

To run the codes for these three tasks, we've prepared the following shell script.

    ./run.sh


