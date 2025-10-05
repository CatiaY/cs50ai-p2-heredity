# CS50’s Introduction to AI with Python

This project was completed as part of the course assignments.  
The following description is adapted from the original project specification.  

## Project 2: Heredity (Lecture 2 – Uncertainty)  

This project implements an AI that estimates genetic inheritance probabilities within a family.

Given information about individuals, their parents, and whether they exhibit a particular trait associated with a specific gene, the AI uses a **Bayesian Network** to model these relationships and infer both the probability distribution for each person’s genes and the likelihood of expressing the trait. The calculations are based on joint probabilities, which account for all possible combinations of gene and trait distributions across the family.


**Notes**
- The description above is adapted from the official project specification.
- Implemented using the course’s starter code, with modifications made to fulfill the project requirements.
- Completed as part of the CS50’s Introduction to AI with Python (2024 edition) coursework.


### How to Run

1. Run the program using the command `python heredity.py ./data/[family].csv`, where `[family]` is the name of the csv file that contains the family data.
