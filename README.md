# Querying Answers from CVs for a Job Description

## Project Description

## Project Description

The primary goal of this project is to create a tool for interacting with job descriptions and CVs using natural language processing (NLP) techniques. This tool is designed to query CVs in relation to specific job description requirements.

Identifying the most relevant CVs that match the specific requirements of a job description can be a time-consuming and challenging task. This project aims to streamline and automate this process, making it more efficient for both recruiters and job seekers.

Key features and functionalities of the project include:

### 1. Natural Language Processing (NLP)

The project leverages advanced NLP techniques to understand the content of both job descriptions and CVs. It can identify key skills, qualifications, and experience mentioned in the job posting, as well as extract corresponding information from CVs.

### 2. Requirement Analysis

By analyzing job descriptions and CVs side by side, the system can determine how well a CV matches the requirements of a specific job. This analysis takes into account factors such as relevant skills, years of experience, and educational background.


### 3. Automation, Time and Resource Efficiency

The project's automation capabilities greatly reduce the manual effort required in the initial CV screening process. It helps in filtering and prioritizing CVs, allowing recruiters to focus their attention on the most promising candidates.

## Installation

Follow these steps to set up and run the project on your local machine:


1. Clone this GitHub repository to your local machine using the following command:

   ```bash
   git clone https://github.com/Amit2416/Langchain_CV_Experiment.git

2. Install requirements
   pip install -r requirements.txt

3. Download pretrained model: Lamini-T5-738M
   git lfs install
   git clone https://huggingface.co/MBZUAI/LaMini-T5-738M

4. Create folder for JD and CV and keep your documents

   "docs/CV"
   "docs/job_description"

5. Run python ingest_jd_cv.py - It will create the embeddings for JD and CV and will store in locally. 

6. Write generalized prompts (prompt_jd.py) - to query a job description
   ![Prompts](https://github.com/Amit2416/Langchain_CV_Experiment/blob/main/Prompts.JPG)


8. Run python summarization_jd.py - It will generate reponses for the prompts. These responses will be used to query a CV.
   ![Responses from JD](https://github.com/Amit2416/Langchain_CV_Experiment/blob/main/Responses_from_JD.JPG)

10. Run python summarization_cv.py - It will generate responses after being queried by job description sections as from Step 7.
    ![Responses from CV](https://github.com/Amit2416/Langchain_CV_Experiment/blob/main/Responses_from_CV.JPG)







