# Replication Package  
## Literate Programming with LLMs  

This repository contains the replication package for experiments on literate programming using large language models (LLMs).  

---

## How to Reproduce the Experiments  

### 1. Set up environment variables  
Create a `.env` file in the root directory with the following content:  

```bash
OPENAI_API_KEY='<your_openai_api_key>'
HUGGINGFACE_API_KEY='<your_huggingface_api_key>'
```

### 2. Run the Experiments  

1. **Generate embeddings**  
   - Ensure your `.env` file is correctly set up with valid API keys.  
   - Run the embedding generation script (replace `script_name.py` with the actual script name):  
     ```bash
      python combined_embedding.py \
        --model code7b \
        --device cuda \
        --train data/train.csv \
        --test data/test.csv \
        --desc data/desc.csv \
        --vis data/vis.csv
     ```

2. **Open the experiment notebooks**  
   - Navigate to the root folder.  
   - Open the notebook corresponding to the experiment you wish to reproduce.  

3. **Process the data**  
   - Follow the steps provided in the notebook.  
   - The notebooks will guide you through loading the embeddings, running the analysis, and reproducing the experimental results.  


### Reproduction Notes  

- Not all embeddings are included in this repository because the files are too large to upload to GitHub.  
- If you plan to generate embeddings yourself, ensure that you have **sufficient storage capacity**.  
  - Approximately **1 TB of storage** is expected to store all the data. 
