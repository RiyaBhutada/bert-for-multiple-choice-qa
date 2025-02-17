# Leveraging LLM for Multiple-Choice Question Answering using

### 🎯 Goal
The **Leveraging LLM for MCQ Answering** project explores the Transformer architecture, focusing on fine-tuning a pre-trained BERT model for cybersecurity-related multiple-choice question answering. The study involves dataset preprocessing, model training, and evaluation while addressing challenges such as overfitting and computational limitations.

### 🌟 Features
- **Transformer Model Fine-Tuning:** Applied fine-tuning techniques on pre-trained BERT and DistilBERT models for multiple-choice QA and sequence classification tasks.
- **Cybersecurity Dataset Utilization:** Processed and analyzed the CyberMetric dataset containing 10,000 expert-verified cybersecurity questions.
- **Custom Tokenizer Enhancement:** Extended BERT’s vocabulary with 1,500 new cybersecurity-related terms extracted from the web.
- **Evaluation Metrics:** Implemented Exact Match (EM), F1 Score, and calibration metrics to assess model performance.
- **Overfitting Analysis & Mitigation:** Investigated model generalization issues and experimented with early stopping techniques.

### 🛠 What did I do?
1. **Dataset:**
   - **Yelp Reviews Case Study:** Used a large-scale text classification dataset to understand model fine-tuning.
   - **CyberMetric Dataset:**
     - Open-source dataset with 10,000 cybersecurity-related MCQs.
     - Data collected from standards, certifications, research papers, and expert sources.
     - JSON format preprocessing for multiple-choice question answering.
2. **Preprocessing:**
   - Tokenized text using BERT tokenizer.
   - Expanded vocabulary with cybersecurity terms using BeautifulSoup, regex, and html2text.
   - Reformatted MCQ dataset for both multiple-choice QA and sequence classification tasks.
3. **Model Training:**
   - Fine-tuned **BERTForMultipleChoice** from Hugging Face’s transformers library.
   - Experimented with **DistilBERT** for sequence classification due to computational constraints.
   - Defined training arguments, including evaluation strategy, epochs, and early stopping mechanisms.
4. **Evaluation:**
   - Assessed model accuracy, calibration, and robustness.
   - Identified overfitting through increasing evaluation loss despite decreasing training loss.

### 🖥 System Components
- **Pre-trained Models:** BERT, DistilBERT
- **Libraries:** PyTorch, Hugging Face Transformers, Scikit-Learn, Numpy, BeautifulSoup, requests, sys, Regex, html2text, accelerate, evaluate, datasets
- **Computational Resources:** Google Colab (GPUs), local Python environment

### 🚀 Use Cases
- **Cybersecurity Training:** AI-powered MCQ answering for professionals preparing for certifications.
- **Automated Question Answering:** Leveraging LLMs to enhance interactive learning experiences.
- **Adaptive Learning Platforms:** Personalized quizzes based on user knowledge gaps.

### 🔮 Challenges & Future Scope
- **Memory and Compute Constraints:** Larger models like Mistral-7B require substantial GPU resources.
- **Multiple-Choice Bias:** Investigate model sensitivity to answer order.
- **Robust Fine-Tuning:** Experiment with parameter optimization, data augmentation, and parallelization techniques.
- **Deployment & Pipelines:** Automate model training and deployment for real-world applications.

### ✅ Conclusion
The **Leveraging LLM for MCQ Answering** project demonstrates the capabilities of fine-tuning transformer models for specialized domains like cybersecurity. Addressing computational challenges and optimizing model performance will enhance future applications in adaptive learning and AI-driven education tools.

