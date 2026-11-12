# Causal Explainable AI for Trustworthy Intelligent Systems

A Python and Streamlit project that demonstrates baseline machine-learning
modelling on a synthetic student-performance dataset, enriched with
Explainable AI (XAI) and Causal Analysis, wrapped in an interactive
multi-page dashboard.

## Project Description
This project bridges the gap between predictive power and causal understanding.
By starting with baseline ML models, we introduce SHAP for explainability and 
Causal Graphs for understanding relationships. We also provide a Counterfactual 
Simulator to test what-if scenarios and understand the boundaries of correlations 
versus true causations.

## Features (Current Completed)
- **Dataset Generation & Analysis:** Auto-generates a realistic synthetic dataset (1,000 records) representing student features (study hours, attendance, sleep, etc.) and a PASS/FAIL target.
- **Machine Learning Models:** Baseline Logistic Regression and Random Forest models with automatic evaluation and best-model selection based on F1 Score.
- **Explainable AI:** Global feature importance from the model and SHAP, plus interactive individual prediction explanations with natural language narratives.
- **Causal Analysis:** Interactive visualization of the assumed causal structure via NetworkX, highlighting the difference between correlation and causation.
- **Counterfactual Simulator:** A what-if engine to interactively tweak inputs and instantly compare the original versus modified predictions and confidences.

## Future Improvements
- DoWhy causal inference
- Causal effect estimation
- Fairness and bias analysis
- Trustworthiness scoring
- Advanced counterfactual explanations
- Robustness testing

## Technologies Used
- **Python 3**
- **Streamlit** for the web dashboard and UI
- **Scikit-Learn** for machine learning models and preprocessing
- **SHAP** for explainable AI
- **NetworkX** for causal graph modeling
- **Pandas & NumPy** for data manipulation
- **Matplotlib & Seaborn** for data visualization

## Project Structure
```
causal-xai-project/
│
├── data/
│   └── student_data.csv        # Auto-generated synthetic dataset
├── models/                     # Best model + preprocessing artefacts
├── src/
│   ├── __init__.py
│   ├── causal_analysis.py      # Causal graph logic and descriptions
│   ├── counterfactual.py       # Counterfactual simulation engine
│   ├── data_preprocessing.py   # Loading, encoding, scaling, splitting
│   ├── explainability.py       # SHAP global and local explanations
│   ├── llm_service.py          # Google Gemini API backend for XAI Agent
│   ├── model_training.py       # Training, evaluation, model selection
│   └── utils.py                # Dataset generation & helpers
│
├── app.py                      # Streamlit dashboard
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation Instructions
1. Clone or download the project.
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate        # macOS / Linux
   # venv\Scripts\activate         # Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Google Gemini API Configuration (Explainable AI Agent)
The Explainable AI Agent backend uses the official **Google Gen AI Python SDK** (`google-genai`) with models supported on the free tier (default: `gemini-2.5-flash`, with fallbacks to `gemini-2.0-flash` and `gemini-1.5-flash`).

### 1. Obtain an API Key
Generate a free API key from Google AI Studio:
- Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
- Click **Create API Key** and copy your key.

### 2. Configure Your Key
Choose either of the two methods below:

#### Option A: Local Environment Variable (Recommended for terminal / local development)
```bash
# macOS / Linux (in terminal or add to ~/.zshrc or ~/.bashrc)
export GEMINI_API_KEY="your-gemini-api-key-here"

# Windows (Command Prompt)
set GEMINI_API_KEY=your-gemini-api-key-here

# Windows (PowerShell)
$env:GEMINI_API_KEY="your-gemini-api-key-here"
```

#### Option B: Streamlit Secrets (Recommended for Streamlit apps & deployment)
Create a `.streamlit/secrets.toml` file in the project root (a template is available in `.streamlit/secrets.toml.example`):
```toml
# .streamlit/secrets.toml
GEMINI_API_KEY = "your-gemini-api-key-here"
```
*(Note: `.streamlit/secrets.toml` is included in `.gitignore` to prevent leaking keys.)*

### 3. Test API Connection
Run the connection diagnostic test anytime to verify connectivity:
```bash
python -m src.llm_service
```
Or run the automated unit tests:
```bash
python test_llm_service.py
```

## How to Run
Once installed and activated, start the Streamlit application:
```bash
streamlit run app.py
```
The dashboard will open automatically in your browser. Navigate through the sidebar to explore the different modules.

## License
This project is for educational and research purposes.
