# Project Structure and Workflows

This document explains the three main flowcharts that describe how the macromill sentiment classification project is structured and how data and control flow from entry point to prediction results.

---

## 1. Main Data Flow Diagram

The main data flow of the code is driven by the CLI wrapper. Execution starts at **`work/run_cli.py`**, which adds the `src` directory to `sys.path` and invokes **`macromill_sentiment.cli.main()`** with the command-line arguments. The CLI uses `argparse` to expose three subcommands: **`train`**, **`eval`**, and **`predict-local`**.

- **Data layer**: All paths that need raw or preprocessed text use **`data/load.py`** → **`load_imdb_csv()`** to read the IMDB CSV (columns `review` and `sentiment`) and **`data/preprocess.py`** → **`clean_text()`** to strip HTML, unescape entities, lowercase, and normalize whitespace according to **`PreprocessConfig`**.

- **Training**: The **`train`** command either calls **`models/train.py`** → **`train_model()`** for TF-IDF models (LR or Linear SVM) or **`models/roberta_train.py`** → **`train_roberta()`** for the transformer. The sklearn path uses **`models/registry.py`** → **`build_model()`** to construct a `Pipeline` (TfidfVectorizer + classifier), then **`sklearn.model_selection.train_test_split`** for an 80/20 split, fits the model, and persists it via **`artifacts/io.py`** → **`save_artifact()`** (joblib). The RoBERTa path loads data, preprocesses with the same `clean_text()`, does its own train/val/test split, fine-tunes with PyTorch, and saves the model and tokenizer under the artifact directory.

- **Evaluation**: The **`eval`** command detects RoBERTa by checking for `config.json` and `.bin`/`.safetensors` in the artifact directory. For TF-IDF models it builds **`EvalConfig`** and calls **`models/evaluate.py`** → **`evaluate_model()`**, which loads the artifact with **`load_artifact()`**, reloads and preprocesses the CSV, optionally uses artifact metadata for split seed/test size, runs predictions, and computes accuracy, F1, confusion matrix, ROC-AUC, latency, throughput, and artifact size. For RoBERTa it calls **`models/roberta_eval.py`** → **`evaluate_roberta()`**, which uses the same split parameters from artifact metadata, loads **`RoBERTaPredictor`**, runs batched inference on the held-out test set, and computes the same metrics. Results are written to the requested **`output_json`** (e.g. `eval.json`).

- **Local prediction**: The **`predict-local`** command again branches on whether the artifact is RoBERTa (config present). For RoBERTa it instantiates **`models/roberta_predict.py`** → **`RoBERTaPredictor(artifact_dir)`** and calls **`predict(text)`**. For TF-IDF it uses **`load_artifact()`** to load the joblib model, **`clean_text()`** with **`PreprocessConfig`**, and then **`model.predict([text])`** (and **`predict_proba`** if available). The CLI prints a JSON object with `label` and optional `scores`.

- **API service**: The FastAPI app in **`api/main.py`** uses a lifespan handler that gets **`api/service.py`** → **`get_model_service()`** (singleton **`ModelService`**) and preloads the default model. **`ModelService`** holds an `artifacts_dir` (e.g. `work/artifacts`), maps names like `tfidf_lr`, `tfidf_linearsvm`, `roberta` to subdirs, and in **`load_model()`** either joblib-loads a `.joblib`/`.pkl` file or, for RoBERTa, loads tokenizer and **`RobertaForSequenceClassification`** via Hugging Face. Endpoints **`GET /health`**, **`GET /models`**, **`GET /models/{model_name}`** expose status and metadata; **`POST /predict`** validates `model_name`, calls **`service.predict(text, model_name, preprocess)`**, which preprocesses with an internal **`_clean_text()`** then **`_predict_sklearn()`** or **`_predict_roberta()`**, and returns sentiment, confidence, and probabilities.

So the main data flow is: **CLI entry** → **command dispatch** → **data load/preprocess** and/or **model build/train or load** → **artifact save/load** → **evaluation metrics or prediction output**; the API is a separate entry that uses the same **ModelService** and artifact directories to serve predictions over HTTP.

![Data Flow Diagram](./Data%20Flow%20Diagram.png)

---

## 2. Training Pipeline Flow

The training pipeline flow describes how raw data becomes trained model artifacts. It starts with **IMDB Dataset.csv** (50K movie reviews with `review` and `sentiment` columns). The data is passed through a shared preprocessing chain implemented in **`data/preprocess.py`**: **strip HTML tags** (regex), **unescape HTML entities** (e.g. `&amp;` → `&`), **lowercase** the text, and **normalize whitespace** (collapse runs of spaces and trim). The result is clean text used consistently for training, evaluation, and inference.

For the **train/test split**, the sklearn path in **`models/train.py`** uses **`train_test_split`** with a fixed seed and `test_size=0.2`, stratified on labels, yielding 80% training and 20% test. The RoBERTa path in **`models/roberta_train.py`** first holds out 20% as test, then splits the remainder into train/val (e.g. 90%/10%) for validation during training.

**Vectorization** applies only to the TF-IDF models: **`models/registry.py`** builds a **`Pipeline`** whose first step is **TfidfVectorizer** with `ngram_range=(1, 2)`, `min_df=5`, `max_features=50_000`, and optional stop words. The training set text is transformed by this pipeline; the classifier step ( **Logistic Regression** or **LinearSVC** ) is then fit on these vectors. The **RoBERTa-base** path does not use TF-IDF; it uses **AutoTokenizer** and **AutoModelForSequenceClassification** (Hugging Face), with a **SentimentDataset** that tokenizes with a max length (e.g. 256), and training is done with **DataLoader**, **AdamW**, and a linear scheduler.

**Output**: The sklearn pipeline is saved via **`artifacts/io.py`** → **`save_artifact()`** as a single **`.joblib`** file (e.g. `tfidf_lr_v3/model.joblib`). RoBERTa is saved with **`model.save_pretrained()`** and **`tokenizer.save_pretrained()`** into the artifact directory (e.g. **`pytorch_model.bin`** or **`model.safetensors`** plus **`config.json`** and tokenizer files). Training metadata (paths, seeds, sizes, preprocess flags) is written to **`model_meta.json`** in the same directory. So the training pipeline is: **raw CSV** → **preprocess** → **split** → **vectorize (TF-IDF) or tokenize (RoBERTa)** → **train classifier/transformer** → **save artifact and meta**.

![Training Pipeline Flow](./Training%20Pipeline%20Flow.png)

---

## 3. API Prediction Flow

The API prediction flow describes what happens when a client sends **POST /predict** with a body like **`{ "text": "...", "model_name": "..." }`**. The FastAPI app in **`api/main.py`** receives the request and the **`predict`** endpoint calls **`get_model_service().predict(text, model_name, preprocess)`** in **`api/service.py`**.

**Request validation**: The service checks that **`model_name`** is one of the known models (e.g. `tfidf_lr`, `tfidf_linearsvm`, `roberta`); if not, the API returns 400. The text is used as-is (optionally preprocessed inside the service).

**ModelService** then ensures the model is loaded (**`load_model(model_name)`**), which either joblib-loads the pipeline from the artifact dir or, for RoBERTa, loads the tokenizer and **RobertaForSequenceClassification** from disk. If **preprocess** is true, the input text is cleaned with the same logic as training (strip HTML, lowercase, normalize whitespace) via an internal **`_clean_text()`** using **`PreprocessConfig`**.

**Model selection**: The service branches on **`model_name == "roberta"`**. If **Yes**, it calls **`_predict_roberta()`**: the cleaned text is **tokenized** with the RoBERTa tokenizer (truncation, max_length=512), passed through the model in **`torch.no_grad()`** to get logits, **softmax** is applied to get class probabilities, and the predicted label is **positive** or **negative** with confidence and a probability dict. If **No**, it calls **`_predict_sklearn()`**: the cleaned text is passed as a single-element list to **`model.predict([text])`** (the sklearn pipeline applies **TF-IDF vectorization** internally, then the **classifier predict**); if the model has **`predict_proba`**, probabilities are read and mapped to class names for **confidence** and **probabilities** in the response.

**HTTP response**: The endpoint returns a **PredictResponse** containing the (possibly truncated) text, **sentiment** (positive/negative), **confidence**, **probabilities**, **model_used**, and **preprocessing** flag. So the API prediction flow is: **POST /predict** → **validate** → **load model** → **preprocess text** → **branch roberta vs sklearn** → **tokenize+forward+softmax** or **TF-IDF+predict+proba** → **JSON response**.

![API Prediction Flow](./API%20Prediction%20Flow.png)
