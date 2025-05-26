
```markdown
# 🌿 Soil Image Classification Project

This repository contains solutions for two challenges:  
- **Challenge-1**: Soil Type Classification (Alluvial, Red, Black, Clay)  
- **Challenge-2**: Hybrid KMeans + CNN for Soil vs Non-Soil Classification

---


---

## 🏃 How to Run

### 🔹 Using Python Scripts
1️⃣ Navigate to the respective challenge folder:
```bash
cd challenge-1/src  # or challenge-2/src
```

2️⃣ Run Training:
```bash
python training.py
```

3️⃣ Run Inference:
```bash
python inference.py
```

4️⃣ (Optional) Run Evaluation:
```bash
python evaluate.py
```

---

### 🔹 Using Jupyter Notebooks
1️⃣ Navigate to the notebooks folder:
```bash
cd challenge-1/notebooks  # or challenge-2/notebooks
```

2️⃣ Open and run:
- `training.ipynb`: Train the model
- `inference.ipynb`: Make predictions

---

### 🔹 Generate Project Card
1️⃣ Navigate to the project-card notebook for metric generation in the cards folder inside docs:
```bash
cd challenge-1/docs/  # or challenge-2/docs/
```

2️⃣ Run:
- `project-card.ipynb`: Generates a visual project summary card

---

## 📝 Notes
- Ensure your Python environment is activated with required libraries (`tensorflow`, `scikit-learn`, `pandas`, etc.)
- All outputs (models, metrics, and submissions) are saved in the `data/` folder
- Generated metrics are stored in `docs/cards/ml-metric.json` for easy review
