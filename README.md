# Clustering Service - Hands-On Assignment

## 🎯 Goal

Design and deliver an **end-to-end clustering service** that can be trained, queried for predictions, evaluated for performance, and incrementally updated with new data.

---

## 🧩 Tasks

### Task #1 - Model Inference Implementation

- Implement the missing methods in the `ElementsClustering` class inside `model.py`.
- Complete the implementation of the FastAPI app (`app.py`):
  - `POST /fit` route:  
    Receives a training set, fits the model, and saves it.
  - `POST /predict` route:  
    Receives validation data (one item at a time), loads the model, and returns the prediction.
- Run `main.py` to send requests to both `/fit` and `/predict` to demonstrate a full pipeline. Extend it as needed.

---

### Task #2 - Error Analysis

- Define a metric to evaluate the performance of the clustering model.
- Analyze the model's performance based on this metric.
- Suggest at least one way to improve the model’s performance.
- Implement your improvement and demonstrate that the metric has improved.

---

### Task #3 - Model Persistency

- Implement an `update` method for the model.
- This method should:
  - Accept a previously fitted model and new data.
  - Preserve cluster labels for repeated clusters.
  - Add new labels only for newly detected clusters.

---

## 🛠️ Guidelines

- Clone the repository locally.
- [Download the data](https://drive.google.com/drive/folders/1O4gbkJNW7IU8xcS2yA0Jmz_XlCRIP2cg?usp=drive_link) and save it in a location accessible to the model.
- Update this `README.md` file as needed with relevant instructions or documentation.

---

## 🚀 Submission Instructions

1. Create a new branch for your implementation.
2. Complete all tasks in this branch.
3. Open a Pull Request (PR) when your code is ready for review.
4. You may include additional artifacts in the repository (e.g., reports, visualizations, analysis notebooks).

---

## 🤖 AI Tool Usage

You are encouraged to use AI tools (e.g., Cursor, ChatGPT, Claude) to assist with the implementation.  
**However, you must understand and be able to explain any code or logic generated with their help.**

---

## 📬 Questions?

Don’t hesitate to reach out if you need any clarifications.

---

**Good luck!**