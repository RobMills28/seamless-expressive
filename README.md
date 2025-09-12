# SeamlessExpressive Benchmark Testbed

### Academic Context

This repository contains the source code for the experimental testbed used to evaluate Meta AI's **Seamless Expressive** model. This system was developed as the benchmark for comparison in a Master's thesis at the University of Nottingham.

- **Thesis Title:** *Identity-Preserved Multimodal Speech Translation: A Validated Modular Framework for Preserving Vocal Characteristics and Emotional Congruence Across Languages*
- **Author:** Robert Mills

### License & Usage

This implementation uses Meta's SeamlessExpressive model under their non-commercial license. For academic/research use only.

### Note

This repository contains the experimental testbed portion of the thesis research. The full implementation of the primary system being evaluated, the **Modern Cascaded Framework**, can be found here: [https://github.com/RobMills28/modern-cascaded-framework](https://github.com/RobMills28/modern-cascaded-framework).

---

## 🚀 Running the Project

This project is composed of a frontend application and a backend API that wraps the Seamless Expressive model. The backend requires a Conda environment named `seamless-expressive`.

### 1. Backend Setup

- **Navigate to the `backend` directory:**
  ```bash
  cd backend

Activate the correct Conda environment:
conda activate seamless-expressive

Run the FastAPI application:
python app.py

The backend API will now be running on http://localhost:8004.

### 2. Frontend Setup

Navigate to the frontend directory:
cd frontend

Install dependencies:
npm install

Start the development server:
npm start

The frontend application will now be accessible at http://localhost:3001.
