# MNIST Real-Time Digit Recognition

A decoupled web application for recognizing handwritten digits using a Deep Learning model trained on the MNIST dataset. The project separates the visualization layer from the inference logic to ensure scalability and performance.

## 🏗 Architecture

The application follows a client-server architecture:

![MNIST_FASTAPI](./mnist_architecture.png)

1.  **Frontend (Streamlit):** A user-friendly interface providing a canvas for drawing digits. It converts the drawing into a pixel array and sends it to the API.
2.  **Backend (FastAPI):** A fast, asynchronous API that receives the image data, performs preprocessing (normalization, resizing), and runs the inference.
3.  **Model:** A Convolutional Neural Network (CNN) trained on the MNIST dataset.

## 🛠 Tech Stack

* **Frontend:** Streamlit, Streamlit-Drawable-Canvas
* **Backend:** FastAPI, Uvicorn
* **Machine Learning:** PyTorch / TensorFlow (CNN Model)
* **Data Processing:** NumPy, PIL (Python Imaging Library)

## 🚀 Getting Started

### Prerequisites

* Python 3.8+
* Pip

### Installation

1.  Clone the repository and install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Start the Backend (API)**
    Run the FastAPI server on port 8000:
    ```bash
    uvicorn api.main:app --reload --port 8000
    ```

3.  **Start the Frontend (UI)**
    In a new terminal, launch the Streamlit app:
    ```bash
    streamlit run frontend/app.py
    ```

4.  **Usage**
    Open your browser at `http://localhost:8501`, draw a number, and click "Predict"!
