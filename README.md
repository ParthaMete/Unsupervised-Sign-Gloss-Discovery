# Unsupervised Sign Gloss Segmentation from Continuous Sign Language Videos

## Project Overview

This project presents a fully unsupervised pipeline for segmenting gloss-level units from continuous Indian Sign Language (ISL) videos. The primary objective is to discover meaningful gloss boundaries without relying on any labeled data, a crucial capability for low-resource sign languages where annotated datasets are scarce.

The system processes raw video data through a multi-stage pipeline involving 3D keypoint extraction, robust normalization, modality-specific autoencoder learning, hierarchical clustering, and Hidden Markov Model (HMM) sequence modeling. The ultimate goal is to infer latent gloss-level segmentations, providing a foundation for downstream tasks like sign-to-text translation and video summarization.

**Developed by:** Partha Mete

## Features & Key Contributions

  * **Fully Unsupervised Pipeline**: Achieves gloss segmentation without requiring any frame-level or gloss-aligned labels.
  * **3D Keypoint Extraction**: Utilizes MediaPipe Holistic to extract comprehensive 3D pose, left hand, and right hand keypoints per frame.
  * **Robust Keypoint Normalization**: Implements custom normalization and rotation-alignment techniques to ensure scale and orientation invariance of skeletal data.
  * **Multi-Stream Latent Representation Learning**: Trains three separate autoencoders (for pose, left hand, and right hand) to learn compact, low-dimensional latent features, capturing modality-specific motion patterns.
  * **Hierarchical Clustering for Symbol Generation**:
      * Applies K-Means clustering on the individual latent codes (pose, left hand, right hand).
      * Combines these cluster IDs into unique triplets.
      * Performs a second-level K-Means reclustering on these triplets (9870 unique triplets) to generate a discrete vocabulary of 512 observation symbols, suitable for HMM input.
  * **Hidden Markov Model (HMM) for Sequence Modeling**:
      * Trains a Categorical HMM using the Baum-Welch algorithm on the generated symbol sequences.
      * Infers gloss-like state sequences via Viterbi decoding, demonstrating unsupervised segmentation of repeated glosses.
  * **Designed for Low-Resource Languages**: Particularly suited for scenarios like ISL where large annotated corpora are unavailable.

## Technologies Used

  * **Python**: The core programming language.
  * **OpenCV (`cv2`)**: For video processing and frame handling.
  * **MediaPipe (`mediapipe`)**: For robust 3D pose and hand landmark detection.
  * **NumPy (`numpy`)**: For numerical operations and array manipulation.
  * **PyTorch (`torch`)**: For building and training autoencoders.
  * **Scikit-learn (`sklearn`)**: For K-Means clustering.
  * **HMMlearn (`hmmlearn`)**: For Hidden Markov Model training and inference.
  * **Plotly (`plotly`) / Matplotlib (`matplotlib`)**: For 3D and 2D visualizations of skeletons and cluster centers.
  * **Tqdm (`tqdm`)**: For progress bars during long processing steps.
  * **Pandas (`pandas`)**: For data handling and CSV exports.
  * **Joblib (`joblib`)**: For saving and loading HMM models.
  * **Pickle (`pickle`)**: For serializing and deserializing Python objects (data and models).

## Installation

To set up and run this project, follow these steps:

1.  **Clone the repository**:

    ```bash
    git clone https://github.com/ParthaMete/Unsupervised-Sign-Gloss-Discovery.git
    cd Unsupervised-Sign-Gloss-Discovery
    ```

2.  **Create a virtual environment (recommended)**:

    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install the required libraries**:
    The project uses several external libraries. You can install them using `pip`:

    ```bash
    pip install opencv-python mediapipe numpy==1.26.4 torch torchvision scikit-learn hmmlearn plotly tqdm pandas jupyter
    ```

      * **Note on `numpy`**: A specific version (`1.26.4`) is specified to ensure compatibility with `mediapipe` and other libraries.
      * **Google Colab**: The provided `AMLCODE.ipynb` notebook is designed to be run in Google Colab, which handles most environment setups and dependencies. If running locally, ensure all dependencies are correctly installed.

## Usage

This project is primarily implemented within a single Jupyter Notebook: `AMLCODE.ipynb`. It is highly recommended to run this notebook in a Google Colab environment due to its computational demands and reliance on Google Drive for data and model storage.

**Before you begin:**

  * **Dataset**: The project expects video data to be present in a specific Google Drive path (e.g., `/content/drive/MyDrive/data_composite_01/data/`). You will need to upload your video files to this location or modify the `video_dir` variable in the notebook to point to your data. The dataset consists of 352 unlabeled sign language videos.
  * **Google Drive Mount**: The notebook includes a cell to mount your Google Drive: `from google.colab import drive; drive.mount('/content/drive')`. Ensure this cell is run successfully.

**Steps to run the project:**

1.  **Open the Notebook**:

      * Go to [Google Colab](https://colab.research.google.com/).
      * Click `File > Upload notebook` and select `AMLCODE.ipynb` from your cloned repository.

2.  **Run Cells Sequentially**:
    The notebook is structured to be executed cell by cell, following the pipeline stages.

      * **Initial Setup**: Run the `pip install` commands at the beginning of the notebook to ensure all necessary libraries are installed in the Colab environment.
      * **Keypoint Extraction & Normalization**:
          * The first few code blocks handle MediaPipe keypoint extraction from videos and their subsequent normalization. This step processes each video frame, extracts 3D pose and hand keypoints, and saves them in a `.pkl` file (e.g., `/content/drive/MyDrive/all_transformed_keypoints.pkl`). This can be time-consuming depending on your dataset size.
          * Visualization cells are included to help you inspect the extracted and normalized skeletons.
      * **Autoencoder Training**:
          * This section defines and trains three separate Autoencoders: one for pose, one for the left hand, and one for the right hand.
          * The models learn compact 16-dimensional latent representations.
          * Trained model weights are saved to your Google Drive (e.g., `/content/drive/MyDrive/autoencoder_models/`).
          * Visualizations show reconstructed skeletons to assess reconstruction quality.
      * **KMeans Clustering**:
          * The trained autoencoders are used to extract latent vectors for all frames.
          * K-Means clustering is applied to these latent vectors for each modality (pose, left hand, right hand) to assign initial cluster IDs.
          * These triplet cluster IDs are then re-clustered (from 9870 unique triplets) into a smaller, fixed vocabulary of 512 observation symbols.
          * Elbow plots are generated to help determine optimal cluster numbers.
          * Visualizations show how different cluster centers represent distinct poses or hand shapes.
      * **Hidden Markov Model (HMM)**:
          * The reclustered symbol sequences are used to train a Categorical Hidden Markov Model (HMM) using the Baum-Welch algorithm.
          * The Viterbi algorithm is then used to decode the most probable latent state sequence for each video, effectively segmenting the continuous sign language into "gloss-like" units.
          * Visualizations display the inferred latent state timelines for videos, highlighting segments of similar motion.

**Important Considerations:**

  * **Resource Management**: Running the entire notebook, especially keypoint extraction and HMM training on a large dataset, can consume significant computational resources (RAM, GPU). Colab's free tier might have limitations.
  * **Path Configuration**: Ensure all file paths (for input videos, saved `.pkl` files, and model checkpoints) are correctly configured to point to your Google Drive or local directories as needed.
  * **Sequential Execution**: Do not skip cells unless you are certain about their dependencies. The pipeline is designed for sequential execution.

## Project Pipeline

The project follows a modular, multi-stage unsupervised learning pipeline:

1.  **3D Keypoint Extraction**:

      * **Tool**: MediaPipe Holistic
      * **Output**: Raw 3D coordinates (x, y, z) and visibility for 33 pose landmarks, 21 left hand landmarks, and 21 right hand landmarks per frame.
      * **Process**: Videos are read frame-by-frame using `cv2.VideoCapture()`, landmarks are extracted, and frames with missing data are discarded. Coordinates are normalized by image width and height.

2.  **Keypoint Normalization and Alignment**:

      * **Purpose**: To make keypoint representations invariant to signer's size, position, and orientation.
      * **Method**: Custom normalization and rotation-alignment functions are applied to center and scale the pose and hand skeletons, aligning them to a canonical "upright" orientation.
      * **Output**: Normalized 3D keypoints stored in a consolidated `.pkl` file.

3.  **Modality-Specific Autoencoder Training**:

      * **Purpose**: To learn compressed, low-dimensional latent representations for each body part.
      * **Architecture**: Three separate feed-forward autoencoders (one for pose, one for left hand, one for right hand).
          * Pose AE: Input (27D) -\> Hidden Layers (64, 32) -\> Latent (16D) using Sigmoid activations.
          * Hand AEs: Input (63D) -\> Hidden Layers (128, 64) -\> Latent (16D) using ReLU activations.
      * **Training**: Trained using Mean Squared Error (MSE) loss with Adam optimizer (`lr=1e-3`) and `ReduceLROnPlateau` scheduler (factor=0.5, patience=50) for stable convergence. Early stopping is triggered after 50 stagnant epochs.

4.  **Hierarchical K-Means Clustering**:

      * **First Level**: Latent vectors from each autoencoder are clustered independently using K-Means (e.g., 50 clusters for pose, 100 for each hand). Cluster counts are selected using the elbow method.
      * **Triplet Formation**: Per-frame cluster IDs are combined into unique triplets ($c\_{pose}, c\_{left}, c\_{right}$).
      * **Second Level**: These unique triplets (total 9870) are re-clustered into a smaller, fixed vocabulary of 512 discrete observation symbols, which serve as the HMM's input.

5.  **Hidden Markov Model (HMM) Training and Inference**:

      * **Model**: Categorical HMM.
      * **Training**: The HMM is trained on the sequences of 512-dimensional observation symbols using the Baum-Welch algorithm.
      * **Inference**: The Viterbi algorithm is applied to each video's symbol sequence to infer the most probable underlying hidden state sequence. These hidden states are hypothesized to correspond to gloss-level segments.

## Results & Visualizations

The project's output includes various visualizations that demonstrate the effectiveness of each pipeline stage:

  * **Skeletal Visualizations**: 2D and 3D plots of raw, normalized, and reconstructed keypoints, allowing visual inspection of data quality and autoencoder performance.
  * **Elbow Method Plots**: Graphs showing the inertia vs. number of clusters for latent spaces, used to determine optimal K-Means parameters.
  * **Cluster Center Visualizations**: Plots of representative skeletons for various cluster centers, illustrating the distinct poses and hand shapes learned by the clustering process.
  * **HMM Latent State Timelines**: Visual timelines for individual videos showing the sequence of inferred hidden states, which represent the system's unsupervised segmentation of gloss-like units. Examples demonstrate how similar glosses across different videos are mapped to similar latent state patterns.
 

## Limitations

  * **Data Scarcity**: The dataset of 352 videos (approx. 75k valid data points) is relatively small for training a complex HMM with 210869 free scalar parameters, potentially leading to degenerate or overfitted models.
  * **Absence of Ground Truth**: Without frame-level gloss annotations, quantitative evaluation of segmentation accuracy (e.g., F1-score, Jaccard index) is not possible.
  * **Clustering Instability**: K-Means is sensitive to initialization and can lead to sub-optimal clusters, which might propagate errors through the pipeline.
  * **HMM Assumptions**: The first-order Markov and output independence assumptions of standard HMMs may not fully capture the temporal complexity of sign language.
  * **Static HMM Structure**: The number of hidden states in the HMM is fixed and must be chosen heuristically.

## Future Work & Possible Extensions

  * **Larger and Labeled Datasets**: Incorporating larger, gloss-aligned datasets (e.g., RWTH-PHOENIX) would enable quantitative evaluation and potentially supervised fine-tuning of the learned representations.
  * **Advanced Sequential Models**: Exploring more sophisticated deep sequential models like Bi-LSTMs, GRUs, or Transformers instead of HMMs to capture richer temporal dependencies.
  * **Hybrid Architectures**: Developing hybrid models where autoencoder latent embeddings serve as input to a Bi-LSTM or Transformer, followed by a classifier to learn gloss boundaries or segment tags.
  * **Multimodal Fusion**: Integrating additional modalities such as RGB-based visual features or audio (if available) alongside skeletal keypoints for richer representations.
  * **Online Segmentation**: Adapting the pipeline for real-time gloss segmentation on streaming sign language input. The overall goal is to make the system usable for continuous ISL transcription with minimal supervision.

-----

**Developed by Partha Mete**
