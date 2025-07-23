# Unsupervised Sign Gloss Segmentation from Continuous Sign Language Videos

## Project Overview

[cite\_start]This project presents a fully unsupervised pipeline for segmenting gloss-level units from continuous Indian Sign Language (ISL) videos[cite: 10]. [cite\_start]The primary objective is to discover meaningful gloss boundaries without relying on any labeled data [cite: 15][cite\_start], a crucial capability for low-resource sign languages where annotated datasets are scarce[cite: 15, 21].

[cite\_start]The system processes raw video data through a multi-stage pipeline involving 3D keypoint extraction [cite: 11][cite\_start], robust normalization [cite: 11][cite\_start], modality-specific autoencoder learning [cite: 12][cite\_start], hierarchical clustering [cite: 13][cite\_start], and Hidden Markov Model (HMM) sequence modeling[cite: 14]. [cite\_start]The ultimate goal is to infer latent gloss-level segmentations [cite: 14][cite\_start], providing a foundation for downstream tasks like sign-to-text translation and video summarization[cite: 25].

[cite\_start]**Developed by:** Partha Mete [cite: 2]

## Features & Key Contributions

  * [cite\_start]**Fully Unsupervised Pipeline**: Achieves gloss segmentation without requiring any frame-level or gloss-aligned labels[cite: 15, 46].
  * [cite\_start]**3D Keypoint Extraction**: Utilizes MediaPipe Holistic to extract comprehensive 3D pose, left hand, and right hand keypoints per frame[cite: 11, 68, 69, 70, 71].
  * [cite\_start]**Robust Keypoint Normalization**: Implements custom normalization and rotation-alignment techniques to ensure scale and orientation invariance of skeletal data[cite: 11].
  * [cite\_start]**Multi-Stream Latent Representation Learning**: Trains three separate autoencoders (for pose, left hand, and right hand) to learn compact, low-dimensional latent features[cite: 12, 48, 161], capturing modality-specific motion patterns.
  * **Hierarchical Clustering for Symbol Generation**:
      * [cite\_start]Applies K-Means clustering on the individual latent codes (pose, left hand, right hand)[cite: 13, 247, 248, 249, 265, 266, 267].
      * [cite\_start]Combines these cluster IDs into unique triplets[cite: 251, 347].
      * [cite\_start]Performs a second-level K-Means reclustering on these triplets (9870 unique triplets) to generate a discrete vocabulary of 512 observation symbols, suitable for HMM input[cite: 13, 252, 350, 351].
  * **Hidden Markov Model (HMM) for Sequence Modeling**:
      * [cite\_start]Trains a Categorical HMM using the Baum-Welch algorithm on the generated symbol sequences[cite: 14, 546].
      * [cite\_start]Infers gloss-like state sequences via Viterbi decoding [cite: 14, 547][cite\_start], demonstrating unsupervised segmentation of repeated glosses[cite: 852].
  * [cite\_start]**Designed for Low-Resource Languages**: Particularly suited for scenarios like ISL where large annotated corpora are unavailable[cite: 15, 51].

## Technologies Used

  * **Python**: The core programming language.
  * [cite\_start]**OpenCV (`cv2`)**: For video processing and frame handling[cite: 74].
  * [cite\_start]**MediaPipe (`mediapipe`)**: For robust 3D pose and hand landmark detection[cite: 11, 39, 68].
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

[cite\_start]This project is primarily implemented within a single Jupyter Notebook: `AMLCODE.ipynb`[cite: 22, 57]. It is highly recommended to run this notebook in a Google Colab environment due to its computational demands and reliance on Google Drive for data and model storage.

**Before you begin:**

  * **Dataset**: The project expects video data to be present in a specific Google Drive path (e.g., `/content/drive/MyDrive/data_composite_01/data/`). You will need to upload your video files to this location or modify the `video_dir` variable in the notebook to point to your data. [cite\_start]The dataset consists of 352 unlabeled sign language videos[cite: 22, 57].
  * **Google Drive Mount**: The notebook includes a cell to mount your Google Drive: `from google.colab import drive; drive.mount('/content/drive')`. Ensure this cell is run successfully.

**Steps to run the project:**

1.  **Open the Notebook**:

      * Go to [Google Colab](https://colab.research.google.com/).
      * Click `File > Upload notebook` and select `AMLCODE.ipynb` from your cloned repository.

2.  **Run Cells Sequentially**:
    The notebook is structured to be executed cell by cell, following the pipeline stages.

      * **Initial Setup**: Run the `pip install` commands at the beginning of the notebook to ensure all necessary libraries are installed in the Colab environment.
      * **Keypoint Extraction & Normalization**:
          * [cite\_start]The first few code blocks handle MediaPipe keypoint extraction from videos and their subsequent normalization[cite: 67]. [cite\_start]This step processes each video frame, extracts 3D pose and hand keypoints [cite: 69, 70, 71][cite\_start], and saves them in a `.pkl` file (e.g., `/content/drive/MyDrive/all_transformed_keypoints.pkl`)[cite: 92]. This can be time-consuming depending on your dataset size.
          * Visualization cells are included to help you inspect the extracted and normalized skeletons.
      * **Autoencoder Training**:
          * [cite\_start]This section defines and trains three separate Autoencoders: one for pose, one for the left hand, and one for the right hand[cite: 161].
          * [cite\_start]The models learn compact 16-dimensional latent representations[cite: 188, 192].
          * Trained model weights are saved to your Google Drive (e.g., `/content/drive/MyDrive/autoencoder_models/`).
          * Visualizations show reconstructed skeletons to assess reconstruction quality.
      * **KMeans Clustering**:
          * The trained autoencoders are used to extract latent vectors for all frames.
          * [cite\_start]K-Means clustering is applied to these latent vectors for each modality (pose, left hand, right hand) to assign initial cluster IDs[cite: 247, 248, 249, 263].
          * [cite\_start]These triplet cluster IDs are then re-clustered (from 9870 unique triplets) into a smaller, fixed vocabulary of 512 observation symbols[cite: 252, 350, 351].
          * [cite\_start]Elbow plots are generated to help determine optimal cluster numbers[cite: 268].
          * Visualizations show how different cluster centers represent distinct poses or hand shapes.
      * **Hidden Markov Model (HMM)**:
          * [cite\_start]The reclustered symbol sequences are used to train a Categorical Hidden Markov Model (HMM) using the Baum-Welch algorithm[cite: 14, 546].
          * [cite\_start]The Viterbi algorithm is then used to decode the most probable latent state sequence for each video[cite: 14, 547], effectively segmenting the continuous sign language into "gloss-like" units.
          * Visualizations display the inferred latent state timelines for videos, highlighting segments of similar motion.

**Important Considerations:**

  * **Resource Management**: Running the entire notebook, especially keypoint extraction and HMM training on a large dataset, can consume significant computational resources (RAM, GPU). Colab's free tier might have limitations.
  * **Path Configuration**: Ensure all file paths (for input videos, saved `.pkl` files, and model checkpoints) are correctly configured to point to your Google Drive or local directories as needed.
  * **Sequential Execution**: Do not skip cells unless you are certain about their dependencies. The pipeline is designed for sequential execution.

## Project Pipeline

The project follows a modular, multi-stage unsupervised learning pipeline:

1.  **3D Keypoint Extraction**:

      * [cite\_start]**Tool**: MediaPipe Holistic [cite: 68]
      * [cite\_start]**Output**: Raw 3D coordinates (x, y, z) and visibility for 33 pose landmarks, 21 left hand landmarks, and 21 right hand landmarks per frame[cite: 69, 70, 71].
      * [cite\_start]**Process**: Videos are read frame-by-frame using `cv2.VideoCapture()` [cite: 74][cite\_start], landmarks are extracted, and frames with missing data are discarded[cite: 75]. [cite\_start]Coordinates are normalized by image width and height[cite: 76].

2.  **Keypoint Normalization and Alignment**:

      * **Purpose**: To make keypoint representations invariant to signer's size, position, and orientation.
      * **Method**: Custom normalization and rotation-alignment functions are applied to center and scale the pose and hand skeletons, aligning them to a canonical "upright" orientation.
      * [cite\_start]**Output**: Normalized 3D keypoints stored in a consolidated `.pkl` file[cite: 92].

3.  **Modality-Specific Autoencoder Training**:

      * [cite\_start]**Purpose**: To learn compressed, low-dimensional latent representations for each body part[cite: 159, 160].
      * [cite\_start]**Architecture**: Three separate feed-forward autoencoders (one for pose, one for left hand, one for right hand)[cite: 161].
          * [cite\_start]Pose AE: Input (27D) -\> Hidden Layers (64, 32) -\> Latent (16D) using Sigmoid activations[cite: 188, 189, 190].
          * [cite\_start]Hand AEs: Input (63D) -\> Hidden Layers (128, 64) -\> Latent (16D) using ReLU activations[cite: 192].
      * [cite\_start]**Training**: Trained using Mean Squared Error (MSE) loss [cite: 195] [cite\_start]with Adam optimizer (`lr=1e-3`) [cite: 194] [cite\_start]and `ReduceLROnPlateau` scheduler (factor=0.5, patience=50) [cite: 198, 199] for stable convergence. [cite\_start]Early stopping is triggered after 50 stagnant epochs[cite: 200].

4.  **Hierarchical K-Means Clustering**:

      * [cite\_start]**First Level**: Latent vectors from each autoencoder are clustered independently using K-Means [cite: 263] [cite\_start](e.g., 50 clusters for pose, 100 for each hand)[cite: 248, 249, 250, 265, 266, 267]. [cite\_start]Cluster counts are selected using the elbow method[cite: 268].
      * [cite\_start]**Triplet Formation**: Per-frame cluster IDs are combined into unique triplets ($c\_{pose}, c\_{left}, c\_{right}$)[cite: 251, 347].
      * [cite\_start]**Second Level**: These unique triplets (total 9870) are re-clustered into a smaller, fixed vocabulary of 512 discrete observation symbols[cite: 252, 350, 351], which serve as the HMM's input.

5.  **Hidden Markov Model (HMM) Training and Inference**:

      * [cite\_start]**Model**: Categorical HMM[cite: 14].
      * [cite\_start]**Training**: The HMM is trained on the sequences of 512-dimensional observation symbols [cite: 351, 541, 545] [cite\_start]using the Baum-Welch algorithm[cite: 512, 546].
      * [cite\_start]**Inference**: The Viterbi algorithm is applied to each video's symbol sequence to infer the most probable underlying hidden state sequence[cite: 487, 547]. These hidden states are hypothesized to correspond to gloss-level segments.

## Results & Visualizations

The project's output includes various visualizations that demonstrate the effectiveness of each pipeline stage:

  * [cite\_start]**Skeletal Visualizations**: 2D and 3D plots of raw, normalized, and reconstructed keypoints, allowing visual inspection of data quality and autoencoder performance[cite: 576, 577].
  * [cite\_start]**Elbow Method Plots**: Graphs showing the inertia vs. number of clusters for latent spaces, used to determine optimal K-Means parameters[cite: 268, 269, 271, 272, 273].
  * [cite\_start]**Cluster Center Visualizations**: Plots of representative skeletons for various cluster centers, illustrating the distinct poses and hand shapes learned by the clustering process[cite: 320, 322, 339, 340].
  * [cite\_start]**HMM Latent State Timelines**: Visual timelines for individual videos showing the sequence of inferred hidden states, which represent the system's unsupervised segmentation of gloss-like units[cite: 587, 603, 625, 639, 665, 679, 703, 718, 741, 752, 778, 791]. [cite\_start]Examples demonstrate how similar glosses across different videos are mapped to similar latent state patterns[cite: 582, 622, 660, 700, 735, 774].
  * **Dominant State Analysis**: Bar charts illustrating the most common dominant latent states (based on longest continuous runs) across the entire dataset, providing insights into frequently occurring sign components.

## Limitations

  * [cite\_start]**Data Scarcity**: The dataset of 352 videos (approx. 75k valid data points) is relatively small for training a complex HMM with 210869 free scalar parameters, potentially leading to degenerate or overfitted models[cite: 815, 816, 817].
  * [cite\_start]**Absence of Ground Truth**: Due to the absence of gloss-aligned labels, true segmentation accuracy could not be quantitatively evaluated[cite: 818].
  * [cite\_start]**Clustering Instability**: Errors in K-Means cluster assignment can propagate into reclustered symbols and HMM input[cite: 819].
  * [cite\_start]**HMM Assumptions**: The first-order Markov and output independence assumptions of standard HMMs may not fully capture the temporal complexity of sign language[cite: 820].
  * [cite\_start]**Static HMM Structure**: The number of hidden states in the HMM is fixed and must be chosen heuristically[cite: 821].

## Future Work & Possible Extensions

  * [cite\_start]**Larger and Labeled Datasets**: Incorporating gloss-aligned datasets like RWTH-PHOENIX would enable supervised training and objective benchmarking[cite: 827].
  * [cite\_start]**Advanced Sequential Models**: Exploring more sophisticated deep sequential models like Bi-LSTMs, GRUs, or Transformers instead of HMMs to capture richer temporal dependencies[cite: 828].
  * [cite\_start]**Hybrid Architectures**: Developing hybrid models where autoencoder latent embeddings serve as input to a Bi-LSTM or Transformer, followed by a classifier to learn gloss boundaries or segment tags[cite: 829].
  * [cite\_start]**Multimodal Fusion**: Integrating additional modalities such as RGB-based visual features or audio (if available) alongside skeletal keypoints for richer representations[cite: 830].
  * [cite\_start]**Online Segmentation**: Adapting the pipeline for real-time gloss segmentation on streaming sign language input[cite: 831]. [cite\_start]The overall goal is to make the system usable for continuous ISL transcription with minimal supervision[cite: 832].

-----

**Developed by Partha Mete**
