
# Face Recognition System

This repository contains the code for a **Face Recognition System** built using Python, OpenCV, and machine learning techniques. The system includes functionalities for creating a dataset, training a model, and recognizing faces in real time using a webcam.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## Features
- **Face Detection**: Detects faces in images or via webcam in real time using Haar cascades.
- **Dataset Creation**: Captures face data from the webcam and stores it as a dataset for training.
- **Model Training**: Uses the dataset to train a machine learning model for recognizing faces.
- **Face Recognition**: Identifies and recognizes faces from live video or image input.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/KIRAN-KUMAR-K3/face-recognition-system.git
   ```
2. **Navigate to the project directory**:
   ```bash
   cd face-recognition-system
   ```
3. **Install the required dependencies**:
   Install the dependencies listed in `requirements.txt` using pip:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Create Dataset:
To create a dataset for training, run the `datasetCreator.py` script. This will capture images from your webcam:
   ```bash
   python datasetCreator.py
   ```

### 2. Train the Model:
Once the dataset is created, train the model by running the `train_model.py` script:
   ```bash
   python train_model.py
   ```

### 3. Recognize Faces:
After training the model, you can start face recognition by running the `app.py` script:
   ```bash
   python app.py
   ```

## Project Structure

```bash
face-recognition-system/
│
├── app.py                        # Main application script for face recognition
├── datasetCreator.py              # Script to create dataset by capturing face images
├── train_model.py                 # Script to train the face recognition model
├── haarcascade_frontalface_default.xml  # Haar Cascade classifier for face detection
├── requirements.txt               # List of dependencies required for the project
└── README.md                      # Project documentation (this file)
```

## Dependencies

The required libraries for this project can be found in `requirements.txt`. Main dependencies include:
- OpenCV
- Numpy
- Scikit-learn

To install these dependencies, run:
```bash
pip install -r requirements.txt
```

## Contributing
Contributions are welcome! If you'd like to improve this project, feel free to fork the repository and create a pull request.

1. Fork the repository.
2. Create a new feature branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add some feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
