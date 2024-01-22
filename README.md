# Voting Classifier App with Streamlit

A web application built using Streamlit for creating and visualizing a Voting Classifier with various datasets and machine learning algorithms.

<details>
  <summary>Click to see project screenshots</summary>
  <!-- Add your screenshots here -->
  ![Screenshot 1](url_to_screenshot_1)
  ![Screenshot 2](url_to_screenshot_2)
</details>

## Datasets Used
- XOR
- Dataset with Outliers
- Linearly Separable Dataset
- Two Spirals
- Concentric Circles

## Visualization
Click on the "Visualize" button to plot a scatterplot of the selected dataset.

## Voting Type
Choose between "Hard" and "Soft" voting for the classifier.

## Estimators
Choose multiple algorithms to include in the Voting Classifier:
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Gaussian Naive Bayes
- Support Vector Machine (SVM)
- Random Forest

## Parameter Selection
Specify parameters for the Voting Classifier (e.g., soft or hard voting) and click on the "Run Algorithm" button to see Voting Classifier accuracy and individual model accuracies.

## Initial DataFrame
A dataframe is shown at the start of the application to provide an overview of the data.

## Features and Parameter Effects
- **Voting Classifier Parameters:**
  - Users can explore and adjust parameters for the Voting Classifier itself, such as choosing between soft and hard voting. Observe how these choices influence the overall accuracy of the ensemble.

## How to Use
1. **Fork or Clone the Project:**
   ```
     git clone https://github.com/iamRahulB/Streamlit-voting-classifier.git
   ```
2. Navigate to the Project Directory:
   ```
   cd Streamlit-voting-classifier
   ```
3. Create a Virtual Environment (Optional but Recommended):
   ```
   python -m venv venv
   ```
4. Activate the Virtual Environment:
   1. On Windows:
     ```
     .\venv\Scripts\activate
     ```
   2. On macOS/Linux:
     ```
     source venv/bin/activate
5. Install the Required Dependencies:
   ```
   pip install -r requirements.txt
   ```
6. Run the Application Locally Using Streamlit:
   ```
   streamlit run app.py
   ```
   

