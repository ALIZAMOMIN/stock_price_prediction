
  <h1>Nifty 50 Stock Movement Predictor</h1>

  <p>
    This project predicts whether the Nifty 50 index will move <strong>up or down</strong> in the next few days using a <strong>Random Forest Classifier</strong>. The app is built with a <strong>Tkinter GUI</strong> and trained using historical stock data.
  </p>

  <h2>Features</h2>
  <ul>
    <li>Predicts future market direction (up/down) based on previous data</li>
    <li>Uses Random Forest with hyperparameter tuning and cross-validation</li>
    <li>Displays confusion matrix for model evaluation</li>
    <li>Visualizes predicted vs actual results with bar graphs</li>
    <li>Easy-to-use Tkinter GUI for interactive usage</li>
  </ul>

  <h2>Technologies Used</h2>
  <ul>
    <li>Language: Python</li>
    <li>GUI: Tkinter</li>
    <li>Model: Random Forest Classifier (Scikit-learn)</li>
    <li>Data: Historical Nifty 50 stock prices</li>
    <li>Visualization: Matplotlib</li>
  </ul>

  <h2>Project Structure</h2>
  <pre>
├── main.py                # Contains training, prediction, and GUI logic
├── nifty50.csv            # Dataset (if available locally)
├── requirements.txt       # List of dependencies
├── README.html            # This file
  </pre>

  <h2>Requirements</h2>
  <p>Install the required libraries using:</p>
  <pre><code>pip install -r requirements.txt</code></pre>

  <h2>How to Run</h2>
  <ol>
    <li>Ensure <code>nifty50.csv</code> is present or fetch it using the commented code in <code>data_fetcher.py</code>.</li>
    <li>Run <code>main.py</code>.</li>
    <li>Use the GUI to enter a date and predict the next 5 days of market movement.</li>
  </ol>

  <h2>Output Example</h2>
  <p>The GUI displays predictions and actual market movement with a bar chart.</p>

  <h2>Dataset Source</h2>
  <p>
    You can download historical Nifty 50 data from <a href="https://finance.yahoo.com/quote/%5ENSEI/history" target="_blank">Yahoo Finance - NSEI</a>.
  </p>

  <h2>Status</h2>
  <p>Completed and tested</p>

Below is a sample output from the Nifty 50 Stock Movement Predictor:

![confusion_matrix](images/confusion_matrix.png)

The graph and interface below shows the predicted vs actual prices:

![interface](images/interface.png)

  <h2>Contact</h2>
  <p>For any issues, feel free to raise an issue or reach out.</p>
