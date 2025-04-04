#import yfinance as yf 
import pandas as pd 
import os # for file handling
import tkinter as tk # gui
from tkinter import ttk # Themed Tkinter Widgets.
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt #for graph
from datetime import  timedelta # for date 
from sklearn.ensemble import RandomForestClassifier #model
from sklearn.model_selection import GridSearchCV, KFold,cross_val_score # for hyprparmeter tuning, cross validation
from sklearn.metrics import ConfusionMatrixDisplay 
import numpy as np 

# Load Nifty 50 Data
def load_data():
    if os.path.exists(r"C:\Users\aliza\Desktop\python_mini_project\to_git\nifty50.csv"):
        return pd.read_csv(r"C:\Users\aliza\Desktop\python_mini_project\to_git\nifty50.csv", index_col=0, parse_dates=True)
    #index_col=0 : take 1st column as index
    #parse_dates=True : parse the index as date... make time seris operation easy to perform
    '''
    else:
        nifty = yf.Ticker("^NSEI") # Nifty 50 ticker symbol
        data = nifty.history(period="max") #take all historical data
        data.to_csv("nifty50.csv")
        return data
    '''
        
# Prepare Data
nifty50 = load_data() #function call 
print(nifty50.head(2)) # Display the first few rows of the DataFrame
#nifty50.index = pd.to_datetime(nifty50.index, utc=True)
nifty50 = nifty50.loc["1998-01-01":].copy() 
#loc is used to access a group of rows and columns by labels or a boolean array.
#loc is primarily label based data selection. 
#iloc is primarily integer position based data selection.
nifty50 = nifty50[["Open", "Close"]].copy() # Select only the Open and Close columns
nifty50["Tomorrow"] = nifty50["Close"].shift(-1) #shift up 
nifty50["Target"] = (nifty50["Tomorrow"] > nifty50["Close"]).astype(int) #output as int 1:true , 0:false
nifty50 = nifty50.dropna() # Drop rows with NaN values

#(nifty50.head(5))

# Feature Engineering (new columns)
horizons = [2,5,60,250,1000]#  # Define the horizons for rolling averages
#horizons = [2,5,30,60,250] # Define the horizons for rolling averages
new_predictors = [] # Initialize an empty list to store new predictor columns
#print(f"Number of rows in nifty50: {len(nifty50)}") #4299
for horizon in horizons:
    
    rolling_averages = nifty50.rolling(horizon).mean()
    #print(rolling_averages.head(2)) # Display the first few rows of the rolling averages DataFrame
    ratio_column = f"Close_Ratio_{horizon}" # create dynamic column name>>eg close_atio_2
    #print(ratio_column)
    nifty50[ratio_column] = nifty50["Close"] / rolling_averages["Close"]
    #print(nifty50[ratio_column].head(2))# Display the new column
    trend_column = f"Trend_{horizon}"
    #print(trend_column)
    nifty50[trend_column] = nifty50.shift(1).rolling(horizon).sum()["Target"] #calc
    new_predictors += [ratio_column, trend_column]


#nifty50 = nifty50.dropna() # Drop rows with NaN values
(nifty50.head(5)) # Display the first few rows of the DataFrame


# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],          # Number of trees
    'max_depth': [None, 10, 20],              # Maximum depth of trees
    'min_samples_split': [10, 20, 50],        # Minimum samples to split a node
    'min_samples_leaf': [1, 5, 10],           # Minimum samples in a leaf node
         
}






# Initialize the RandomForestClassifier
rf = RandomForestClassifier(random_state=42)


# K-Fold Cross-Validation method for cross validation
num_folds = 3
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)#default>flase
#shuffle true;shuffle data before solitting


# Initialize GridSearchCV
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=kf,  # 5-fold cross-validation
    verbose=2,  # Print progress
    n_jobs=-1   # Use all available CPU cores
)
train = nifty50.iloc[:-100]
test = nifty50.iloc[-100:]
# Perform the grid search on the training data
grid_search.fit(train[new_predictors], train["Target"])
# Best parameters and model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print("Best Hyperparameters:", best_params)
print(best_model)
#evaluation of best model using cvs

cross_val_results = cross_val_score(best_model, train[new_predictors], train["Target"], cv=kf)
print("Cross-Validation Results:", cross_val_results)
print("Mean Accuracy:", np.mean(cross_val_results))
# Visualization: Confusion Matrix
plt.figure(figsize=(8, 6))

# Use the test set for confusion matrix
test_features = test[new_predictors]  # Features from the test set
test_target = test["Target"]          # Target labels from the test set

# Create the ConfusionMatrixDisplay object
cm = ConfusionMatrixDisplay.from_estimator(
    best_model,
    test_features,  # Features
    test_target,    # Target labels
    cmap='gray'     # Set colormap
)

# Add labels and title
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
rf.fit(train[new_predictors], train["Target"])
def predict_future(start_date, days=5):
    start_date = pd.to_datetime(start_date, utc=True)
    last_data = nifty50.loc[:start_date].iloc[-5:].copy()  # Get the last 5 rows of data

    future_dates = [] #date
    predictions = [] #outcome

    for i in range(days):
        next_day = last_data.index[-1] + timedelta(days=1) #get lat date and add 1 day  #start from next day
        while next_day.weekday() >= 5:  # Skip weekends
            next_day += timedelta(days=1) 

        # Use last row to create next prediction input
        new_row = last_data.tail(1).copy()
        new_row.index = pd.DatetimeIndex([next_day]) 

        # Add required predictor columns
        for horizon in [2,5,60,250,1000]:
            rolling_avg = last_data.rolling(horizon).mean()
            ratio_col = f"Close_Ratio_{horizon}"
            trend_col = f"Trend_{horizon}"

            new_row[ratio_col] = new_row["Close"] / rolling_avg["Close"].iloc[-1]  #uptil last 
            new_row[trend_col] = last_data["Target"].shift(1).rolling(horizon).sum().iloc[-1]

        new_row = new_row[new_predictors].ffill()  # Apply forward fill

        # Prediction using best_model
        pred = best_model.predict_proba(new_row)[:, 1]
        pred = 1 if pred[0] >= 0.5 else 0

        future_dates.append(next_day)
        predictions.append(pred)

        # Append new_row with prediction to last_data 
        new_row["Target"] = pred  #  real label
        last_data = pd.concat([last_data, new_row])

    future_dates = [d.date() for d in future_dates]
    return pd.DataFrame({"Date": future_dates, "Prediction": predictions})

# Run Model

def run_model():
    start_date = date_entry.get()
    if not start_date:
        text_box.delete("1.0", tk.END)
        text_box.insert(tk.END, "Please enter a start date.")
        return

    global predictions
    predictions = predict_future(start_date, days=5)

    # Add actual values to predictions   
    actuals = []
    for date in predictions["Date"]:
        d = pd.to_datetime(date)
        try:
            today_close = nifty50.loc[d.strftime('%Y-%m-%d')]["Close"] #date format
            prev_day = d - timedelta(days=1) #moves to previous day before current day
            while prev_day.strftime('%Y-%m-%d') not in nifty50.index or prev_day.weekday() >= 5:
                prev_day -= timedelta(days=1)
            yesterday_close = nifty50.loc[prev_day.strftime('%Y-%m-%d')]["Close"]
            actual = 1 if today_close > yesterday_close else 0
        except:
            actual = None
        actuals.append(actual)
    predictions["Actual"] = actuals

    update_display()


def update_display():
    text_box.delete("1.0", tk.END)
    text_box.insert(tk.END, predictions.to_string(index=False))
    plot_predictions()


def plot_predictions():
    if not graph_frame.winfo_exists():
        print("Graph frame no longer exists. GUI might have been closed.")
        return

    # Create bar chart comparing predicted vs actual
    fig, ax = plt.subplots(figsize=(6, 4))  # Adjust as needed

    x = range(len(predictions))
    bar_width = 0.4

    ax.bar( #move to left
        [i - bar_width/2 for i in x], 
        predictions["Prediction"], width=bar_width, 
        label="Predicted", 
        color="skyblue"
        )
    #move to right
    ax.bar([i + bar_width/2 for i in x], 
           predictions["Actual"], 
           width=bar_width, 
           label="Actual", 
           color="orange", 
           alpha=0.7
           )

    ax.set_title("Predicted vs Actual Movement")
    ax.set_ylabel("Market Direction (0 = Down, 1 = Up)")
    ax.set_xticks(x)
    ax.set_xticklabels(predictions["Date"].astype(str), rotation=45) # Rotate x-axis labels for better readability
    ax.set_ylim(-0.1, 1.1) #y walue limits
    ax.grid(axis='y', linestyle='--', alpha=0.6) 
    ax.legend() #color representation ( like add labels to the graph for differenciation)

    plt.tight_layout() #automatically adjust to give space between elements

    # Update canvas in GUI
    for widget in graph_frame.winfo_children():
        widget.destroy() # Clear previous graph

    canvas = FigureCanvasTkAgg(fig, master=graph_frame) #connect
    canvas.draw() # rander figure to canvas (make it visible)
    canvas.get_tk_widget().pack(pady=10, padx=10) # convert canvas (drawing spave) to tk widget (label, txt, button , ect)


# Tkinter GUI Setup
root = tk.Tk()
root.title("Nifty 50 Prediction")
root.geometry("900x700")

frame = ttk.Frame(root)
frame.pack(pady=20)

top_frame = ttk.Frame(frame)
top_frame.pack()

btn = ttk.Button(top_frame, text="Run Model", command=run_model)
btn.pack(side=tk.LEFT, padx=5)

date_label = ttk.Label(top_frame, text="Enter Start Date (YYYY-MM-DD):")
date_label.pack(side=tk.LEFT, padx=5)

date_entry = ttk.Entry(top_frame)
date_entry.pack(side=tk.LEFT, padx=5)

text_box = tk.Text(root, height=10, width=90)
text_box.pack()

graph_frame = ttk.Frame(root, height=350)  # Adjusted height to half window size
graph_frame.pack()

root.mainloop()


# Commit on 2025-04-04T12:00:00+05:30
# Commit on 2025-04-04T12:00:00+05:30
# Commit on 2025-04-04T12:00:00+05:30