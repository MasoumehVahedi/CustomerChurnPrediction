import pandas as pd
import numpy as np

# Visualization
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder





def plot_pie(df):
  labels = ["No", "Yes"]
  values = df["Churn"].value_counts().to_list()

  colors = ['gold', 'royalblue']

  # Pie plot
  fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4)])
  fig.update_traces(hoverinfo="label+value+text", textfont_size=20, textinfo="value",
                   marker=dict(colors=colors, line=dict(color="white", width=2)))
  fig.update_layout(dict(title="Customer Churn"))
  fig.show()


def distribution_pie_plot(df, column):
    churn = df[df["Churn"] == "Yes"]
    no_churn = df[df["Churn"] == "No"]

    # Create subplots: use 'domain' type for Pie subplot
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "domain"}, {"type": "domain"}]])
    # Churn
    fig.add_trace(go.Pie(values=churn[column].value_counts().values.tolist(),
                         labels=churn[column].value_counts().keys().tolist(),
                         name="Churn"),
                  1, 1)
    # No Churn
    fig.add_trace(go.Pie(values=no_churn[column].value_counts().values.tolist(),
                         labels=no_churn[column].value_counts().keys().tolist(),
                         name="No Churn"),
                  1, 2)

    fig.update_traces(hole=.4, hoverinfo="label+percent+name")

    fig.update_layout(title_text=column + "\n" + "Distribution Customer Analysis",
                      # Add annotations in the center of the donut pies
                      annotations=[
                          dict(text="Churn", x=0.18, y=0.5, font_size=20, showarrow=False),
                          dict(text="No Churn", x=0.82, y=0.5, font_size=20, showarrow=False)
                      ])
    fig.show()


# Histogram for distribution of numerical columns
def distribution_histogram(df, column):
    churn = df[df["Churn"] == "Yes"]
    no_churn = df[df["Churn"] == "No"]

    # Create subplots: use 'domain' type for Pie subplot
    fig = go.Figure()
    # Churn
    fig.add_trace(go.Histogram(x=churn[column],
                               histnorm="percent",
                               name="Churn",
                               marker=dict(line=dict(width=0.5, color="black")),
                               opacity=0.75)
                  )
    # No Churn
    fig.add_trace(go.Histogram(x=no_churn[column],
                               histnorm="percent",
                               name="No Churn",
                               marker=dict(line=dict(width=0.5, color="black")),
                               opacity=0.75)
                  )

    fig.update_layout(title_text=column + "\n" + "Histogram Customer Analysis",
                      bargap=0.2,
                      bargroupgap=0.1,
                      # xaxis label
                      xaxis=dict(gridcolor="white",
                                 title=column,
                                 zerolinewidth=1,
                                 ticklen=5,
                                 gridwidth=2),
                      # yaxis label
                      yaxis=dict(gridcolor="white",
                                 title="percent",
                                 zerolinewidth=1,
                                 ticklen=5,
                                 gridwidth=2)
                      )
    fig.show()


# Correlation Matrix
def get_correlation(df):
  correlation = df.corr()
  # Labels
  cols_matrix = correlation.columns.tolist()
  # Convert to numpy array
  correlation_arr = np.array(correlation)

  # Plot
  fig = go.Figure()
  fig.add_trace(go.Heatmap(x = cols_matrix,
                           y = cols_matrix,
                           z = correlation_arr,
                           colorscale = "Viridis",
                           colorbar = dict(title = "Pearson Correlation coefficient",
                                           titleside = "right"))
  )
  fig.update_layout(dict(title = "Correlation Matrix",
                      height = 770,
                      width = 900,
                      autosize = False,
                      yaxis = dict(tickfont = dict(size = 9)),
                      xaxis = dict(tickfont = dict(size = 9)),
                      )
  )
  fig.show()


if __name__ == "__main__":
    path = "/input/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    data = pd.read_csv(path)
    # print(data.head())
    print("info : {}".format(data.info()))
    print("Data Shape : {}".format(data.shape()))
    print(data.isnull().sum())

    # Pie plot
    plot_pie(data)

    # Distribution customers analysis
    category_columns = ["Contract", "gender", "Partner", "Dependents", "PhoneService",
                        "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                        "TechSupport", "StreamingTV", "StreamingMovies", "PaperlessBilling", "PaymentMethod"]
    # for all categorical columns plot pie and distribution
    for col in category_columns:
        distribution_pie_plot(data, col)


    # Histogram for distribution of numerical columns
    num_columns = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
    for col in num_columns:
        distribution_histogram(data, col)


    # Correlation
    # Encode Categorical Columns
    labelencoder = LabelEncoder()
    data[category_columns] = data[category_columns].apply(labelencoder.fit_transform)

    get_correlation(data)




