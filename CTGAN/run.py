from ctgan_me_numeric_regression import CTGAN
from data_sampler import DataSampler
from data_transformer import DataTransformer
import pandas as pd
import numpy as np
import plotly
import plotly.express as px

data = pd.read_csv("data_mis.csv")
discrete_columns = ["Z"]
ctgan = CTGAN(epochs = 5000)
ctgan.fit(data, discrete_columns)
imp = ctgan.impute()
pd.DataFrame(imp).to_csv("imputation.csv", sep = "\t")


fig = px.scatter(pd.DataFrame(imp), x="X", y="Y", title='Epoch vs. Loss')
fig.write_html("distru_fig.html")


loss_values = ctgan.loss_values
loss_values_reformatted = pd.melt(
    loss_values,
    id_vars=['Epoch'],
    var_name='Loss Type'
)

fig = px.line(loss_values_reformatted, x="Epoch", y="value", color="Loss Type", title='Epoch vs. Loss')
fig.write_html("loss_fig.html")