import streamlit as st
from sklearn.datasets import (
    load_digits,
    load_breast_cancer,
    make_circles,
    make_moons,
    make_swiss_roll,
)
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
from plotting_library import grouped_scatterplot, grouped_3dscatterplot
from sklearn.decomposition import KernelPCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
sns.set_style("darkgrid")

def compute_metrics(y_true, y_pred):
    result = dict(
        accuracy=accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred, average="weighted", zero_division=1),
        recall=recall_score(y_true, y_pred, average="weighted"),
        f1=f1_score(y_true, y_pred, average="weighted"),
    )

    for key, value in result.items():
        print("{} = {:.3f}%".format(key, value * 100))
    return result


def evaluate_classifier(cf, X, y, test_size=0.30):
    Z_train, Z_test, y_train, y_test = train_test_split(
        X, y, train_size=test_size, stratify=y, random_state=1111
    )
    cf.fit(X=Z_train, y=y_train)

    y_pred = cf.predict(Z_test)
    result = compute_metrics(y_test, y_pred)
    return (cf, result)


def plot_projection(df, k, grouping_name, axis_prefix, title, marker_size=5):

    xlabel = axis_prefix + "1"
    ylabel = axis_prefix + "2"
    zlabel = axis_prefix + "3"

    fig = None

    if k == 1:
        df["dummy"] = [1] * df.shape[0]
        fig = grouped_scatterplot(
            df=df,
            x_name=xlabel,
            y_name="dummy",
            grouping_name=grouping_name,
            marker_size=marker_size,
        )
        fig.update_layout(title=title, xaxis_title=xlabel)

    elif k == 2:
        fig = grouped_scatterplot(
            df=df,
            x_name=xlabel,
            y_name=ylabel,
            grouping_name=grouping_name,
            marker_size=marker_size,
        )
        fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title=ylabel)

    elif k == 3:
        fig = grouped_3dscatterplot(
            df=df,
            x_name=xlabel,
            y_name=ylabel,
            z_name=zlabel,
            grouping_name=grouping_name,
            marker_size=marker_size,
        )
        fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title=ylabel)
    return fig


def generate_dataset(name, num_samples=1000):
    if name == "Circles":
        X, y = make_circles(
            n_samples=num_samples, factor=0.3, noise=0.01, random_state=42
        )
    elif name == "Moons":
        X, y = make_moons(n_samples=num_samples, noise=0.01, random_state=42)

    elif name == "Digits":
        dataset = load_digits()
        X = dataset.data
        y = dataset.target
    return (X, y)


def main():
    # Set the title of the app
    st.set_page_config(page_title="Manifold Learning", layout="wide")
    st.title("Manifold Learning Demo")
    st.markdown(
        "##### By: Dharyll Prince M. Abellana | Assistant Professor of Computer Science | University of the Philippines Cebu"
    )

    with st.sidebar:
        dataset_name = st.selectbox(
            label="Select Dataset", options=["Moons", "Circles"]
        )
        X, y = generate_dataset(dataset_name, 500)
        st.header("Preprocessing")
        is_standardized = st.checkbox("Standardized", value=True)
        k = st.number_input(
            label="Number of Components", min_value=1, max_value=X.shape[1], value=2
        )
        st.header("Kernel PCA Parameters")

    # Preprocessing

    if is_standardized:
        sc = StandardScaler()
        X = sc.fit_transform(X)

    # Dimensionality Reduction
    ## PCA
    pca = PCA(n_components=k).fit(X)
    Z_pca = pca.transform(X)

    df_pca = pd.DataFrame(data=Z_pca, columns=[f"PC{i+1}" for i in range(k)])
    df_pca["target"] = y

    ## KPCA
    with st.sidebar:
        with st.container():
            col1, col2 = st.columns([3, 1])
            with col1:
                kernel = st.selectbox(
                    label="Kernel",
                    options=["Radial Basis Function", "Linear", "Polynomial"],
                )
            with col2:
                step_size = st.number_input("Step Size", min_value=0.0001, step=0.01, value=0.1)
    kpca = None
    if kernel == "Radial Basis Function":
        with st.sidebar:
            with st.container():
                col1, col2 = st.columns(2)
                with col1:
                    gamma = st.number_input("Gamma", min_value=0.0001, step=step_size, value=0.3)
                with col2:
                    alpha = st.number_input("Alpha", min_value=0.0001, step=step_size, value=0.4)
        kpca = KernelPCA(
            n_components=k,
            kernel="rbf",
            gamma=gamma,
            alpha=alpha,
            fit_inverse_transform=True,
        )  ## moons
    elif kernel == "Linear":
        kpca = KernelPCA(
            n_components=k,
            kernel="linear",
            fit_inverse_transform=True,
        )
    elif kernel == "Polynomial":
        with st.sidebar:
            deg = st.number_input("Degree", min_value=1, step=1)
            with st.container():
                col1, col2 = st.columns(2)
                with col1:
                    alpha = st.number_input("Alpha", min_value=0.0, value=0.1)
                with col2:
                    coef = st.number_input("Coefficient", min_value=0.0, value=0.1)

        kpca = KernelPCA(
            n_components=k,
            kernel="poly",
            degree=deg,
            coef0=coef,
            alpha=alpha,
            fit_inverse_transform=True,
        )
    Z_kpca = kpca.fit_transform(X)

    df_kpca = pd.DataFrame(data=Z_kpca, columns=[f"KPC{i+1}" for i in range(k)])
    df_kpca["target"] = y

    ## TSNE

    with st.sidebar:
        st.header("t-SNE Parameters")
        perplexity = st.number_input(label="Perplexity", min_value=1, value=4)

    tsne = TSNE(
        n_components=k,
        perplexity=perplexity,
    )

    Z_tsne = tsne.fit_transform(X)

    df_tsne = pd.DataFrame(data=Z_tsne, columns=[f"TSNE{i+1}" for i in range(k)])
    df_tsne["target"] = y

    # Plotting

    col1, col2 = st.columns(2)
    with st.container():
        with col1:
            ## original data
            col_names = [f"X{i+1}" for i in range(X.shape[1])]
            df_orig = pd.DataFrame(data=X, columns=col_names)
            df_orig["target"] = y
            nvars = X.shape[1]
            if X.shape[1] < 4:
                fig1 = plot_projection(
                    df_orig, X.shape[1], "target", "X", "Original"
                )
                st.plotly_chart(fig1)
        with col2:
            ## PCA
            fig2 = plot_projection(
                df_pca, k, "target", "PC", "Principal Component Analysis"
            )
            st.plotly_chart(fig2)
    with st.container():
        with col1:
            ## KPCA
            fig3 = plot_projection(
                df_kpca, k, "target", "KPC", "Kernel Principal Component Analysis"
            )
            st.plotly_chart(fig3)
        with col2:
            ## TSNE
            fig4 = plot_projection(
                df_tsne,
                k,
                "target",
                "TSNE",
                "t-distributed Stochastic Neighbor Embedding",
            )
            st.plotly_chart(fig4)
    

if __name__ == "__main__":
    main()
