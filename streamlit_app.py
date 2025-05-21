import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import umap
import os
import json

tab_train, tab_eval = st.tabs(["Train Model", "Evaluate New Data"])

def preprocess_data(df, numerical_cols=None, categorical_cols=None):
    """Preprocess data with column validation and missing column handling"""
    # Add missing columns with appropriate defaults
    if numerical_cols:
        for col in numerical_cols:
            if col not in df.columns:
                df[col] = 0  # Default for numerical columns
                
    if categorical_cols:
        for col in categorical_cols:
            if col not in df.columns:
                df[col] = ''  # Default for categorical columns

    # Existing preprocessing logic
    for column in df.columns:
        if df[column].dtype == 'object' or pd.api.types.is_string_dtype(df[column]):
            df[column] = df[column].fillna('')
        elif pd.api.types.is_numeric_dtype(df[column]):
            df[column] = df[column].fillna(0)
            try:
                if (df[column] % 1 == 0).all():
                    df[column] = df[column].astype(int)
            except TypeError:
                pass
                
    # Ensure column order matches training structure
    if numerical_cols and categorical_cols:
        return df[numerical_cols + categorical_cols]
    return df

def create_preprocessor(numerical_cols, categorical_cols):
    """Create column transformer with dynamic column validation"""
    return ColumnTransformer([
        ('num', Pipeline([
            ('scaler', StandardScaler())
        ]), numerical_cols),
        ('cat', Pipeline([
            ('onehot', OneHotEncoder(
                handle_unknown='ignore', 
                sparse_output=False))
        ]), categorical_cols)
    ], verbose_feature_names_out=False)

def plot_visualizations(data, labels):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    data_tsne = tsne.fit_transform(data)
    ax1.scatter(data_tsne[:, 0], data_tsne[:, 1], c=labels, cmap='tab10', s=10)
    ax1.set_title('t-SNE Visualization')
    
    # UMAP
    reducer = umap.UMAP(random_state=42, metric='correlation')
    data_umap = reducer.fit_transform(data)
    ax2.scatter(data_umap[:, 0], data_umap[:, 1], c=labels, cmap='tab10', s=10)
    ax2.set_title('UMAP Visualization')
    
    return fig

def list_pkl_files(bucket_dir='.'):
    """List all .pkl files in the specified directory."""
    return [f for f in os.listdir(bucket_dir) if f.endswith('.pkl')]

def main():
    with tab_train:
        st.header("Train Model")
        st.title("Advanced Clustering Analysis")

        # Sidebar controls
        st.sidebar.header("Actions")

        # Bucket for saved .pkl files
        st.sidebar.subheader("Artifact Bucket")
        pkl_files = list_pkl_files()
        selected_preprocessor = st.sidebar.selectbox("Preprocessor", [f for f in pkl_files if 'preprocessor' in f])
        selected_pca = st.sidebar.selectbox("PCA", [f for f in pkl_files if 'pca' in f])
        selected_kmeans = st.sidebar.selectbox("KMeans Model", [f for f in pkl_files if 'kmeans' in f])

        st.sidebar.subheader("Train Model")
        train_file = st.sidebar.file_uploader("Upload Training Data (JSON)", type=['json'])
        train_button = st.sidebar.button("Run Clustering Analysis")

    with tab_eval:
        st.header("Evaluate New Data")
        st.sidebar.subheader("Evaluate New Data")
        val_file = st.sidebar.file_uploader("Upload Validation Data (JSON)", type=['json'])
        eval_button = st.sidebar.button("Evaluate")
        
        # Display model selection in the eval tab
        if list_pkl_files():
            selected_model = st.sidebar.selectbox(
                "Select Model for Evaluation", 
                [f for f in list_pkl_files() if f.startswith('cluster_model')]
            )

    # Main area
    if train_file and train_button:
        df_train = pd.read_json(train_file)
        df_train = preprocess_data(df_train)
        
        # Get and store column metadata
        numerical_cols = df_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = df_train.select_dtypes(include=['object']).columns.tolist()
        
        # Create versioned artifact name
        model_version = f"cluster_model_v{len([f for f in list_pkl_files() if f.startswith('cluster_model')]) + 1}.pkl"
        
        # Save column structure
        column_metadata = {
            'numerical': numerical_cols,
            'categorical': categorical_cols,
            'all_columns': numerical_cols + categorical_cols
        }
        
        with open('column_metadata.json', 'w') as f:
            json.dump(column_metadata, f)
        
        preprocessor = create_preprocessor(numerical_cols, categorical_cols)
        X_processed = preprocessor.fit_transform(df_train)
        pca = PCA(n_components=10, random_state=42)
        data_reduced = pca.fit_transform(X_processed)
        
        range_n_clusters = range(7, 30)
        silhouette_scores = []
        for n_clusters in range_n_clusters:
            kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=1000, random_state=42)
            labels = kmeans.fit_predict(data_reduced)
            score = silhouette_score(data_reduced, labels, sample_size=10000, random_state=42)
            silhouette_scores.append(score)
        
        fig1, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range_n_clusters, silhouette_scores, 'o-', color='blue')
        ax.set_title('Silhouette Score Analysis')
        ax.set_xlabel('Number of Clusters')
        ax.set_ylabel('Silhouette Score')
        st.pyplot(fig1)
        
        optimal_n_clusters = range_n_clusters[np.argmax(silhouette_scores)]
        final_kmeans = MiniBatchKMeans(n_clusters=optimal_n_clusters, batch_size=1000, random_state=42)
        final_labels = final_kmeans.fit_predict(data_reduced)
        
        # Save complete artifact
        joblib.dump({
            'preprocessor': preprocessor,
            'pca': pca,
            'kmeans': final_kmeans,
            'metadata': {
                'numerical': numerical_cols,
                'categorical': categorical_cols,
                'all_columns': numerical_cols + categorical_cols,
                'version': model_version
            }
        }, model_version)
        
        st.success(f"Model saved as {model_version}")
        
        # Visualizations
        st.header("Training Data Visualizations")
        fig2 = plot_visualizations(data_reduced, final_labels)
        st.pyplot(fig2)

    if val_file and eval_button:
        try:
            # Load selected model
            selected_model = st.sidebar.selectbox(
                "Select Model", 
                [f for f in list_pkl_files() if f.startswith('cluster_model')]
            )
            artifact = joblib.load(selected_model)
            
            # Process validation data
            df_val = pd.read_json(val_file)
            # Add missing columns with appropriate defaults
            numerical_cols = artifact['metadata']['numerical']
            categorical_cols = artifact['metadata']['categorical']
            all_columns = artifact['metadata']['all_columns']
            for col in all_columns:
                if col not in df_val.columns:
                    if col in numerical_cols:
                        df_val[col] = 0
                    else:
                        df_val[col] = ''
            for col in categorical_cols:
                df_val[col] = df_val[col].astype(str).fillna('')
            for col in numerical_cols:
                df_val[col] = pd.to_numeric(df_val[col], errors='coerce').fillna(0)
            df_val = df_val[all_columns]
            
            # Transform and predict
            X_val = artifact['preprocessor'].transform(df_val)
            val_reduced = artifact['pca'].transform(X_val)
            val_labels = artifact['kmeans'].predict(val_reduced)
            
            st.header("Validation Results")
            
            # fig3: t-SNE visualization of validation data
            tsne = TSNE(n_components=2, random_state=42)
            val_tsne = tsne.fit_transform(val_reduced)
            fig3 = plt.figure(figsize=(10, 6))
            plt.scatter(val_tsne[:, 0], val_tsne[:, 1], c=val_labels, cmap='tab10', marker='x', alpha=0.7)
            plt.title('t-SNE Visualization of Validation Data Clusters')
            plt.colorbar()
            st.pyplot(fig3)
            
            # fig4: UMAP visualization of validation data
            reducer = umap.UMAP(random_state=42, metric='correlation')
            val_umap = reducer.fit_transform(val_reduced)
            fig4 = plt.figure(figsize=(10, 6))
            plt.scatter(val_umap[:, 0], val_umap[:, 1], c=val_labels, cmap='tab10', marker='x', alpha=0.7)
            plt.title('UMAP Visualization of Validation Data Clusters')
            plt.colorbar()
            st.pyplot(fig4)
            # Add cluster assignments to the validation DataFrame
            df_val_with_clusters = df_val.copy()
            df_val_with_clusters['cluster_id'] = val_labels

            # Convert to CSV (as bytes)
            csv = df_val_with_clusters.to_csv(index=False).encode('utf-8')

            # Add download button
            st.download_button(
                label="Download validation data with cluster_id as CSV",
                data=csv,
                file_name="validation_with_clusters.csv",
                mime="text/csv",
                key="download-csv"
            )
        except Exception as e:
            st.error(f"Evaluation failed: {str(e)}")



if __name__ == "__main__":
    main()
