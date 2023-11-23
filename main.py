import tkinter as tk
from tkinter import filedialog, messagebox
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from collections import defaultdict
import spacy
from scipy.cluster.hierarchy import dendrogram, linkage

class DataMiningApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Data Mining Application")

        self.data = None
        self.wordvec_model = None

        self.load_data_button = tk.Button(root, text="Load Data", command=self.load_data)
        self.load_data_button.pack(pady=10)

        self.perform_regression_button = tk.Button(root, text="Perform Multiple Regression", command=self.perform_regression)
        self.perform_regression_button.pack(pady=10)

        self.perform_clustering_button = tk.Button(root, text="Perform KMeans Clustering", command=self.perform_clustering)
        self.perform_clustering_button.pack(pady=10)

        self.load_more_vectors_button = tk.Button(root, text="Load More Vectors", command=self.load_more_vectors)
        self.load_more_vectors_button.pack(pady=10)

    def load_data(self):
        file_path = filedialog.askopenfilename(title="Select CSV File", filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.data = pd.read_csv(file_path)
            messagebox.showinfo("Success", "Data loaded successfully!")

    def preprocess_data(self):
        # Your preprocessing logic here
        pass

    def perform_regression(self):
        if self.data is None:
            messagebox.showerror("Error", "Please load the data first.")
            return

        # Your regression logic here
        X = self.data.drop(['Diagnosis'], axis=1)
        Y = self.data['Diagnosis']
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)
        model = LinearRegression()
        model.fit(X_train, Y_train)
        predictions = model.predict(X_test)
        mse = mean_squared_error(Y_test, predictions)
        messagebox.showinfo("Regression Results", f"Mean Squared Error: {mse:.4f}")

    def perform_clustering(self):
        if self.data is None:
            messagebox.showerror("Error", "Please load the data first.")
            return

        if self.wordvec_model is None:
            messagebox.showerror("Error", "Please load word vectors first.")
            return

        vocab = ["斉藤", "リンゴ", "シマウマ", "東京", "ライオン", "名古屋", "ミカン", "ウシ", "メロン", "田中", "横浜", "鈴木"]
        wordvec, vocab_new = self.load_word_vectors(self.wordvec_model, vocab)
        kmeans_model = KMeans(n_clusters=4, verbose=1, max_iter=30)
        kmeans_model.fit(wordvec)

        cluster_labels = kmeans_model.labels_
        cluster_to_words = defaultdict(list)
        for cluster_id, word in zip(cluster_labels, vocab_new):
            cluster_to_words[cluster_id].append(word)

        for words in cluster_to_words.values():
            print(words)

        self.visualize_clusters(wordvec, vocab_new, kmeans_model)

    def load_word_vectors(self, model_path, vocab):
        # Your word vector loading logic here
        # Example: Load more vectors using the provided model
        wordvec = []
        vocab_new = []

        for x in vocab:
            try:
                wordvec.append(model_path.get_vector(x))
                vocab_new.append(x)
            except KeyError:
                print(f"Ignoring {x}")
                continue

        return wordvec, vocab_new

    def visualize_clusters(self, wordvec, vocab_new, kmeans_model):
        # Your visualization logic here
        # Example: PCA for dimensionality reduction and scatter plot
        pca = PCA(n_components=2)
        wordvec_r = pca.fit_transform(wordvec)

        plt.figure(figsize=(10, 8))
        for (i, label) in enumerate(kmeans_model.labels_):
            plt.scatter(wordvec_r[i, 0], wordvec_r[i, 1], label=label)
            plt.annotate(vocab_new[i], xy=(wordvec_r[i, 0], wordvec_r[i, 1]), size=8)

        plt.title("KMeans Clustering Visualization")
        plt.legend()
        plt.show()

    def load_more_vectors(self):
        model_path = filedialog.askopenfilename(title="Select Word Vector Model", filetypes=[("Model files", "*.bin")])
        if model_path:
            self.wordvec_model = torch.load(model_path)
            messagebox.showinfo("Success", "Word Vectors loaded successfully!")

def main():
    root = tk.Tk()
    app = DataMiningApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
