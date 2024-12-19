# Load libraries
library(data.table)
library(Rtsne)
library(ggplot2)
library(ClusterR)

# Set seed
set.seed(3)

# Load data
data <- fread("./project/volume/data/raw/data.csv")

# Perform PCA
pca <- prcomp(data[, -1, with = FALSE])

# Scree plot
screeplot(pca)

# PCA summary
summary(pca)

# PCA data
pca_dt <- data.table(unclass(pca)$x)

# Perform t-SNE
tsne <- Rtsne(pca_dt, pca = FALSE, perplexity = 100, check_duplicates = FALSE)

# t-SNE data
tsne_dt <- data.table(tsne$Y)

# Determine optimal clusters using GMM and BIC
k_bic <- Optimal_Clusters_GMM(tsne_dt, max_clusters = 10, criterion = "BIC")

# Calculate delta BIC
delta_k <- c(NA, k_bic[-1] - k_bic[-length(k_bic)])

# Plot delta BIC values
del_k_tab <- data.table(delta_k = delta_k, k = 1:length(delta_k))
ggplot(del_k_tab, aes(x = k, y = -delta_k)) +
  geom_point() +
  geom_line() +
  scale_x_continuous(breaks = scales::pretty_breaks(n = 10)) +
  geom_text(aes(label = k), hjust = 0, vjust = -1)

# Choose optimal k value
opt_k <- 4

# Fit GMM with optimal k value
gmm_data <- GMM(tsne_dt, opt_k)

# Extract log-likelihoods and convert to probabilities
l_clust <- data.table(gmm_data$Log_likelihood^5)
net_lh <- apply(l_clust, 1, FUN = function(x) {sum(1 / x)})
cluster_prob <- 1 / l_clust / net_lh

# Prepare submission format
submission <- data.table(id = data$id)
for (i in 1:opt_k) {
  submission[[paste0("breed_", i)]] <-
    cluster_prob[[paste0("V", i)]]
}

# Swap values in breed_4 column with values in breed_3 column
submission[, `:=`(
  breed_4 = submission$breed_3,
  breed_3 = submission$breed_4
)]

# Export submission file
fwrite(submission, "./project/volume/data/processed/submission.csv")