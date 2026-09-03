# Step 1: Read raw corpora, report their basic size, and cache a proportional
# random sample for downstream cleaning/EDA/modeling.
#
# Raw files live in data/ (gitignored). This script writes cached artifacts to
# data/processed/ (gitignored) so later scripts don't need to re-read ~570MB of
# raw text every time.

library(stringr)

set.seed(8137)

data_dir <- "data"
out_dir <- file.path(data_dir, "processed")
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

raw_files <- c(
  twitter = file.path(data_dir, "en_US.twitter.txt"),
  blogs   = file.path(data_dir, "en_US.blogs.txt"),
  news    = file.path(data_dir, "en_US.news.txt")
)

read_lines_safe <- function(path) {
  con <- file(path, "r", encoding = "UTF-8")
  lines <- readLines(con, skipNul = TRUE, warn = FALSE)
  close(con)
  lines
}

raw_lines <- lapply(raw_files, read_lines_safe)

# --- Raw corpus summary stats (based on full files, not the sample) ---------
raw_stats <- data.frame(
  source = names(raw_lines),
  file_size_mb = round(file.size(raw_files) / 1024^2, 1),
  n_lines = vapply(raw_lines, length, integer(1)),
  n_words = vapply(raw_lines, function(x) sum(str_count(x, "\\S+")), numeric(1))
)
raw_stats$avg_words_per_line <- round(raw_stats$n_words / raw_stats$n_lines, 2)

saveRDS(raw_stats, file.path(out_dir, "raw_stats.rds"))
print(raw_stats)

# --- Proportional 1,000,000-line sample across the three sources ------------
target_total <- 1e6
total_lines <- sum(raw_stats$n_lines)
sample_p <- target_total / total_lines

sample_lines_from <- function(lines) {
  keep <- rbinom(length(lines), 1, sample_p) == 1
  lines[keep]
}

sampled <- lapply(raw_lines, sample_lines_from)
sample_counts <- vapply(sampled, length, integer(1))
cat("Sampled line counts per source:\n")
print(sample_counts)
cat("Total sampled lines:", sum(sample_counts), "(target:", target_total, ")\n")

rm(raw_lines)
gc()

# Tag each line with its source before combining, and shuffle.
sample_df <- data.frame(
  source = rep(names(sampled), sample_counts),
  text = unlist(sampled, use.names = FALSE),
  stringsAsFactors = FALSE
)
sample_df <- sample_df[sample(nrow(sample_df)), ]
rownames(sample_df) <- NULL

# Three-way split: 80% train (n-gram counts), 10% dev (hyperparameter tuning /
# model selection), 10% test (final, untouched evaluation). Keeping dev and
# test separate avoids tuning MIN_COUNT/TOP_K_PER_PREFIX/ALPHA and then
# re-using the *same* held-out lines to report "final" accuracy.
n <- nrow(sample_df)
shuffled_idx <- sample(n)
dev_idx  <- shuffled_idx[1:floor(0.1 * n)]
test_idx <- shuffled_idx[(floor(0.1 * n) + 1):floor(0.2 * n)]
train_idx <- shuffled_idx[(floor(0.2 * n) + 1):n]

train_df <- sample_df[train_idx, ]
dev_df   <- sample_df[dev_idx, ]
test_df  <- sample_df[test_idx, ]

saveRDS(train_df, file.path(out_dir, "train_lines.rds"))
saveRDS(dev_df, file.path(out_dir, "dev_lines.rds"))
saveRDS(test_df, file.path(out_dir, "test_lines.rds"))

cat("Train lines:", nrow(train_df), " Dev lines:", nrow(dev_df),
    " Test lines:", nrow(test_df), "\n")
