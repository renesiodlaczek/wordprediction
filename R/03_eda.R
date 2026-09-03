# Step 3: Exploratory analysis on the cleaned, profanity-filtered training
# sample. Computes and caches the tables the milestone report plots/tables are
# built from (data/processed/eda_*.rds), so the .qmd just loads and plots.

library(stringr)
library(data.table)

source("R/02_tokenize_clean.R")

out_dir <- "data/processed"
train_df <- as.data.table(readRDS(file.path(out_dir, "train_lines.rds")))
train_df[, uid := seq_len(.N)]

# --- Tokenize + profanity-filter the training sample ------------------------
tokens <- tokenize_text(train_df$text)
tokens[, source := train_df$source[line_id]]
n_profane_removed <- tokens[word %in% profanity_words, .N]
tokens <- filter_profanity(tokens)

cat("Total word tokens (post-cleaning, pre-profanity-filter):",
    n_profane_removed + nrow(tokens), "\n")
cat("Profane tokens removed:", n_profane_removed,
    sprintf("(%.3f%%)\n", 100 * n_profane_removed / (n_profane_removed + nrow(tokens))))

saveRDS(tokens, file.path(out_dir, "tokens_train.rds"))

# --- Words per line, by source ----------------------------------------------
words_per_line <- tokens[, .N, by = .(line_id, source)]
saveRDS(words_per_line, file.path(out_dir, "eda_words_per_line.rds"))

# --- Unigram frequencies -----------------------------------------------------
unigram_freq <- tokens[, .N, by = word][order(-N)]
saveRDS(unigram_freq, file.path(out_dir, "eda_unigram_freq.rds"))

# --- Coverage curve: unique words needed to cover X% of all word instances --
unigram_freq[, cum_share := cumsum(N) / sum(N)]
coverage_pts <- c(0.5, 0.75, 0.9, 0.95)
coverage_summary <- data.table(
  coverage = coverage_pts,
  n_words_needed = vapply(coverage_pts, function(p) which(unigram_freq$cum_share >= p)[1], integer(1))
)
saveRDS(coverage_summary, file.path(out_dir, "eda_coverage_summary.rds"))
# Thin the curve for plotting (every 50th rank) to keep the report light.
coverage_curve <- unigram_freq[seq(1, .N, by = 50), .(rank = .I * 50, cum_share)]
saveRDS(coverage_curve, file.path(out_dir, "eda_coverage_curve.rds"))

# --- Bigrams & trigrams (within-line only) -----------------------------------
# Build n-grams via in-place, grouped shift() rather than merge()-ing n
# full-size copies of the token table together -- much lighter on memory for
# a ~21M-row token table. Temporary w1..wn columns are added to `tokens` and
# removed again on exit, so no extra copy of `tokens` itself is needed.
setorder(tokens, line_id)

make_ngrams <- function(tokens, n) {
  cols <- paste0("w", seq_len(n))
  on.exit(tokens[, (cols) := NULL], add = TRUE)
  for (i in seq_len(n)) {
    tokens[, (cols[i]) := shift(word, n = i - 1, type = "lead"), by = line_id]
  }
  na.omit(tokens[, ..cols])
}

bigrams <- make_ngrams(tokens, 2)
bigrams[, ngram := paste(w1, w2)]
bigram_freq <- bigrams[, .N, by = ngram][order(-N)]
saveRDS(bigram_freq, file.path(out_dir, "eda_bigram_freq.rds"))
rm(bigrams); gc()

trigrams <- make_ngrams(tokens, 3)
trigrams[, ngram := paste(w1, w2, w3)]
trigram_freq <- trigrams[, .N, by = ngram][order(-N)]
saveRDS(trigram_freq, file.path(out_dir, "eda_trigram_freq.rds"))
rm(trigrams); gc()

# --- Per-source summary table -------------------------------------------------
source_summary <- tokens[, .(n_tokens = .N, n_unique_words = uniqueN(word)), by = source]
n_lines_by_source <- train_df[, .N, by = source]
setnames(n_lines_by_source, "N", "n_lines")
source_summary <- merge(source_summary, n_lines_by_source, by = "source")
saveRDS(source_summary, file.path(out_dir, "eda_source_summary.rds"))

cat("\nTop 10 words:\n"); print(head(unigram_freq, 10))
cat("\nTop 10 bigrams:\n"); print(head(bigram_freq, 10))
cat("\nTop 10 trigrams:\n"); print(head(trigram_freq, 10))
cat("\nCoverage summary:\n"); print(coverage_summary)
cat("\nSource summary:\n"); print(source_summary)
