# Step 4: N-gram model, hyperparameter tuning, and two scoring strategies for
# unseen n-grams: Stupid Backoff and linear interpolation.
#
# Data splits (from 01_sample_data.R): train (80%) builds the n-gram counts;
# dev (10%) is used below to grid-search MIN_COUNT / TOP_K_PER_PREFIX and the
# interpolation weights, and to pick backoff vs. interpolation; test (10%) is
# touched exactly once, at the end, for the numbers that go in the report.
#
# Two kinds of n-gram table are kept per order:
#  - "full" tables (count >= min_count only): used to score the *actual* next
#    word for perplexity, since restricting to a few candidates per prefix
#    would make many true words score 0.
#  - "pruned" tables (also capped at the top TOP_K_PER_PREFIX candidates per
#    prefix): what the deployed predictor actually uses, since that's all a
#    next-word suggestion UI needs, and it's dramatically smaller.
#
# Stupid Backoff (Brants et al. 2007) and linear interpolation are both
# implemented; see the comparison at the bottom for which one this project
# keeps.

library(data.table)
library(stringr)

source("R/02_tokenize_clean.R")

`%||%` <- function(a, b) if (length(a) == 0 || is.na(a)) b else a

out_dir <- "data/processed"
tokens <- readRDS(file.path(out_dir, "tokens_train.rds"))
setDT(tokens)
setorder(tokens, line_id)

unigram_freq <- readRDS(file.path(out_dir, "eda_unigram_freq.rds"))
total_unigrams <- sum(unigram_freq$N)
unigram_top <- copy(unigram_freq)[order(-N)]
setkey(unigram_freq, word)

# --- Raw (n-1)-word-prefix -> word -> count, via in-place grouped shift() ---
# (see R/03_eda.R for why shift() rather than merge()-ing shifted copies)
build_ngram_counts <- function(tokens, n) {
  cols <- paste0("w", seq_len(n))
  on.exit(tokens[, (cols) := NULL], add = TRUE)
  for (i in seq_len(n)) {
    tokens[, (cols[i]) := shift(word, n = i - 1, type = "lead"), by = line_id]
  }
  complete <- na.omit(tokens[, ..cols])
  prefix_cols <- cols[-n]
  complete[, prefix := do.call(paste, c(.SD, sep = " ")), .SDcols = prefix_cols]
  setnames(complete, cols[n], "word")
  counts <- complete[, .N, by = .(prefix, word)]
  rm(complete); gc()
  setnames(counts, "N", "count")
  counts[]
}

cat("Counting bigrams...\n");  bigram_counts  <- build_ngram_counts(tokens, 2); gc()
cat("Counting trigrams...\n"); trigram_counts <- build_ngram_counts(tokens, 3); gc()
cat("Counting fourgrams...\n"); fourgram_counts <- build_ngram_counts(tokens, 4); gc()
rm(tokens); gc()

# --- Turn raw counts into a lookup table for a given (min_count, top_k) ----
finalize_table <- function(counts, min_count, top_k = Inf) {
  tab <- counts[count >= min_count]
  setorder(tab, prefix, -count)
  if (is.finite(top_k)) {
    tab[, rank := seq_len(.N), by = prefix]
    tab <- tab[rank <= top_k]
    tab[, rank := NULL]
  }
  tab[, prefix_total := sum(count), by = prefix]
  setkey(tab, prefix)
  tab[]
}

# --- Stupid Backoff prediction (top-N candidate words) ----------------------
make_predictor <- function(tabs) {  # tabs = list(bigram, trigram, fourgram)
  function(phrase, top_n = 3) {
    words <- str_split(clean_lines(phrase), " ")[[1]]
    words <- words[words != ""]
    if (length(words) == 0) return(head(unigram_top, top_n)$word)
    max_context <- min(3, length(words))
    for (k in max_context:1) {
      context <- paste(tail(words, k), collapse = " ")
      matches <- tabs[[k]][.(context)]
      if (!is.na(matches$word[1])) return(matches$word[seq_len(min(top_n, nrow(matches)))])
    }
    head(unigram_top, top_n)$word
  }
}

# --- Load dev/test sets and tokenize (same pipeline as training) ------------
prep_eval_tokens <- function(rds_name) {
  df <- readRDS(file.path(out_dir, rds_name))
  tk <- filter_profanity(tokenize_text(df$text))
  setDT(tk)
  tk
}
dev_tokens  <- prep_eval_tokens("dev_lines.rds")
test_tokens <- prep_eval_tokens("test_lines.rds")

# Build a fixed set of (context, truth) next-word-prediction cases from a
# token set: first 3 words as context, 4th word as the target.
make_cases <- function(tk, n_cases = 5000, seed = 4926) {
  set.seed(seed)
  lines <- tk[, .N, by = line_id][N >= 4, line_id]
  ids <- lines[sample(length(lines), min(n_cases, length(lines)))]
  rbindlist(lapply(ids, function(lid) {
    w <- tk[line_id == lid, word]
    list(context = paste(w[1:3], collapse = " "), truth = w[4])
  }))
}
dev_cases  <- make_cases(dev_tokens)
test_cases <- make_cases(test_tokens)

top3_accuracy <- function(predictor, cases) {
  hits <- vapply(seq_len(nrow(cases)), function(i) {
    cases$truth[i] %in% predictor(cases$context[i], top_n = 3)
  }, logical(1))
  mean(hits)
}

# --- Grid search MIN_COUNT x TOP_K_PER_PREFIX on the dev set -----------------
# (ALPHA, the backoff discount, does *not* affect which words rank in the
# top-3 for a fixed context under strict backoff -- it only rescales scores
# uniformly within whichever single order is used -- so it isn't part of this
# grid; it matters once scores from multiple orders are blended, in the
# interpolation model below.)
grid <- CJ(min_count = c(1, 2), top_k = c(5, 10, 20))
grid_results <- rbindlist(lapply(seq_len(nrow(grid)), function(i) {
  mc <- grid$min_count[i]; tk <- grid$top_k[i]
  tabs <- list(
    finalize_table(bigram_counts, mc, tk),
    finalize_table(trigram_counts, mc, tk),
    finalize_table(fourgram_counts, mc, tk)
  )
  predictor <- make_predictor(tabs)
  total_rows <- sum(vapply(tabs, nrow, integer(1)))
  data.table(min_count = mc, top_k = tk,
             dev_accuracy = top3_accuracy(predictor, dev_cases),
             n_rows = total_rows)
}))
print(grid_results[order(-dev_accuracy)])

best <- grid_results[order(-dev_accuracy)][1]
cat(sprintf("\nBest config: min_count=%d, top_k=%d (dev accuracy %.1f%%, %s rows)\n",
            best$min_count, best$top_k, 100 * best$dev_accuracy, format(best$n_rows, big.mark = ",")))

MIN_COUNT <- best$min_count
TOP_K_PER_PREFIX <- best$top_k
ALPHA <- 0.4  # Stupid Backoff discount factor (see note above)

bigram_tab   <- finalize_table(bigram_counts, MIN_COUNT, TOP_K_PER_PREFIX)
trigram_tab  <- finalize_table(trigram_counts, MIN_COUNT, TOP_K_PER_PREFIX)
fourgram_tab <- finalize_table(fourgram_counts, MIN_COUNT, TOP_K_PER_PREFIX)
tables_by_order <- list(bigram_tab, trigram_tab, fourgram_tab)
predict_backoff <- make_predictor(tables_by_order)

# --- Full (min_count-only) tables, for scoring the *actual* next word -------
bigram_full   <- finalize_table(bigram_counts, MIN_COUNT)
trigram_full  <- finalize_table(trigram_counts, MIN_COUNT)
fourgram_full <- finalize_table(fourgram_counts, MIN_COUNT)
full_tables_by_order <- list(bigram_full, trigram_full, fourgram_full)
rm(bigram_counts, trigram_counts, fourgram_counts); gc()

# Stupid Backoff score for one specific candidate word (used for perplexity;
# uses the *full* tables so a word being outside the top-K doesn't force a
# zero score for words that were genuinely observed).
#
# NB: the candidate word argument is named `target_word`, not `word` --
# data.table's `[` looks up bare symbols against the table's own columns
# before the enclosing environment, and these tables have a `word` column, so
# naming the argument `word` would silently shadow it and break filtering.
backoff_score <- function(words, target_word) {
  max_context <- min(3, length(words))
  for (k in max_context:1) {
    context <- paste(tail(words, k), collapse = " ")
    tab <- full_tables_by_order[[k]]
    # `tab[.(context)]` binary-joins on prefix only; filter to the candidate word.
    row <- tab[.(context)]
    row <- row[word == target_word]
    if (nrow(row) == 1) return(ALPHA^(max_context - k) * row$count / row$prefix_total)
  }
  ALPHA^max_context * (unigram_freq[.(target_word)]$N %||% 1) / total_unigrams
}

# --- Linear interpolation: blend scores across all matching orders ---------
# score(w | context) = sum_k lambda_k * P_k(w | last k words), summed over
# every order k in {1 (unigram), 2 (bigram context), 3 (trigram), 4 (4-gram)}
# for which the context/word pair is observed; lambdas sum to 1.
interpolated_score <- function(words, target_word, lambdas) {
  max_context <- min(3, length(words))
  score <- lambdas[1] * (unigram_freq[.(target_word)]$N %||% 0) / total_unigrams
  for (k in seq_len(max_context)) {
    context <- paste(tail(words, k), collapse = " ")
    tab <- full_tables_by_order[[k]]
    row <- tab[.(context)]
    row <- row[word == target_word]
    if (nrow(row) == 1) score <- score + lambdas[k + 1] * row$count / row$prefix_total
  }
  score
}

predict_interpolated <- function(phrase, lambdas, top_n = 3) {
  words <- str_split(clean_lines(phrase), " ")[[1]]
  words <- words[words != ""]
  if (length(words) == 0) return(head(unigram_top, top_n)$word)
  max_context <- min(3, length(words))
  # Candidate pool: union of each applicable order's top candidates for this
  # context (keeps lookups cheap -- we don't score the whole vocabulary).
  candidates <- character(0)
  for (k in seq_len(max_context)) {
    context <- paste(tail(words, k), collapse = " ")
    matches <- tables_by_order[[k]][.(context)]
    if (!is.na(matches$word[1])) candidates <- c(candidates, matches$word)
  }
  candidates <- unique(candidates)
  if (length(candidates) == 0) return(head(unigram_top, top_n)$word)
  scores <- vapply(candidates, function(w) interpolated_score(words, w, lambdas), numeric(1))
  candidates[order(-scores)][seq_len(min(top_n, length(candidates)))]
}

# Small grid over interpolation weights (lambda_unigram, lambda_bigram,
# lambda_trigram, lambda_4gram), each summing to 1, favoring progressively
# more weight on longer contexts -- tuned on the dev set.
lambda_grid <- list(
  c(0.10, 0.20, 0.30, 0.40),
  c(0.05, 0.15, 0.30, 0.50),
  c(0.15, 0.25, 0.25, 0.35),
  c(0.25, 0.25, 0.25, 0.25)
)
lambda_results <- rbindlist(lapply(lambda_grid, function(lam) {
  predictor <- function(phrase, top_n = 3) predict_interpolated(phrase, lam, top_n)
  data.table(lambdas = paste(lam, collapse = "/"), dev_accuracy = top3_accuracy(predictor, dev_cases))
}))
print(lambda_results[order(-dev_accuracy)])
best_lambdas <- lambda_grid[[which.max(lambda_results$dev_accuracy)]]

# --- Compare Stupid Backoff vs. tuned interpolation on the dev set ----------
comparison <- data.table(
  model = c("Stupid Backoff", "Linear interpolation"),
  dev_accuracy = c(top3_accuracy(predict_backoff, dev_cases),
                    top3_accuracy(function(p, top_n = 3) predict_interpolated(p, best_lambdas, top_n), dev_cases))
)
print(comparison)

use_interpolation <- comparison$dev_accuracy[2] > comparison$dev_accuracy[1]
cat("\nSelected scoring strategy:", if (use_interpolation) "Linear interpolation" else "Stupid Backoff", "\n")

final_predict_next_word <- if (use_interpolation) {
  function(phrase, top_n = 3) predict_interpolated(phrase, best_lambdas, top_n)
} else {
  predict_backoff
}

# --- Perplexity (pseudo-perplexity: backoff/interpolation scores aren't a
# normalized probability distribution, so treat this as a relative diagnostic
# rather than a directly-comparable-across-model-families number) -----------
pseudo_perplexity <- function(cases, score_fn) {
  probs <- vapply(seq_len(nrow(cases)), function(i) {
    words <- str_split(clean_lines(cases$context[i]), " ")[[1]]
    max(score_fn(words, cases$truth[i]), 1e-10)  # floor to avoid log(0)
  }, numeric(1))
  exp(-mean(log(probs)))
}

# --- Final, one-time evaluation on the untouched test set -------------------
final_score_fn <- if (use_interpolation) {
  function(words, target_word) interpolated_score(words, target_word, best_lambdas)
} else {
  backoff_score
}
test_accuracy <- top3_accuracy(final_predict_next_word, test_cases)
test_perplexity <- pseudo_perplexity(test_cases, final_score_fn)

cat(sprintf("\n=== Final test-set results (%s) ===\n", if (use_interpolation) "interpolation" else "backoff"))
cat(sprintf("Top-3 accuracy: %.1f%%\n", 100 * test_accuracy))
cat(sprintf("Pseudo-perplexity: %.1f\n", test_perplexity))

model_sizes <- data.table(
  table = c("unigram", "bigram", "trigram", "fourgram"),
  n_rows = c(nrow(unigram_freq), nrow(bigram_tab), nrow(trigram_tab), nrow(fourgram_tab)),
  size_mb = c(object.size(unigram_freq), object.size(bigram_tab),
              object.size(trigram_tab), object.size(fourgram_tab)) / 1024^2
)
print(model_sizes)
cat("Total deployed model size (MB):", round(sum(model_sizes$size_mb), 1), "\n")

# quick manual sanity checks
print(final_predict_next_word("thanks for the"))
print(final_predict_next_word("i love"))
print(final_predict_next_word("this is a very unusual sequence of zzz"))

# --- Save everything the report / a future Shiny app would need ------------
saveRDS(unigram_freq, file.path(out_dir, "model_unigram.rds"))
saveRDS(bigram_tab, file.path(out_dir, "model_bigram.rds"))
saveRDS(trigram_tab, file.path(out_dir, "model_trigram.rds"))
saveRDS(fourgram_tab, file.path(out_dir, "model_fourgram.rds"))
saveRDS(total_unigrams, file.path(out_dir, "model_total_unigrams.rds"))
saveRDS(model_sizes, file.path(out_dir, "model_sizes.rds"))
saveRDS(grid_results, file.path(out_dir, "model_grid_results.rds"))
saveRDS(lambda_results, file.path(out_dir, "model_lambda_results.rds"))
saveRDS(comparison, file.path(out_dir, "model_comparison.rds"))
saveRDS(list(min_count = MIN_COUNT, top_k = TOP_K_PER_PREFIX, alpha = ALPHA,
             use_interpolation = use_interpolation, lambdas = best_lambdas),
        file.path(out_dir, "model_config.rds"))
saveRDS(list(top3_accuracy = test_accuracy, pseudo_perplexity = test_perplexity),
        file.path(out_dir, "model_eval_test.rds"))
