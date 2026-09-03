# wordprediction

Coursera Data Science Capstone: a next-word prediction Shiny app, built from
three large English text corpora (Twitter, blogs, news).

## Status

**Task 1 (this milestone):** data cleaning, exploratory analysis, and a basic
n-gram model with Stupid Backoff for unseen n-grams. See
[`report/milestone_report.qmd`](report/milestone_report.qmd) (rendered:
`report/milestone_report.html`) for the write-up.

The Shiny app itself is a later milestone and is not built yet.

## Data

Raw corpora (gitignored, not tracked in this repo) live in `data/`:

- `data/en_US.twitter.txt`
- `data/en_US.blogs.txt`
- `data/en_US.news.txt`

Each is one line of text per entry (tweet, blog post paragraph, or news
sentence/paragraph).

## Pipeline

Scripts in `R/` run in order; each caches its output to `data/processed/`
(gitignored) so later scripts don't need to re-read the raw files:

1. **`01_sample_data.R`** — reads the three raw files, reports full-corpus
   summary stats, and draws a random sample of ~1,000,000 lines total
   (allocated proportionally across sources). Splits the sample into an 80%
   training set, 10% dev set (for tuning), and 10% test set (touched once,
   for final reported numbers). Outputs: `raw_stats.rds`, `train_lines.rds`,
   `dev_lines.rds`, `test_lines.rds`.
2. **`02_tokenize_clean.R`** — defines `clean_lines()`, `tokenize_text()`, and
   `filter_profanity()`. Profanity list: `R/profanity_list.csv` (curated from
   the `lexicon` package's profanity word lists; no runtime dependency on
   `lexicon`). Sourced by the scripts below rather than run standalone.
3. **`03_eda.R`** — tokenizes + profanity-filters the training sample, and
   computes/caches the tables behind the report's plots and tables (word
   frequencies, coverage curve, bigram/trigram frequencies, per-source
   summary). Outputs: `tokens_train.rds`, `eda_*.rds`.
4. **`04_ngram_model.R`** — builds unigram/bigram/trigram/4-gram count
   tables; grid-searches pruning settings (`MIN_COUNT`, `TOP_K_PER_PREFIX`) on
   the dev set; compares Stupid Backoff against linear interpolation across
   n-gram orders (also tuned on the dev set) and keeps the winner; reports
   final top-3 accuracy and pseudo-perplexity on the untouched test set.
   Outputs: `model_*.rds`, including `model_config.rds` (chosen
   hyperparameters) and `model_eval_test.rds` (final metrics).

   Note: `object_size`/`gc()`-based memory checks showed the original
   n-gram builder (`merge()`-ing several full-size shifted copies of the
   token table) could spike R's memory to ~9GB for a ~21M-row token table.
   It was rewritten to use in-place, grouped `data.table::shift()` instead,
   roughly halving peak memory with identical output.

## Reproducing

```r
source("R/01_sample_data.R")
source("R/03_eda.R")       # sources 02_tokenize_clean.R itself
source("R/04_ngram_model.R")
```

Then render the report:

```sh
quarto render report/milestone_report.qmd
```
