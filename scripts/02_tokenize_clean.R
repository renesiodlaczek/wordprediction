# Step 2: Tokenization and profanity filtering.
#
# clean_lines(): normalizes raw text lines before tokenization (strip URLs,
#   @handles, hashtags, control chars; lowercase; collapse whitespace).
# tokenize_text(): splits cleaned lines into word tokens, one row per token,
#   tagged with the line id it came from (so n-grams don't cross line
#   boundaries). Keeps simple punctuation-free word tokens; numbers are
#   dropped since they are not useful for a next-*word* predictor.
# filter_profanity(): drops tokens (or whole rows) matching the static
#   profanity list in scripts/profanity_list.csv.

library(stringr)
library(data.table)

profanity_words <- read.csv("scripts/profanity_list.csv", stringsAsFactors = FALSE)$word

clean_lines <- function(lines) {
  lines |>
    str_to_lower() |>
    str_replace_all("https?://\\S+|www\\.\\S+", " ") |>   # URLs
    str_replace_all("@\\w+", " ") |>                       # @handles
    str_replace_all("#", " ") |>                            # hashtag marker (keep the word)
    str_replace_all("&amp;|&lt;|&gt;", " ") |>               # stray HTML entities
    str_replace_all("[^a-z0-9'\\s]", " ") |>                 # drop punctuation/emoji/symbols
    str_replace_all("\\s+", " ") |>
    str_trim()
}

tokenize_text <- function(lines) {
  cleaned <- clean_lines(lines)
  dt <- data.table(line_id = seq_along(cleaned), text = cleaned)
  dt <- dt[text != ""]
  tokens <- dt[, .(word = str_split(text, " ")[[1]]), by = line_id]
  tokens <- tokens[word != "" & !str_detect(word, "^[0-9]+$")]  # drop pure numbers
  tokens[]
}

filter_profanity <- function(tokens, profanity = profanity_words) {
  tokens[!(word %in% profanity)]
}
